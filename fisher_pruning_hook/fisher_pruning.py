import os.path as osp
import re
from collections import OrderedDict, defaultdict
from types import MethodType
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import HOOKS
from mmcv.runner import load_checkpoint, save_checkpoint
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import Hook
from torch.nn import Conv2d, Linear
from torch.nn.modules.batchnorm import _BatchNorm
from .utils import *


@HOOKS.register_module()
class FisherPruningHook(Hook):
    """Use fisher information to pruning the model, must register after
    optimizer hook.

    Args:
        pruning (bool): When True, the model in pruning process,
            when False, the model is in finetune process.
            Default: True
        delta (str): "acts" or "flops" or "none", normalize the fisher score by
            "acts" or flops.
            Default: "acts"
        batch_size (int): The batch_size when pruning model.
            Default: 2
        interval (int): The interval of  pruning two channels.
            Default: 10
        deploy_from (str): Path of checkpoint containing the structure
            of pruning model. Defaults to None and only effective
            when pruning is set True.
        save_flops_thr  (list): Checkpoint would be saved when
            the flops reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
        save_acts_thr (list): Checkpoint would be saved when
            the acts reached specific value in the list:
            Default: [0.75, 0.5, 0.25]
    """
    def __init__(
        self,
        pruning=True,
        delta='acts',
        batch_size=2,
        interval=10,
        internal_only=False,
        deploy_from=None,
        save_flops_thr=[0.75, 0.5, 0.25],
        save_acts_thr=[0.75, 0.5, 0.25],
    ):

        assert delta in ('acts', 'flops', 'none')
        self.pruning = pruning
        self.delta = delta
        self.interval = interval
        self.batch_size = batch_size
        # The key of self.input is conv module, and value of it
        # is list of conv' input_features in forward process
        self.nonpass_inputs = {}
        # The key of self.flops is conv module, and value of it
        # is the summation of conv's flops in forward process
        self.flops = {}
        # The key of self.acts is conv module, and value of it
        # is number of all the out feature's activations(N*C*H*W)
        # in forward process
        self.acts = {}
        # The key of self.temp_fisher_info is conv module, and value
        # is a temporary variable used to estimate fisher.
        self.temp_fisher_info = {}

        # The key of self.batch_fishers is conv module, and value
        # is the estimation of fisher by single batch.
        self.batch_fishers = {}

        # The key of self.accum_fishers is conv module, and value
        # is the estimation of parameter's fisher by all the batch
        # during number of self.interval iterations.
        self.accum_fishers = {}
        self.channels = 0
        self.delta = delta
        self.deploy_from = deploy_from
        self.internal_only = internal_only
        self.module_to_mask_length = defaultdict(dict)

        for i in range(len(save_acts_thr) - 1):
            assert save_acts_thr[i] > save_acts_thr[i + 1]
        for i in range(len(save_flops_thr) - 1):
            assert save_flops_thr[i] > save_flops_thr[i + 1]

        self.save_flops_thr = save_flops_thr
        self.save_acts_thr = save_acts_thr
        # if self.pruning:
        #     assert torch.__version__.startswith('1.3'), (
        #         'Due to the frequent changes of the autograd '
        #         'interface, we only guarantee it works well in pytorch==1.3.')

    def after_build_model(self, model, work_dir):
        """Remove all pruned channels in finetune stage.

        We add this function to ensure that this happens before DDP's
        optimizer's initialization
        """

        if not self.pruning:
            with open(osp.join(work_dir, "module_to_mask_length.json"), 'r') as f:
                module_to_mask_length = json.load(f)
            for name, module in model.named_modules():
                add_pruning_attrs(module, len_dict=module_to_mask_length.get(f"module.{name}", None))
            assert self.deploy_from, 'You have to give a ckpt' \
                'containing the structure information of the pruning model'
            load_checkpoint(model, self.deploy_from)
            deploy_pruning(model)



    def before_run(self, runner):
        """Initialize the relevant variables(fisher, flops and acts) for
        calculating the importance of the channel, and use the layer-grouping
        algorithm to make the coupled module shared the mask of input
        channel."""

        self.conv_names = OrderedDict()
        self.bn_names = OrderedDict()
        self.fc_names = OrderedDict()
        self.nonpass_names = OrderedDict()
        self.logger = runner.logger

        for n, m in runner.model.named_modules():
            m.finetune = not self.pruning
            if isinstance(m, Conv2d):
                m.name = n
                m.forward = MethodType(modified_forward_conv, m)
                self.conv_names[m] = n
                self.nonpass_names[m] = n
            if isinstance(m, _BatchNorm):
                m.name = n
                self.bn_names[m] = n
            if isinstance(m, Linear):
                m.name = n
                m.forward = MethodType(modified_forward_linear, m)
                self.fc_names[m] = n
                self.nonpass_names[m] = n

        model = runner.model

        if self.pruning:
            # divide the conv to several group and all convs in same
            # group used same input at least once in model's
            # forward process.
            model.eval()
            self.set_group_masks(model)
            for name, module in runner.model.named_modules():
                if hasattr(module, "in_mask"):
                    self.module_to_mask_length[name]["in"] = module.in_mask.size(0)
                if hasattr(module, "out_mask"):
                    self.module_to_mask_length[name]["out"] = module.out_mask.size(0)
            with open(osp.join(runner.work_dir, "module_to_mask_length.json"), 'w') as f:
                json.dump(self.module_to_mask_length, f)
            model.train()
            for module, name in self.nonpass_names.items():
                self.nonpass_inputs[module] = []
                self.temp_fisher_info[module] = module.weight.data.new_zeros(
                    self.batch_size, get_channel_num(module, "in"))
                self.accum_fishers[module] = module.weight.data.new_zeros(
                    get_channel_num(module, "in"))
            for group_id in self.groups:
                module = self.groups[group_id][0]
                self.temp_fisher_info[group_id] = module.weight.data.new_zeros(
                    self.batch_size, self.in_masks[group_id].size(0))
                self.accum_fishers[group_id] = module.weight.data.new_zeros(
                    self.in_masks[group_id].size(0))
            self.register_hooks()
            self.init_flops_acts()
            self.init_temp_fishers()

            if self.internal_only:
                self.ancest = {group: self.ancest[group] for group in self.groups if len(self.groups[group]) == 1}
                self.groups = {group: self.groups[group] for group in self.groups if len(self.groups[group]) == 1}
            self.nonpass_names_group = [[item.name for item in v] for idx, v in self.groups.items()]
            json.dump(self.nonpass_names_group, open(f"{runner.work_dir}/nonpass_names_group.json", 'w'), indent='')

        self.print_model(runner, print_flops_acts=False)

    def after_train_iter(self, runner):
        if not self.pruning:
            return
        self.group_fishers()
        if runner.world_size > 1:
            self.reduce_fishers()
        self.accumulate_fishers()
        self.init_temp_fishers()
        if self.every_n_iters(runner, self.interval):
            self.channel_prune()
            self.init_accum_fishers()
            self.print_model(runner, print_channel=False)
        self.init_flops_acts()

    @master_only
    def print_model(self, runner, print_flops_acts=True, print_channel=True):
        """Print the related information of the current model.

        Args:
            runner (Runner): Runner in mmcv
            print_flops_acts (bool): Print the remained percentage of
                flops and acts
            print_channel (bool): Print information about
                the number of reserved channels.
        """

        if print_flops_acts:
            flops, acts = self.compute_flops_acts()
            runner.logger.info('Flops: {:.2f}%, Acts: {:.2f}%'.format(
                flops * 100, acts * 100))
            if len(self.save_flops_thr):
                flops_thr = self.save_flops_thr[0]
                if flops < flops_thr:
                    self.save_flops_thr.pop(0)
                    path = osp.join(
                        runner.work_dir, 'flops_{:.0f}_acts_{:.0f}.pth'.format(
                            flops * 100, acts * 100))
                    save_checkpoint(runner.model, filename=path)
            if len(self.save_acts_thr):
                acts_thr = self.save_acts_thr[0]
                if acts < acts_thr:
                    self.save_acts_thr.pop(0)
                    path = osp.join(
                        runner.work_dir, 'acts_{:.0f}_flops_{:.0f}.pth'.format(
                            acts * 100, flops * 100))
                    save_checkpoint(runner.model, filename=path)
        if print_channel:
            for module, name in self.nonpass_names.items():
                runner.logger.info(f"{name}: channels [in|out]: "
                                   f"[{get_channel_num(module, 'in')}|{get_channel_num(module, 'out')}]; "
                                   f"in_mask: [{module.in_mask.sum().long().item()}|{module.in_mask.numel()}], "
                                   f"out_mask: [{module.out_mask.sum().long().item()}|{module.out_mask.numel()}]")

    def compute_flops_acts(self):
        """Computing the flops and activation remains."""
        flops = 0
        max_flops = 0
        acts = 0
        max_acts = 0
        for module, name in self.nonpass_names.items():
            max_flop = self.flops[module]
            i_mask = module.in_mask
            o_mask = module.out_mask
            flops += max_flop / (i_mask.numel() * o_mask.numel()) * (
                i_mask.sum() * o_mask.sum())
            max_flops += max_flop
            max_act = self.acts[module]
            acts += max_act / o_mask.numel() * o_mask.sum()
            max_acts += max_act
        return flops.cpu().numpy() / max_flops, acts.cpu().numpy() / max_acts



    def find_pruning_channel(self, module, fisher, in_mask, info):
        """Find the the channel of a model to pruning.

        Args:
            module (nn.Conv | int ): Conv module of model or idx of self.group
            fisher(Tensor): the fisher information of module's in_mask
            in_mask (Tensor): the squeeze in_mask of modules
            info (dict): store the channel of which module need to pruning
                module: the module has channel need to pruning
                channel: the index of channel need to pruning
                min : the value of fisher / delta

        Returns:
            dict: store the current least important channel
                module: the module has channel need to be pruned
                channel: the index of channel need be to pruned
                min : the value of fisher / delta
        """
        module_info = {}
        if fisher.sum() > 0 and in_mask.sum() > 0:
            nonzero = in_mask.nonzero().view(-1)
            fisher = fisher[nonzero]
            min_value, argmin = fisher.min(dim=0)
            if min_value < info['min']:
                module_info['module'] = module
                module_info['channel'] = nonzero[argmin]
                module_info['min'] = min_value
        return module_info

    def channel_prune(self):
        """Select the channel in model with smallest fisher / delta set corresponding in_mask 0."""

        info = {'module': None, 'channel': None, 'min': 1e9}
        for group in self.groups:
            in_mask = self.in_masks[group]
            fisher = self.accum_fishers[group].double()
            if self.delta == 'flops':
                fisher /= float(self.flops[group] / 1e9)
            elif self.delta == 'acts':
                fisher /= float(self.acts[group] / 1e6)
            info.update(self.find_pruning_channel(group, fisher, in_mask, info))
        group, channel = info['module'], info['channel']
        self.in_masks[group][channel] = 0.0

    def accumulate_fishers(self):
        """Accumulate all the fisher during self.interval iterations."""
        for group in self.groups:
            self.accum_fishers[group] += self.batch_fishers[group]

    def reduce_fishers(self):
        """Collect fisher from all rank."""
        for group in self.groups:
            dist.all_reduce(self.batch_fishers[group])

    def group_fishers(self):
        """Accumulate all module.in_mask's fisher and flops in same group."""
        for group in self.groups:
            self.flops[group] = 0
            self.acts[group] = 0
            in_channels = self.in_masks[group].size(0)
            for module in self.groups[group]:
                # compute fisher
                module_fisher = self.temp_fisher_info[module]
                module_fisher = module_fisher.view(module_fisher.size(0), in_channels, -1).sum(dim=2)
                self.temp_fisher_info[group] += module_fisher
                # compute flops (out)
                delta_flops = self.flops[module] / in_channels * module.out_mask.mean()
                self.flops[group] += delta_flops
            self.batch_fishers[group] = (self.temp_fisher_info[group] ** 2).sum(0)
            # compute flops (in) & acts
            for ancest_module in self.ancest[group]:
                delta_flops = self.flops[ancest_module] / in_channels * ancest_module.in_mask.mean()
                self.flops[group] += delta_flops
                acts = self.acts[ancest_module] // in_channels
                self.acts[group] += acts

    def init_flops_acts(self):
        """Clear the flops and acts of model in last iter."""
        for module, name in self.nonpass_names.items():
            self.flops[module] = 0
            self.acts[module] = 0

    def init_temp_fishers(self):
        """Clear fisher info of single conv and group."""
        for module, name in self.nonpass_names.items():
            self.temp_fisher_info[module].zero_()
        for group in self.groups:
            self.temp_fisher_info[group].zero_()

    def init_accum_fishers(self):
        """Clear accumulated fisher info."""
        for group in self.groups:
            self.accum_fishers[group].zero_()

    def register_hooks(self):
        """Register forward and backward hook to Conv module."""
        for module, name in self.nonpass_names.items():
            module.register_forward_hook(self.save_input_forward_hook)
            module.register_backward_hook(self.compute_fisher_backward_hook)

    def save_input_forward_hook(self, module, inputs, outputs):
        """Save the input and flops and acts for computing fisher and flops or
        acts.

        Args:
            module (nn.Module): the module of register hook
            inputs (tuple): input of module
            outputs (tuple): out of module
        """

        ic = get_channel_num(module, "in") // getattr(module, "groups", 1)
        kh, kw = getattr(module, "kernel_size", (1, 1))
        self.flops[module] += np.prod([ic, kh, kw, *outputs.shape])
        self.acts[module] += np.prod(outputs.shape)
        # a conv module may has several inputs in graph, for example head in Retinanet
        if inputs[0].requires_grad:
            self.nonpass_inputs[module].append(inputs)



    def compute_fisher_backward_hook(self, module, grad_input, *args):
        """
        Args:
            module (nn.Module): module register hooks
            grad_input (tuple): tuple contains grad of input and parameters,
                grad_input[0]is the grad of input in Pytorch 1.3, it seems
                has changed in Higher version
        """
        if module in self.nonpass_names:
            layer_name = type(module).__name__
            grad_feature = grad_input[0] if layer_name == "Conv2d" else grad_input[1]
            if grad_feature is not None:
                feature = self.nonpass_inputs[module].pop(-1)[0]
                # avoid that last batch is't full, but actually it's always full in mmdetection.
                grads = feature * grad_feature
                while grads.dim() > 2:
                    grads = grads.sum(dim=-1)
                grads = grads.view(self.batch_size, -1, grads.size(1)).sum(dim=1)
                self.temp_fisher_info[module][:grad_feature.size(0)] += grads


    def set_group_masks(self, model):
        """the modules(convolutions and BN) connect to same convolutions need
        change the out channels at same time when pruning, divide the modules
        into different groups according to the connection.

        Args:
            model(nn.Module): the model contains all modules
        """

        # step 1: establish computing graph
        loss = feed_forward_once(model)
        # step 2: find ancestors for each module
        self.conv2ancest = self.find_module_ancestors(loss, CONV)
        self.conv_link = {k.name: [item.name for item in v] for k, v in self.conv2ancest.items()}
        self.bn2ancest = self.find_module_ancestors(loss, BN)
        self.fc2ancest = self.find_module_ancestors(loss, FC, len(self.fc_names.keys()))
        self.fc_link = {k.name: [item.name for item in v] for k, v in self.fc2ancest.items()}
        self.nonpass2ancest = {**self.conv2ancest, **self.fc2ancest}
        self.nonpass_link = {**self.conv_link, **self.fc_link}
        loss.sum().backward()

        # step 3: construct groups via ancestor relationship
        self.make_groups()

        # step 4: set masks for each module
        # step 4.1: set in_masks/out_masks for all groups
        self.in_masks = {}
        for group, modules in self.groups.items():
            in_channels = float("inf")
            for module in modules:
                chn = get_channel_num(module, "in")
                chn /= module.groups if module in self.conv_names else 1
                in_channels = min(in_channels, chn)
            self.in_masks[group] = module.weight.new_ones((int(in_channels), ))
            for module in modules:
                module.register_buffer("in_mask", self.in_masks[group])
            for module in self.ancest[group]:
                module.register_buffer("out_mask", self.in_masks[group])
        # step 4.2: set out_masks for output modules
        for module in self.nonpass_names:
            if not hasattr(module, "out_mask"):
                out_channels = get_channel_num(module, "out")
                out_channels /= module.groups if module in self.conv_names else 1
                module.register_buffer("out_mask", module.weight.new_ones((int(out_channels), )))
        # step 4.3: set out_masks for BNs
        for bn in self.bn_names:
            module = self.bn2ancest[bn][0]
            bn.register_buffer("out_mask", module.out_mask)


    def find_module_ancestors(self, loss, pattern, max_pattern_layer=-1):
        """find the nearest Convolution of the module
        matching the pattern
        Args:
            loss(Tensor): the output of the network
            pattern(Tuple[str]): the pattern name

        Returns:
            dict: the key is the module match the pattern(Conv or Fc),
             and value is the list of it's nearest ancestor Convolution
        """

        # key is the op (indicate a Conv or Fc) and value is a list
        # contains all the nearest ops (indicate a Conv or Fc)
        op2parents = {}
        traverse(loss.grad_fn, op2parents, pattern, max_pattern_layer)

        var2module = {}
        if pattern is BN:
            module_names = self.bn_names
        elif pattern is CONV:
            module_names = self.conv_names
        else:
            module_names = self.fc_names

        if pattern is FC:
            for module, name in module_names.items():
                var2module[id(module.bias)] = module
        else:
            for module, name in module_names.items():
                var2module[id(module.weight)] = module

        # same module may appear several times in computing graph,
        # so same module can correspond to several op, for example,
        # different feature pyramid level share heads.
        # op2module select one op as the flag of module.
        op2module = {}
        for op, parents in op2parents.items():
            # TODO bfs to get variable
            if pattern is FC:
                var_id = id(op.next_functions[0][0].variable)
            else:
                var_id = id(op.next_functions[1][0].variable)
            module = var2module[var_id]
            exist = False
            # may several op link to same module
            for temp_op, temp_module in op2module.items():
                # temp_op(has visited in loop) and op
                # link to same module, so their should share
                # all parents, so we need extend the value of
                # op to value of temp_op
                if temp_module is module:
                    op2parents[temp_op].extend(op2parents[op])
                    exist = True
                    break
            if not exist:
                op2module[op] = module

        if not hasattr(self, 'nonpass_module'):
            # save for find bn's ancestor convolutions
            self.nonpass_module = op2module
        else:
            self.nonpass_module.update(op2module)
        return {
            module: [
                self.nonpass_module[parent] for parent in op2parents[op]
                if parent in self.nonpass_module
            ]
            for op, module in op2module.items()
        }


    def make_groups(self):
        """The modules (convolutions and BNs) connected to the same conv need
        to change the channels simultaneously when pruning.

        This function divides all modules into different groups according to
        the connections.
        """

        idx = -1
        groups, groups_ancest = {}, {}
        for module, name in reversed(self.nonpass_names.items()):
            added = False
            for group in groups:
                module_ancest = set(self.nonpass2ancest[module])
                group_ancest = set(groups_ancest[group])
                group_gconvs = set([m for m in groups[group] if m in self.conv_names and m.groups > 1])
                if (len(module_ancest.intersection(group_ancest.union(group_gconvs))) > 0) or \
                    (module in self.conv_names and module.groups > 1 and module in group_ancest):
                    groups[group].append(module)
                    groups_ancest[group] = list(module_ancest.union(group_ancest))
                    added = True
                    break
            if not added:
                idx += 1
                groups[idx] = [module]
                groups_ancest[idx] = self.nonpass2ancest[module]
        # key is the ids the group, and value contains all conv
        # of this group
        self.groups = {}
        # key is the ids the group, and value contains all nearest
        # ancestor of this group
        self.ancest = {}
        self.module2group = {}
        idx = 0
        for group in groups:
            modules = groups[group]
            self.groups[idx] = modules
            for m in modules:
                self.module2group[m] = idx
            self.ancest[idx] = groups_ancest[group]
            idx += 1



def add_pruning_attrs(module, pruning=False, len_dict=None):
    """When module is conv, add `finetune` attribute, register `mask` buffer
    and change the origin `forward` function. When module is BN, add `out_mask`
    attribute to module.

    Args:
        conv (nn.Conv2d):  The instance of `torch.nn.Conv2d`
        pruning (bool): Indicating the state of model which
            will make conv's forward behave differently.
    """
    # TODO: mask  change to bool
    if type(module).__name__ == 'Conv2d':
        module.register_buffer('in_mask', module.weight.new_ones(len_dict["in"]))
        module.register_buffer('out_mask', module.weight.new_ones(len_dict["out"]))
        module.finetune = not pruning

    if 'BatchNorm' in type(module).__name__:
        module.register_buffer('out_mask', module.weight.new_ones(len_dict["out"]))

    if type(module).__name__ == "Linear":
        module.register_buffer('in_mask', module.weight.new_ones(len_dict["in"]))
        module.register_buffer('out_mask', module.weight.new_ones(len_dict["out"]))
        module.finetune = not pruning



def deploy_pruning(model):
    """To speed up the finetune process, We change the shape of parameter
    according to the `in_mask` and `out_mask` in it."""

    for name, module in model.named_modules():
        if type(module).__name__ == 'Conv2d':
            module.finetune = True
            requires_grad = module.weight.requires_grad
            out_mask = module.out_mask.bool()
            out_mask = out_mask.unsqueeze(1).expand(-1, module.out_channels // out_mask.size(0)).view(-1)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[out_mask], requires_grad=requires_grad)
            temp_weight = module.weight.data[out_mask.bool()]
            in_mask = module.in_mask.bool()
            in_mask = in_mask.unsqueeze(1).expand(-1, module.in_channels // in_mask.size(0)).view(-1)
            module.weight = nn.Parameter(temp_weight[:, in_mask].data, requires_grad=requires_grad)

            module.in_channels = int(in_mask.sum())
            module.out_channels = int(out_mask.sum())
            if module.groups > 1:
                module.groups = int(module.groups * in_mask.sum() // in_mask.numel())

        elif type(module).__name__ == 'Linear':
            module.finetune = True
            requires_grad = module.weight.requires_grad
            out_mask = module.out_mask.bool()
            out_mask = out_mask.unsqueeze(1).expand(-1, module.out_features // out_mask.size(0)).view(-1)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(module.bias.data[out_mask], requires_grad=requires_grad)
            temp_weight = module.weight.data[out_mask.bool()]
            in_mask = module.in_mask.bool()
            in_mask = in_mask.unsqueeze(1).expand(-1, module.in_features // in_mask.size(0)).view(-1)
            module.weight = nn.Parameter(temp_weight[:, in_mask].data, requires_grad=requires_grad)

            module.in_features = int(in_mask.sum())
            module.out_features = int(out_mask.sum())

        elif 'BatchNorm2d' in type(module).__name__:
            out_mask = module.out_mask.bool()
            out_mask = out_mask.unsqueeze(1).expand(-1, module.weight.size(0) // out_mask.size(0)).view(-1)
            requires_grad = module.weight.requires_grad
            module.weight = nn.Parameter(module.weight.data[out_mask].data, requires_grad=requires_grad)
            module.bias = nn.Parameter(module.bias.data[out_mask].data, requires_grad=requires_grad)
            module.running_mean = module.running_mean[out_mask]
            module.running_var = module.running_var[out_mask]



