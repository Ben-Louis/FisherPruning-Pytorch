import re
import numpy as np
import torch
import torch.nn.functional as F
import random


# These grad_fn pattern are flags of specific a nn.Module
CONV = ('ThnnConv2DBackward', 'CudnnConvolutionBackward')
FC = ('ThAddmmBackward', 'AddmmBackward', 'MmBackward')
BN = ('ThnnBatchNormBackward', 'CudnnBatchNormBackward')
# the modules which contains NON_PASS grad_fn need to change the parameter size
# according to channels after pruning
NON_PASS = CONV + FC

def feed_forward_once(model):
    inputs = torch.zeros(1, 3, 256, 256).cuda()
    inputs_meta = [{"img_shape": (256, 256, 3), "scale_factor": np.zeros(4, dtype=np.float32)}]
    neck_out = model.module.neck(model.module.backbone(inputs))

    if hasattr(model.module, "head"):
        # for classification models
        return model.module.head.fc(neck_out[-1]).sum()
    elif hasattr(model.module, "bbox_head"):
        # for one-stage detectors
        bbox_out = model.module.bbox_head(neck_out)
        return sum([sum([level.sum() for level in levels]) for levels in bbox_out])
    elif hasattr(model.module, "rpn_head") and hasattr(model.module, "roi_head"):
        # for two-stage detectors
        from mmdet.core import bbox2roi
        rpn_out = model.module.rpn_head(neck_out)
        proposals = model.module.rpn_head.get_bboxes(*rpn_out, inputs_meta)
        rois = bbox2roi(proposals)
        roi_out = model.module.roi_head._bbox_forward(neck_out, rois)
        loss = sum([sum([level.sum() for level in levels]) for levels in rpn_out])
        loss += roi_out['cls_score'].sum() + roi_out['bbox_pred'].sum()
        return loss
    else:
        raise NotImplementedError("This kind of model has not been supported yet.")



def traverse(op, op2parents, pattern=NON_PASS, max_pattern_layer=-1):
    """to get a dict which can describe the computer Graph,

    Args:
        op (grad_fn): as a root of DFS
        op2parents (dict): key is the grad_fn match the patter,and
            value is first grad_fn match NON_PASS when DFS from Key
        pattern (Tuple[str]): the patter of grad_fn to match
    """

    if op is not None:
        parents = op.next_functions
        if parents is not None:
            if match(op, pattern):
                if pattern is FC:
                    op2parents[op] = dfs(parents[1][0], [])
                else:
                    op2parents[op] = dfs(parents[0][0], [])
            if len(op2parents.keys()) == max_pattern_layer:
                return
            for parent in parents:
                parent = parent[0]
                if parent not in op2parents:
                    traverse(parent, op2parents, pattern, max_pattern_layer)


def dfs(op, visited):
    """DFS from a op,return all op when find a op match the patter
    NON_PASS.

    Args:
        op (grad_fn): the root of DFS
        visited (list[grad_fn]): contains all op has been visited

    Returns:
        list : all the ops  match the patter NON_PASS
    """

    ret = []
    if op is not None:
        visited.append(op)
        if match(op, NON_PASS):
            return [op]
        parents = op.next_functions
        if parents is not None:
            for parent in parents:
                parent = parent[0]
                if parent not in visited:
                    ret.extend(dfs(parent, visited))
    return ret


def match(op, op_to_match):
    """Match an operation to a group of operations; In pytorch graph, there
    may be an additional '0' or '1' (e.g. Addbackward1) after the ops
    listed above.

    Args:
        op (grad_fn): the grad_fn to match the pattern
        op_to_match (list[str]): the pattern need to match

    Returns:
        bool: return True when match the pattern else False
    """

    for to_match in op_to_match:
        if re.match(to_match + '[0-1]?$', type(op).__name__):
            return True
    return False


def get_channel_num(module, flag="in"):
    if type(module).__name__ == 'Conv2d':
        return getattr(module, f"{flag}_channels")
    elif type(module).__name__ == 'Linear':
        return getattr(module, f"{flag}_features")
    else:
        for attr in dir(module):
            if attr.startswith(f"{flag}_"):
                return getattr(module, attr)
    raise NotImplementedError(f"The module {type(module).__name__} has not been supported yet.")


def modified_forward_conv(self, feature):
    if not self.finetune and hasattr(self, "in_mask"):
        in_mask = self.in_mask.unsqueeze(1).expand(-1, feature.size(1) // self.in_mask.size(0))
        feature = feature * in_mask.reshape(1, -1, 1, 1)
    return F.conv2d(feature, self.weight, self.bias, self.stride,
                    self.padding, self.dilation, self.groups)


def modified_forward_linear(self, feature):
    if not self.finetune and hasattr(self, "in_mask"):
        in_mask = self.in_mask.unsqueeze(1).expand(-1, feature.size(1) // self.in_mask.size(0))
        feature = feature * in_mask.reshape(1, -1)
    return F.linear(feature, self.weight, self.bias)