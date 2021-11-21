import argparse
import mmcv
import numpy as np
import torch
from mmcls.models import build_classifier
import time
from fisher_pruning_hook import FisherPruningHook
from torch.nn import Conv2d, Linear
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.activation import ReLU
from functools import partial

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('config', help='test config file path')

    parser.add_argument(
        '--device',
        choices=['cpu', 'cuda'],
        default='cuda',
        help='device used for testing')
    args = parser.parse_args()

    return args

def speed_test(model, device, batchsize, iterations):
    x = torch.randn(batchsize, 3, 224, 224).to(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        start = time.time()
        for _ in range(iterations):
            _ = model(x)
        mid = time.time()
        for _ in range(iterations):
            _ = model(x)
        end = time.time()
    return start, mid, end

def compute_parameters(model):
    params = sum(p.numel() for p in model.parameters())
    return params

class FlopsActsHook:
    def __init__(self, model):
        self.flops = {}
        self.acts = {}
        self.non_registered = []
        for n, m in model.named_modules():
            self.flops[n] = 0
            self.acts[n] = 0

            if isinstance(m, Conv2d): m.register_forward_hook(self.forward_hook_conv)
            elif isinstance(m, Linear): m.register_forward_hook(self.forward_hook_fc)
            elif isinstance(m, _BatchNorm): m.register_forward_hook(self.forward_hook_bn)
            elif isinstance(m, ReLU): m.register_forward_hook(self.forward_hook_relu)
            else:
                # print(n, type(m))
                self.non_registered.append([n, type(m)])

    def forward_hook_conv(self, module, inputs, outputs):
        ic = module.in_channels // module.groups
        kh, kw = module.kernel_size
        self.flops[module.name] += np.prod([ic, kh, kw, *outputs.shape])
        if module.bias is not None:
            self.flops[module.name] += np.prod(outputs.shape)
        self.acts[module.name] += np.prod(outputs.shape)

    def forward_hook_fc(self, module, inputs, outputs):
        ic = module.in_features
        self.flops[module.name] += np.prod([ic, *outputs.shape])
        if module.bias is not None:
            self.flops[module.name] += np.prod(outputs.shape)
        self.acts[module.name] += np.prod(outputs.shape)

    def forward_hook_bn(self, module, inputs, outputs):
        self.flops[module.name] += np.prod(outputs.shape) * (4 if module.affine else 2)
        self.acts[module.name] += np.prod(outputs.shape)

    def forward_hook_relu(self, module, inputs, outputs):
        self.flops[module.name] += np.prod(outputs.shape)
        self.acts[module.name] += 0 if module.inplace else np.prod(outputs.shape)

    def init(self):
        for n in self.flops:
            self.flops[n] = 0
            self.acts[n] = 0

    def summarize(self):
        flops, acts = 0, 0
        for n in self.flops:
            flops += self.flops[n]
            acts += self.acts[n]
        return flops, acts

def compute_flops_acts(model, device):
    model.eval()
    model.to(device)
    hook = FlopsActsHook(model)
    hook.init()
    x = torch.randn(32, 3, 224, 224).to(device)
    _ = model(x)
    flops, acts = hook.summarize()
    return flops / x.size(0), acts / x.size(0)

def compute_flops_params_thop(model, device):
    from thop import profile
    model.eval()
    model.to(device)
    x = torch.randn(32, 3, 224, 224).to(device)
    flops, params = profile(model, inputs=(x,))
    return flops / x.size(0), params

def main():
    args = parse_args()
    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.device= args.device

    # build the model
    model = build_classifier(cfg.model)
    model.forward = partial(model.forward, return_loss=False, img_metas=None)
    for n, m in model.named_modules():
        m.name = n

    if 'custom_hooks' in cfg:
        for hook in cfg.custom_hooks:
            if hook.type.startswith('FisherPruningHook'):
                hook_cfg = hook.copy()
                hook_cfg.pop('priority', None)
                from mmcv.runner.hooks import HOOKS
                hook_cls = HOOKS.get(hook_cfg['type'])
                if hasattr(hook_cls, 'after_build_model'):
                    pruning_hook = mmcv.build_from_cfg(hook_cfg, HOOKS)
                    pruning_hook.after_build_model(model, cfg.work_dir)

    # test speed
    batchsize, iterations = (16, 50) if cfg.device == "cpu" else (64, 100)
    start, mid, end = speed_test(model, cfg.device, batchsize, iterations)
    print(f"time elapse for each iteration with batchsize {batchsize}:")
    print(f"first {iterations} iterations: {(mid - start) * 1000 / iterations:.3f}ms")
    print(f"last {iterations} iterations:  {(end - mid) * 1000 / iterations:.3f}ms")

    # flops and acts
    flops, acts = compute_flops_acts(model, cfg.device)
    print(f"flops:  {flops / (10 ** 9):.3f}G")
    print(f"memory: {acts / (10 ** 6):.3f}M")
    params = compute_parameters(model)
    print(f"params: {params / (10 ** 6):.3f}M")


if __name__ == "__main__":
    main()