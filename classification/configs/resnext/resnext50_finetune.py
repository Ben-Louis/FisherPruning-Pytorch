_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

custom_hooks = [
    dict(type='FisherPruningHook',
         pruning=False,
         deploy_from='path to the pruned model')
]

work_dir = "work_dirs/resnext50"
optimizer = dict(lr=0.1)
data = dict(samples_per_gpu = 256, workers_per_gpu=16) # for single GPU
