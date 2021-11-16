_base_ = [
    '../_base_/models/resnext50_32x4d.py',
    '../_base_/datasets/imagenet_bs32_pil_resize.py',
    '../_base_/schedules/imagenet_bs256.py', '../_base_/default_runtime.py'
]

optimizer = dict(lr=0.004)

custom_hooks = [
    dict(
        type='FisherPruningHook',
        # In pruning process, you need set priority
        # as 'LOWEST' to insure the pruning_hook is excused
        # after optimizer_hook, in fintune process, you
        # should set it as 'HIGHEST' to insure it excused
        # before checkpoint_hook
        pruning=True,
        batch_size=32,
        interval=10,
        priority='LOWEST',
    )
]
work_dir = "work_dirs/resnext50"
load_from = "https://download.openmmlab.com/mmclassification/v0/resnext/resnext50_32x4d_b32x8_imagenet_20210429-56066e27.pth"
