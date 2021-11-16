_base_ = [
    '../_base_/models/resnet50.py', '../_base_/datasets/imagenet_bs32.py',
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
        interval=25,
        priority='LOWEST',
    )
]

work_dir = "work_dirs/resnet50"
load_from = "https://download.openmmlab.com/mmclassification/v0/resnet/resnet50_8xb32_in1k_20210831-ea4938fc.pth"
