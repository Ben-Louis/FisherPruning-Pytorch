_base_ = ["./regnet_3.2G_origin.py"]

work_dir = "work_dirs/regnet_1.6G_pruning"
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

data = dict(samples_per_gpu=32, workers_per_gpu=2)