_base_ = ["./regnet_6.4G_origin.py"]

work_dir = "work_dirs/regnet_3.2G"

custom_hooks = [
    dict(type='FisherPruningHook',
         pruning=False,
         deploy_from='path to the pruned model')
]

optimizer = dict(lr=0.1)
data = dict(samples_per_gpu = 256, workers_per_gpu=16)