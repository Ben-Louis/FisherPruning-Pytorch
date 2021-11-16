_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# model settings
model = dict(
    type='FSAF',
    backbone=dict(frozen_stages=-1, ),
    bbox_head=dict(
        type='FSAFHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        reg_decoded_bbox=True,
        # Only anchor-free branch is implemented. The anchor generator only
        #  generates 1 anchor at each feature point, as a substitute of the
        #  grid of features.
        anchor_generator=dict(type='AnchorGenerator',
                              octave_base_scale=1,
                              scales_per_octave=1,
                              ratios=[1.0],
                              strides=[8, 16, 32, 64, 128]),
        bbox_coder=dict(_delete_=True, type='TBLRBBoxCoder', normalizer=4.0),
        loss_cls=dict(type='FocalLoss',
                      use_sigmoid=True,
                      gamma=2.0,
                      alpha=0.25,
                      loss_weight=1.0,
                      reduction='none'),
        loss_bbox=dict(_delete_=True,
                       type='IoULoss',
                       eps=1e-6,
                       loss_weight=1.0,
                       reduction='none')),
    # training and testing settings
    train_cfg=dict(assigner=dict(_delete_=True,
                                 type='CenterRegionAssigner',
                                 pos_scale=0.2,
                                 neg_scale=0.2,
                                 min_pos_iof=0.01),
                   allowed_border=-1,
                   pos_weight=-1,
                   debug=False))
optimizer = dict(type='SGD', lr=0.002, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True,
                        grad_clip=dict(max_norm=10, norm_type=2))

custom_hooks = [
    dict(
        type='FisherPruningHook',
        # In pruning process, you need set priority
        # as 'LOWEST' to insure the pruning_hook is excused
        # after optimizer_hook, in fintune process, you
        # should set it as 'HIGHEST' to insure it excused
        # before checkpoint_hook
        pruning=True,
        batch_size=2,
        interval=10,
        priority='LOWEST',
    )
]
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fsaf/fsaf_r50_fpn_1x_coco/fsaf_r50_fpn_1x_coco-94ccc51f.pth'  # noqa: E501
work_dir = "work_dirs/fsaf"
