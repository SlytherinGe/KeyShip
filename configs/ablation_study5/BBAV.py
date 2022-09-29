_base_ = [
    '../_base_/datasets/benchmark_rsdd.py', '../_base_/schedules/schedule_benchmark_150e.py',
    '../_base_/benchmark_runtime.py'
]

angle_version = 'oc'
# model settings
model = dict(
    type='BBAV',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='BBAVNeck',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256
    ),
    bbox_head=dict(
        type='BBAVHead',
        num_classes=1,
        in_channels=256,
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=1                               
        ),
        loss_offset=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1
        ),
        loss_rbox=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1
        ),
        loss_theta=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1
        ),
    norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg = None,
    test_cfg = dict(
        num_dets = 500,
        conf_thr = 0.18,
        version = angle_version,
        score_thr = 0.1,
        nms_cfg = dict(type='rnms', iou_thr=0.05),
        max_per_img=100
    ))
optimizer = dict(type='Adam', lr=0.000125)
work_dir = '../exp_results/mmlab_results/ssdd/ablation_study5/bbav'