_base_ = [
    '../_base_/datasets/benchmark_ssdd.py', '../_base_/schedules/schedule_benchmark_150e.py',
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
        in_channels=256,
        head_branches=[dict(type='hm', 
                    out_ch=1,
                    loss=dict(
                        type='GaussianFocalLoss',
                        alpha=2.0,
                        gamma=4.0,
                        loss_weight=1)),
                dict(type='wh', 
                    out_ch=10,
                    loss=dict(
                        type='SmoothL1Loss', 
                        beta=1.0, 
                        loss_weight=1)),
                dict(type='reg', 
                    out_ch=2,
                    loss=dict(
                        type='SmoothL1Loss', 
                        beta=1.0, 
                        loss_weight=1)),
                dict(type='cls_theta', 
                    out_ch=1,
                    loss=dict(
                        type='CrossEntropyLoss', 
                        use_sigmoid=True, 
                        loss_weight=1))],
        norm_cfg=None),
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
work_dir = '../exp_results/mmlab_results/ssdd/ablation_study4/bbav'