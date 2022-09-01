_base_ = [
    '../_base_/datasets/ssdd_official.py', '../_base_/schedules/schedule_benchmark_150e.py',
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

img_norm_cfg = dict(
    mean=[21.55, 21.55, 21.55], std=[24.42, 24.42, 24.42], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(640, 640)),
    dict(
        type='RRandomFlip',
        flip_ratio=[0.25, 0.25, 0.25],
        direction=['horizontal', 'vertical', 'diagonal'],
        version=angle_version),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=None,
        version=angle_version),
    dict(type='RTranslate', prob=0.3, img_fill_val=0, level=3),
    dict(type='BrightnessTransform', level=3, prob=0.3),
    dict(type='ContrastTransform', level=3, prob=0.3),
    dict(type='EqualizeTransform', prob=0.3),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', pad_to_square=True),
    # dict(type='InstanceMaskGenerator'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', pad_to_square=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]


data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(version=angle_version,
               pipeline=train_pipeline),
    val=dict(version=angle_version,
            pipeline=test_pipeline),
    test=dict(version=angle_version,
            pipeline=test_pipeline))

work_dir = '/media/gejunyao/Disk/Gejunyao/exp_results/mmdetection_files/SSDD/BBAV/exp3'

optimizer = dict(type='Adam', lr=0.0001)
checkpoint_config = dict(interval=150)