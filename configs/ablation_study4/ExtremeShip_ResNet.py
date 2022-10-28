_base_ = [
    '../_base_/datasets/ssdd_official.py', '../_base_/schedules/schedule_benchmark_150e.py',
    '../_base_/benchmark_runtime.py'
]

BASE_CONV_SETTING = [('conv',     ('LReLU', 256)),
                    ('conv',     ('LReLU', 256))]
NUM_CLASS=1
INF = 1e8
angle_version = 'le90'
# model settings
model = dict(
    type='ExtremeShip',
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    bbox_head=dict(
        type='ExtremeHeadV4',
        num_classes=NUM_CLASS,
        in_channels=256,
        longside_center_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CLASS))],
        shortside_center_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CLASS))],
        target_center_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CLASS))],
        center_pointer_cfg = [('conv',     ('out',     8))],
        ec_offset_cfg = [('conv',     ('out',     2))],
        regress_ratio=((-1, 2),(0, 0),(0, 0),(0, 0),(0, 0)),
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=1.0               
        ),
        loss_pointer=dict(
            type='SmoothL1Loss', beta=1/8, loss_weight=0.05
        ),
        loss_offsets=dict(
            type='SmoothL1Loss', beta=1/8, loss_weight=0.1
        ),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg = dict(
        gaussioan_sigma_ratio = (0.25, 0.25)
    ),
    test_cfg = dict(
        # cache_cfg = dict(
        #     root = '/home/slytheringe/FastCache',
        # ),
        cache_cfg = None,
        num_kpts_per_lvl = [150],
        num_dets_per_lvl = [60],
        ec_conf_thr = 0.01,
        tc_conf_thr = 0.1,
        sc_ptr_sigma = 0.01,
        lc_ptr_sigma = 0.01,
        valid_size_range = [(-1, 2)],
        score_thr = 0.05,
        nms = dict(type='rnms', iou_thr=0.20),
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
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(version=angle_version,
               pipeline=train_pipeline),
    val=dict(version=angle_version,
            pipeline=test_pipeline),
    test=dict(version=angle_version,
            pipeline=test_pipeline))
# optimizer = dict(type='Adam', lr=0.0003)
# work_dir = '/media/slytheringe/Disk/Gejunyao/exp_results/mmdetection_files/SSDD/ExtremeShipV4/exp29'
# load_from = '/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Gejunyao/utils/centripetalnet_hourglass104_mstest_16x6_210e_coco_20200915_204804-3ccc61e5.pth'
work_dir = '../exp_results/mmlab_results/ssdd/ablation_study4/extreme_ship_resnet'