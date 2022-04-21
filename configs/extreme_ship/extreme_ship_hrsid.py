_base_ = [
    '../_base_/datasets/my_hrsid.py'
]

BASE_CONV_SETTING = [('conv',     ('default', 256)),
                    ('conv',     ('default', 256))]
OFFSET_TYPE = ['sc', 'lc']
NUM_CLASS=1
NUM_CENTRI_CH=2
INF = 1e8
angle_version = 'oc'
# model settings
model = dict(
    type='ExtremeShip',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='ExtremeHeadV2',
        num_classes=1,
        in_channels=256,
        longside_center_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CLASS))],
        shortside_center_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CLASS))],
        target_center_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CLASS))],
        offset_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     len(OFFSET_TYPE) * 2))],
        centipital_shift_cfg = BASE_CONV_SETTING + \
                            [('conv',     ('out',     NUM_CENTRI_CH))],
        centipital_shift_channels=NUM_CENTRI_CH,
        regress_ratio=((-1, 2),(-1, 2)),
        offset_types = OFFSET_TYPE,
        loss_heatmap=dict(
            type='GaussianFocalLoss',
            alpha=2.0,
            gamma=4.0,
            loss_weight=1                     
        ),
        loss_offsets=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=1
        ),
        loss_centripetal_shift=dict(
            type='SmoothL1Loss', beta=1.0, loss_weight=0.1),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg = dict(
        cache_cfg = dict(
            root = '/home/gejunyao/ramdisk/TrainCache',
            save_target=False,
            save_output=False
        ),
        gaussioan_sigma_ratio = (0.1, 0.1)
    ),
    test_cfg = dict(
        cache_cfg = dict(
            root = '/home/gejunyao/ramdisk/TestCache'
        ),
        num_kpts_per_lvl = [0,60],
        num_dets_per_lvl = [0,60],
        ec_conf_thr = 0.01,
        tc_conf_thr = 0.1,
        valid_size_range = [(-1,0), (-1, 2),],
        score_thr = 0.1,
        nms_cfg = dict(type='rnms', iou_thr=0.05),
        # nms_cfg = dict(type='soft_rnms', sigma=0.1, min_score=0.3),
        max_per_img=100
    ))

img_norm_cfg = dict(
    mean=[21.55, 21.55, 21.55], std=[24.42, 24.42, 24.42], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
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
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size=(800, 800)),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=4,
    workers_per_gpu=16,
    train=dict(version=angle_version,
               pipeline=train_pipeline),
    val=dict(version=angle_version,
            pipeline=test_pipeline),
    test=dict(version=angle_version,
            pipeline=test_pipeline))


log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
# yapf:enable

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

work_dir = '/media/gejunyao/Disk/Gejunyao/exp_results/mmdetection_files/HRSID/ExtremeShip/exp07/'

# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto')
# optimizer
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cyclic',
    warmup=None,
    cyclic_times=1,
    target_ratio=(10, 1e-2))
runner = dict(type='EpochBasedRunner', max_epochs=80)
checkpoint_config = dict(interval=1)
