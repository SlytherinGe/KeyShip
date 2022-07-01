dataset_type = 'SSDDDatasetOfficial'
data_root = './data/Official-SSDD-OPEN/RBox_SSDD/voc_style/'
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
        version='oc'),
    dict(
        type='PolyRandomRotate',
        rotate_ratio=0.5,
        angles_range=180,
        auto_bound=False,
        rect_classes=None,
        version='oc'),
    dict(type='RTranslate', prob=0.3, img_fill_val=0, level=3),
    dict(type='BrightnessTransform', level=3, prob=0.3),
    dict(type='ContrastTransform', level=3, prob=0.3),
    dict(type='EqualizeTransform', prob=0.3),
    dict(
        type='Normalize',
        mean=[21.55, 21.55, 21.55],
        std=[24.42, 24.42, 24.42],
        to_rgb=True),
    dict(type='Pad', pad_to_square=True),
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
            dict(
                type='Normalize',
                mean=[21.55, 21.55, 21.55],
                std=[24.42, 24.42, 24.42],
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='SSDDDatasetOfficial',
        ann_file=
        './data/Official-SSDD-OPEN/RBox_SSDD/voc_style/ImageSets/Main/train.txt',
        img_prefix='./data/Official-SSDD-OPEN/RBox_SSDD/voc_style/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='RResize', img_scale=(640, 640)),
            dict(
                type='RRandomFlip',
                flip_ratio=[0.25, 0.25, 0.25],
                direction=['horizontal', 'vertical', 'diagonal'],
                version='oc'),
            dict(
                type='PolyRandomRotate',
                rotate_ratio=0.5,
                angles_range=180,
                auto_bound=False,
                rect_classes=None,
                version='oc'),
            dict(type='RTranslate', prob=0.3, img_fill_val=0, level=3),
            dict(type='BrightnessTransform', level=3, prob=0.3),
            dict(type='ContrastTransform', level=3, prob=0.3),
            dict(type='EqualizeTransform', prob=0.3),
            dict(
                type='Normalize',
                mean=[21.55, 21.55, 21.55],
                std=[24.42, 24.42, 24.42],
                to_rgb=True),
            dict(type='Pad', pad_to_square=True),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
        ],
        version='oc'),
    val=dict(
        type='SSDDDatasetOfficial',
        ann_file=
        './data/Official-SSDD-OPEN/RBox_SSDD/voc_style/ImageSets/Main/test.txt',
        img_prefix='./data/Official-SSDD-OPEN/RBox_SSDD/voc_style/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[21.55, 21.55, 21.55],
                        std=[24.42, 24.42, 24.42],
                        to_rgb=True),
                    dict(type='Pad', pad_to_square=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='oc'),
    test=dict(
        type='SSDDDatasetOfficial',
        ann_file=
        './data/Official-SSDD-OPEN/RBox_SSDD/voc_style/ImageSets/Main/test.txt',
        img_prefix='./data/Official-SSDD-OPEN/RBox_SSDD/voc_style/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='RResize'),
                    dict(
                        type='Normalize',
                        mean=[21.55, 21.55, 21.55],
                        std=[24.42, 24.42, 24.42],
                        to_rgb=True),
                    dict(type='Pad', pad_to_square=True),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ],
        version='oc'))
BASE_CONV_SETTING = [('conv', ('default', 256)), ('conv', ('default', 256))]
NUM_CLASS = 1
INF = 100000000.0
angle_version = 'oc'
model = dict(
    type='ExtremeShip',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='SyncBN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='ExtremeHeadV4',
        num_classes=1,
        in_channels=256,
        longside_center_cfg=[('conv', ('default', 256)),
                             ('conv', ('default', 256)), ('conv', ('out', 1))],
        shortside_center_cfg=[('conv', ('default', 256)),
                              ('conv', ('default', 256)),
                              ('conv', ('out', 1))],
        target_center_cfg=[('conv', ('default', 256)),
                           ('conv', ('default', 256)), ('conv', ('out', 1))],
        center_pointer_cfg=[('conv', ('out', 8))],
        ec_offset_cfg=[('conv', ('out', 2))],
        regress_ratio=((-1, 2), (-1, 2)),
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1.0),
        loss_pointer=dict(type='SmoothL1Loss', beta=0.125, loss_weight=0.05),
        loss_offsets=dict(type='SmoothL1Loss', beta=0.125, loss_weight=0.1),
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)),
    train_cfg=dict(gaussioan_sigma_ratio=(0.1, 0.1)),
    test_cfg=dict(
        cache_cfg=None,
        num_kpts_per_lvl=[0, 150],
        num_dets_per_lvl=[0, 60],
        ec_conf_thr=0.01,
        tc_conf_thr=0.1,
        sc_ptr_sigma=0.01,
        lc_ptr_sigma=0.01,
        valid_size_range=[(-1, 0), (-1, 2)],
        score_thr=0.05,
        nms=dict(type='rnms', iou_thr=0.2),
        max_per_img=100))
log_config = dict(
    interval=10,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
work_dir = '../exp_results/mmlab_results/ssdd/benchmark/extreme_ship_210e.py'
evaluation = dict(interval=1, metric='details', save_best='auto')
optimizer = dict(type='Adam', lr=0.0006)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=210)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.3333333333333333,
    step=[150, 200])
checkpoint_config = dict(interval=21)
auto_resume = False
gpu_ids = range(0, 2)
