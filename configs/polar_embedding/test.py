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
    samples_per_gpu=8,
    workers_per_gpu=4,
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
evaluation = dict(interval=1, metric='details', save_best='auto')
optimizer = dict(type='Adam', lr=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
runner = dict(type='EpochBasedRunner', max_epochs=150)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=50,
    warmup_ratio=0.3333333333333333,
    step=[110])
checkpoint_config = dict(interval=30)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
angle_version = 'oc'
model = dict(
    type='PolarEncodings',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='BBAVNeck', in_channels=[256, 512, 1024, 2048], out_channels=256),
    bbox_head=dict(
        type='PolarEncodingHead',
        in_channels=256,
        head_branches=[
            dict(
                type='hm',
                out_ch=1,
                loss=dict(
                    type='GaussianFocalLoss',
                    alpha=2.0,
                    gamma=4.0,
                    loss_weight=1)),
            dict(
                type='wh',
                out_ch=8,
                loss=dict(
                    type='IOUWeightedSmoothL1Loss', beta=1.0,
                    loss_weight=0.2)),
            dict(
                type='reg',
                out_ch=2,
                loss=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1))
        ],
        norm_cfg=None),
    train_cfg=None,
    test_cfg=dict(
        num_dets=500,
        conf_thr=0.18,
        version='oc',
        score_thr=0.1,
        nms_cfg=dict(type='rnms', iou_thr=0.05),
        max_per_img=100))
work_dir = '/media/slytheringe/Disk/Gejunyao/exp_results/mmdetection_files/SSDD/PolarEmbedding/exp2'
auto_resume = False
gpu_ids = range(0, 1)