dataset_type = 'RSDDDataset'
data_root = './data/RSDD-SAR/'
angle_version = 'oc'
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
    train=dict(
            version=angle_version,
            type=dataset_type,
            ann_file=data_root + 'ImageSets/train.txt',
            img_prefix=data_root + '',
            pipeline=train_pipeline),
    val=dict(
            version=angle_version,
            type=dataset_type,
            ann_file=data_root  + 'ImageSets/test.txt',
            img_prefix=data_root + '',
            pipeline=test_pipeline),
    test=dict(
            version=angle_version,
            type=dataset_type,
            ann_file=data_root  + 'ImageSets/test.txt',
            img_prefix=data_root + '',
            pipeline=test_pipeline))