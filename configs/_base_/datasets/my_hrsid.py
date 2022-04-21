# dataset settings
dataset_type = 'HRSIDDataset'
# dataset root path:
data_root = '/media/gejunyao/Disk1/Datasets/HRSID/'

img_norm_cfg = dict(
    mean=[21.55, 21.55, 21.55], std=[24.42, 24.42, 24.42], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(800, 800)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=(800, 800)),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_bboxes_ignore']),
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
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=16,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'annotations/train2017.json',
            img_prefix=data_root + 'images',
            pipeline=train_pipeline),
    val=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'annotations/test2017.json',
            img_prefix=data_root + 'images',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'annotations/test2017.json',
            img_prefix=data_root + 'images',
            pipeline=test_pipeline))