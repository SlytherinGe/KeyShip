# dataset settings
dataset_type = 'SSDDDataset'
data_root = '/media/gejunyao/Disk1/Datasets/SSDD/'
img_norm_cfg = dict(
    mean=[21.55, 21.55, 21.55], std=[24.42, 24.42, 24.42], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(640, 640)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
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
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=12,
    workers_per_gpu=16,
    train=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'VOC2012/ImageSets/Main/trainval.txt',
            img_prefix=data_root + 'VOC2012/',
            pipeline=train_pipeline),
    val=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'VOC2012/ImageSets/Main/test.txt',
            img_prefix=data_root + 'VOC2012/',
            pipeline=test_pipeline),
    test=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + 'VOC2012/ImageSets/Main/test.txt',
            img_prefix=data_root + 'VOC2012/',
            pipeline=test_pipeline))
