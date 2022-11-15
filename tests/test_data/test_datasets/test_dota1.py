import mmcv
from mmcv import Config, DictAction
from mmdet.datasets.builder import build_dataset
from pathlib import Path
from mmrotate.core.visualization import imshow_det_rbboxes
import os
# dataset settings
dataset_type = 'DOTADataset'
data_root = '/media/ljm/b930b01d-640a-4b09-8c3c-777d88f63e8b/Gejunyao/mmrotate/data/DOTA/split_ss_dota/'
angle_version = 'le90'
classes = ['ship',]
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
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
    train=dict(
        type=dataset_type,
        version=angle_version,
        ann_file=data_root + 'train/annfiles/',
        img_prefix=data_root + 'train/images/',
        classes=classes,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        version=angle_version,
        ann_file=data_root + 'val/annfiles/',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        version=angle_version,
        ann_file=data_root + 'val/images/',
        img_prefix=data_root + 'val/images/',
        classes=classes,
        pipeline=test_pipeline))
dataset = build_dataset(data['train'])

print(dataset)