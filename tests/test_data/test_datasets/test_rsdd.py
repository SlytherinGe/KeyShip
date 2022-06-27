import torch
import numpy as np
import matplotlib.pyplot as plt
from mmrotate.datasets import RSDDDataset
import os
import cv2
from tqdm import tqdm
IMG_ROOT = '/media/gejunyao/Disk1/Datasets/RSDD-SAR/JPEGImages/'
img_norm_cfg = dict(
    mean=[0, 0, 0], std=[1., 1., 1.], to_rgb=True)
data_root = '/media/gejunyao/Disk1/Datasets/RSDD-SAR/'
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RRandomFlip', flip_ratio=0.0),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
ssdd_dataset = RSDDDataset(version='oc',
                            ann_file=data_root + 'ImageSets/test.txt',
                            img_prefix=data_root,
                            pipeline=train_pipeline)

print(ssdd_dataset[0])