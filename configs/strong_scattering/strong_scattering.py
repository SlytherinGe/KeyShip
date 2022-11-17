_base_ = [
    '../_base_/datasets/ssdd_official.py', '../_base_/schedules/schedule_benchmark_150e.py',
    '../_base_/benchmark_runtime.py'
]

angle_version = 'oc'
# model settings
model = dict(
    type='ScatteringSAR',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        strides=(1, 2, 2, 1),
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ACSNeck',
        in_channels=2048,
        out_channels=256
    ),
    bbox_head=dict(
        type='StrongScatteringHead',
                 num_classes=1,
                 in_channels=256,
                 num_feats=256,
                 up_sample_rate=16,
                 loss_heatmap=dict(
                    type='GaussianFocalLoss',
                    alpha=2.0,
                    gamma=1.0,
                    loss_weight=1                     
                ),
                 loss_embedding=dict(
                     type='DenseAssociativeEmbeddingLoss',
                     loss_type='instance'
                 )),
    train_cfg = dict(cache_cfg=None),
    test_cfg = dict(
        cache_cfg = dict(
            root = '/home/slytheringe/FastCache/StrongScattering'
        ),
        num_dets = 500,
        conf_thr = 0.18,
        version = angle_version,
        score_thr = 0.1,
        nms_cfg = dict(type='rnms', iou_thr=0.05),
        max_per_img=100))
train_cfg = dict(
    cache_cfg=None,
)

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
    dict(type='SARScatteringMaskGenerator'),
    dict(type='RTranslate', prob=0.3, img_fill_val=0, level=3),
    dict(type='BrightnessTransform', level=3, prob=0.3),
    dict(type='ContrastTransform', level=3, prob=0.3),
    dict(type='EqualizeTransform', prob=0.3),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', pad_to_square=True),
    # dict(type='InstanceMaskGenerator'),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
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

work_dir = '/media/slytheringe/Disk/Gejunyao/exp_results/mmdetection_files/SSDD/StrongScattering/exp4'

optimizer = dict(type='Adam', lr=0.00001)
checkpoint_config = dict(interval=30)
evaluation = dict(interval=10, metric='mAP', save_best='auto')