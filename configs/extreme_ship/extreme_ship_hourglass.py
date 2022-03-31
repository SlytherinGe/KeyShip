_base_ = [
    '../_base_/datasets/ssdd_official.py'
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

data = dict(
    samples_per_gpu=6,
    workers_per_gpu=16,
    train=dict(version=angle_version),
    val=dict(version=angle_version),
    test=dict(version=angle_version))

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

work_dir = '/media/gejunyao/Disk/Gejunyao/exp_results/mmdetection_files/SSDD/ExtremeShip/exp45/'

# evaluation
evaluation = dict(interval=1, metric='mAP', save_best='auto')
# optimizer
optimizer = dict(type='SGD', lr=0.008, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='cyclic',
    warmup=None,
    cyclic_times=2,
    target_ratio=(10, 1e-2))
runner = dict(type='EpochBasedRunner', max_epochs=100)
checkpoint_config = dict(interval=2)
