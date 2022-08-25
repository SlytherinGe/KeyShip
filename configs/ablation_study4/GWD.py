_base_ = [
    '../_base_/datasets/benchmark_ssdd.py', '../_base_/schedules/schedule_benchmark_150e.py',
    '../_base_/benchmark_runtime.py'
]

_base_ = ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_oc.py']

model = dict(
    backbone=dict(
        type='ResNet',
        depth=101,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet101')),
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

work_dir = '../exp_results/mmlab_results/ssdd/ablation_study4/gwd'