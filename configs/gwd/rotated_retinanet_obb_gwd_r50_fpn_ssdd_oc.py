_base_ = ['../rotated_retinanet/rotated_retinanet_obb_r50_fpn_oc.py']

model = dict(
    bbox_head=dict(
        reg_decoded_bbox=True,
        loss_bbox=dict(type='GDLoss', loss_type='gwd', loss_weight=5.0)))

work_dir = '/media/gejunyao/Disk/Gejunyao/exp_results/mmdetection_files/SSDD/GWD/exp2/'