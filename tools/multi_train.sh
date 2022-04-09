echo "rotated_retinanet_obb_gwd_r50_fpn_ssdd_oc"
python tools/train.py configs/gwd/rotated_retinanet_obb_gwd_r50_fpn_ssdd_oc.py --seed 42
wait
echo "r3det_refine_r50_fpn_1x_ssdd_oc"
python tools/train.py configs/r3det/r3det_refine_r50_fpn_1x_ssdd_oc.py --seed 42
wait
echo "rotated_faster_rcnn_r50_oc"
python tools/train.py configs/rotated_faster_rcnn/rotated_faster_rcnn_r50_oc.py --seed 42
wait
echo "rotated_retinanet_obb_r50_fpn_oc"
python tools/train.py configs/rotated_retinanet/rotated_retinanet_obb_r50_fpn_oc.py --seed 42
wait
# python tools/train.py configs/extreme_ship/extreme_ship_hourglass.py --seed 42
# wait
# python tools/train.py configs/extreme_ship/backup.py --seed 42
# wait
# python tools/train.py configs/extreme_ship/backup1.py --seed 42
# wait