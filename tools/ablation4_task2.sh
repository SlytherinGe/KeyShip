bash tools/dist_train.sh configs/ablation_study4/R3Det.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/RFaster_RCNN.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/ROITransformer.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/RRetinaNet.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/S2ANet.py 2 --seed 42
wait