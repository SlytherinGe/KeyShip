# bash tools/dist_train.sh configs/ablation_study5/R3Det.py 2 --seed 42
# wait
bash tools/dist_train.sh configs/ablation_study5/RFaster_RCNN.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/ROITransformer.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/RRetinaNet.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/S2ANet.py 2 --seed 42
wait
