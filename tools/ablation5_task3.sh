# bash tools/dist_train.sh configs/ablation_study5/RRetinaNet.py 2 --seed 42
# wait
# bash tools/dist_train.sh configs/ablation_study5/S2ANet.py 2 --seed 42
# wait
# bash tools/dist_train.sh configs/ablation_study5/extreme_ship.py 2 --seed 42
# wait
bash tools/dist_train.sh configs/ablation_study5/Oriented_RCNN.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/BBAV.py 2 --seed 42
wait