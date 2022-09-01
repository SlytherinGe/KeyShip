# bash tools/dist_train.sh configs/ablation_study4/Oriented_RepPoints.py 2 --seed 42
# wait
bash tools/dist_train.sh configs/ablation_study5/BBAV.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/Gliding_Vertex.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/GWD.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study5/Oriented_RCNN.py 2 --seed 42
wait
