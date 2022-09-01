bash tools/dist_train.sh configs/ablation_study4/Gliding_Vertex.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/GWD.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/Oriented_RCNN.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/Oriented_RepPoints.py 2 --seed 42
wait
bash tools/dist_train.sh configs/ablation_study4/BBAV.py 2 --seed 666
wait