# benchmark on titan V * 6
python tools/train.py configs/benchmark_hrsid/gliding_vertex.py -seed 42 --gpu-ids 012345
wait
python tools/train.py configs/benchmark_hrsid/rotated_retinanet.py -seed 42 --gpu-ids 012345
wait
