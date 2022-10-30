import mmcv
from mmcv.runner import load_checkpoint

from mmdet.apis import inference_detector, show_result_pyplot
from mmrotate.models import build_detector

# Choose to use a config and initialize the detector
config = '/media/slytheringe/Disk/Gejunyao/exp_results/mmdetection_files/HRSC/ExtremeShip/exp1/extreme_ship.py'
# Setup a checkpoint file to load
checkpoint = '/media/slytheringe/Disk/Gejunyao/exp_results/mmdetection_files/HRSC/ExtremeShip/exp1/epoch_150.pth'

IMG_ROOT = '/media/slytheringe/Disk1/Datasets/HRSC2016/FullDataSet/AllImages/'
# CACHE_ROOT = '/media/gejunyao/Disk/Gejunyao/exp_results/visualization/middle_part/ExtremeShipV3/exp7/TestCache'
CACHE_ROOT = '/home/slytheringe/FastCache/HRSC/'
IMG_ID = '100000736'

# load image
img = IMG_ROOT + IMG_ID + '.bmp'

# Set the device to be used for evaluation
device='cuda:0'

# Load the config
config = mmcv.Config.fromfile(config)
# Set pretrained to be None since we do not need pretrained model here
config.model.pretrained = None

# Initialize the detector
model = build_detector(config.model)

# Load checkpoint
checkpoint = load_checkpoint(model, checkpoint, map_location=device)

# Set the classes of models for inference
model.CLASSES = checkpoint['meta']['CLASSES']

# We need to set the model's cfg for inference
model.cfg = config

# Convert the model to GPU
model.to(device)
# Convert the model into evaluation mode
model.eval()

result = inference_detector(model, img)
# Let's plot the result
show_result_pyplot(model, img, result, score_thr=0.01)