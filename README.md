# KeyShip: Towards High-Precision Oriented SAR Ship Detection Using Key Points

## Abstract

Synthetic Aperture Radar (SAR) is an all-weather sensing technology that has proven its effectiveness for ship detection. However, detecting ships accurately with oriented bounding boxes (OBB) on SAR images is challenging due to arbitrary ship orientations and misleading scattering. In this article, we propose a novel anchor-free key-point-based detection method, KeyShip, for detecting orientated SAR ships with high precision. Our approach uses a shape descriptor to model a ship as a combination of three types of key points located at the short-edge centers, long-edge centers, and the target center. These key points are detected separately and clustered based on predicted shape descriptors to construct the final OBB detection results. To address the boundary problem that arises with the shape descriptor representation, we propose a soft training target assignment strategy that facilitates successful shape descriptor training and implicitly learns the shape information of the targets. Our experimental results on three datasets (SSDD, RSDD, and HRSC2016) demonstrate our proposed method’s high performance and robustness.

Details can be found in the [paper](https://www.mdpi.com/2072-4292/15/8/2035).

## Installation

Our code is implemented from the official MMRotate. Please follow the installation and usage guide from the MMRotate. You can either install our code alone or merge it into the MMRotate codebase. It is worth noting that our code only supports single-class detection.

MMRotate depends on [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv) and [MMDetection](https://github.com/open-mmlab/mmdetection).
Below are quick steps for installation.
Please refer to [Install Guide](https://mmrotate.readthedocs.io/en/latest/install.html) for more detailed instructions.

```shell
conda create -n open-mmlab python=3.7 pytorch==1.7.0 cudatoolkit=10.1 torchvision -c pytorch -y
conda activate open-mmlab
pip install openmim
mim install mmcv-full
mim install mmdet
git clone https://github.com/SlytherinGe/KeyShip.git
cd mmrotate
pip install -r requirements/build.txt
pip install -v -e .
```

## Citation

If you find our work useful in your research, please cite our paper.

```bibtex
@Article{ge2023keyship,
  AUTHOR = {Ge, Junyao and Tang, Yiping and Guo, Kaitai and Zheng, Yang and Hu, Haihong and Liang, Jimin},
  TITLE = {KeyShip: Towards High-Precision Oriented SAR Ship Detection Using Key Points},
  JOURNAL = {Remote Sensing},
  VOLUME = {15},
  YEAR = {2023},
  NUMBER = {8},
  ARTICLE-NUMBER = {2035},
  URL = {https://www.mdpi.com/2072-4292/15/8/2035},
  ISSN = {2072-4292},
  DOI = {10.3390/rs15082035}
}
```

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Acknowledgment

This project is not possible without multiple great open-sourced codebases and datasets. We list some notable examples below.

- [MMCV](https://github.com/open-mmlab/mmcv): OpenMMLab foundational library for computer vision.
- [MMDetection](https://github.com/open-mmlab/mmdetection): OpenMMLab detection toolbox and benchmark.
- [MMRotate](https://github.com/open-mmlab/mmrotate): OpenMMLab rotated object detection toolbox and benchmark.

