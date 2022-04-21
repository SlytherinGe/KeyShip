import cv2
import numpy as np
from mmdet.core import BitmapMasks

from ..builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module()
class InstanceMaskGenerator(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, results):

        img_h, img_w, _ = results['pad_shape']
        gt_bboxes = results['gt_bboxes']
        num_boxes = len(gt_bboxes)
        # mask channel 0~2 represents: target center, short-side center, long-side center
        pseudo_mask = np.zeros((num_boxes, img_h, img_w), np.uint8)  
        for k, gt_bbox in enumerate(gt_bboxes):
            x, y, w, h, a = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], gt_bbox[4]
            pts = cv2.boxPoints(((x, y), (w, h), a))  
            pts = np.int0(pts)
            pseudo_mask[k, ...] = cv2.drawContours(pseudo_mask[k], [pts], -1, 1, -1)

        results['gt_masks'] = BitmapMasks(pseudo_mask, img_h, img_w)
        results['mask_fields'].append('gt_masks')

        return results   