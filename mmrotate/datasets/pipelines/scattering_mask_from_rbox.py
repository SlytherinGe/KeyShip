import cv2
import numpy as np
import mmcv
from mmdet.core import BitmapMasks

from ..builder import ROTATED_PIPELINES

@ROTATED_PIPELINES.register_module()
class SARScatteringMaskGenerator(object):

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, results):

        img_h, img_w, _ = results['pad_shape']
        img = results['img']
        gt_bboxes = results['gt_bboxes']
        img_gray = mmcv.rgb2gray(img)
        valid_img_pix = (img_gray > 255-(255-np.mean(img_gray))*0.75).astype(np.int0)
        num_boxes = len(gt_bboxes)
        # mask channel 0~2 represents: target center, short-side center, long-side center
        pseudo_mask = np.zeros((num_boxes, img_h, img_w), np.uint8)  
        for k, gt_bbox in enumerate(gt_bboxes):
            x, y, w, h, a = gt_bbox[0], gt_bbox[1], gt_bbox[2], gt_bbox[3], gt_bbox[4]
            canvas = np.zeros((img_h, img_w), np.uint8)
            pts = cv2.boxPoints(((x, y), (w, h), a/np.pi*180))  
            pts = np.int0(pts)
            canvas = cv2.drawContours(canvas, [pts], -1, 1, -1)
            canvas = (canvas + valid_img_pix) > 1
            pseudo_mask[k] = canvas.astype(np.uint8)

        results['gt_masks'] = BitmapMasks(pseudo_mask, img_h, img_w)
        results['mask_fields'].append('gt_masks')

        return results   