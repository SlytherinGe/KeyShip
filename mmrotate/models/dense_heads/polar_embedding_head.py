import math
import numpy as np
import torch
from mmrotate.models.dense_heads import BBAVHead
from mmcv.runner import force_fp32
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmrotate.models.utils import polar_encode, polar_decode
from mmdet.core import multi_apply
from mmrotate.core.bbox.transforms import obb2poly, poly2obb
from mmcv.ops import min_area_polygons
from mmrotate.core import (multiclass_nms_rotated)
from ..builder import ROTATED_HEADS
from ..utils import (get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat)
from mmcv import Config

@ROTATED_HEADS.register_module()
class PolarEncodingHead(BBAVHead):

    def __init__(self,
                in_channels,
                feat_channels=256,
                head_branches=[dict(type='hm', 
                            out_ch=1,
                            loss=dict(
                                type='GaussianFocalLoss',
                                alpha=2.0,
                                gamma=4.0,
                                loss_weight=1)),
                        dict(type='wh', 
                            out_ch=8,
                            loss=dict(
                                type='SmoothL1Loss', 
                                beta=1.0, 
                                loss_weight=1)),
                        dict(type='reg', 
                            out_ch=2,
                            loss=dict(
                                type='SmoothL1Loss', 
                                beta=1.0, 
                                loss_weight=1))],
                train_cfg=None,
                test_cfg=None,
                norm_cfg=None,
                init_cfg=None):
        super().__init__(in_channels,
                         feat_channels,
                         head_branches,
                         train_cfg,
                         test_cfg,
                         norm_cfg,
                         init_cfg)

        self.num_encoding_channels = head_branches[1]['out_ch']

    def _get_targets_single(self,
                            gt_bboxes:torch.tensor,
                            gt_bboxes_ignore:torch.tensor,
                            gt_labels:torch.tensor,
                            img_meta,
                            feat_shape):

        _, _, feat_h, feat_w = feat_shape
        img_h, img_w, _ = img_meta.get('pad_shape', None)
        stride_h, stride_w = img_h / feat_h, img_w / feat_w
        gt_heatmap = gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))
        gt_encoding = gt_bboxes.new_zeros((self.num_encoding_channels, feat_h, feat_w))
        gt_offset = gt_bboxes.new_zeros((2, feat_h, feat_w))

        scaled_gt_bboxes = gt_bboxes.clone()
        # rescale the gt_bboxes to fit the feature map size
        scaled_gt_bboxes[:, 0] = scaled_gt_bboxes[:, 0] / stride_w
        scaled_gt_bboxes[:, 1] = scaled_gt_bboxes[:, 1] / stride_h
        scaled_gt_bboxes[:, 2:4] = scaled_gt_bboxes[:, 2:4] / np.sqrt(stride_h * stride_w)        

        sacled_gt_poly = obb2poly(scaled_gt_bboxes, version=self.version)

        for gt_bbox, gt_poly, gt_label in zip(scaled_gt_bboxes, sacled_gt_poly, gt_labels):
            cat = int(gt_label)
            x, y, w, h, a = float(gt_bbox[0]), float(gt_bbox[1]), float(gt_bbox[2]), float(gt_bbox[3]), float(gt_bbox[4])
            if x < 0 or y < 0:
                continue
            # gt_heatmap
            radius = gaussian_radius((math.ceil(w), math.ceil(h)), min_overlap = 0.7)
            radius = max(0, int(radius))
            ct_x = int(min(x, feat_w - 1))
            ct_y = int(min(y, feat_h - 1))
            gt_heatmap[cat] = gen_gaussian_target(gt_heatmap[cat], [ct_x, ct_y], radius)
            # gt_offset
            off = gt_offset.new_tensor([x - ct_x, y - ct_y])
            gt_offset[:,ct_y, ct_x] = off
            # gt_rbox
            polar_encodings = polar_encode(gt_poly, gt_bbox[:2], self.num_encoding_channels)
            gt_encoding[:, ct_y, ct_x] = polar_encodings

        return gt_heatmap, gt_encoding, gt_offset

    def get_targets(self, gt_bboxes_list:list,
                          gt_bboxes_ignore:list,
                          gt_labels_list:list,
                          img_metas:list,
                          feat_shapes:list):

        num_imgs = len(img_metas)
        if gt_bboxes_ignore == None:
            gt_bboxes_ignore = [None for _ in range(num_imgs)]
        multi_lvl_targets = []
        for f_shape in feat_shapes:
            
            res = multi_apply(self._get_targets_single,
                              gt_bboxes_list,
                              gt_bboxes_ignore,
                              gt_labels_list,
                              img_metas,
                              [f_shape for _ in range(num_imgs)])
            
            batched_gt_heatmap = torch.stack(res[0], dim=0)
            batched_gt_encoding = torch.stack(res[1], dim=0)
            batched_gt_offset = torch.stack(res[2], dim=0)

            target_result = dict(
                gt_heatmap = batched_gt_heatmap,
                gt_encoding = batched_gt_encoding,
                gt_offset = batched_gt_offset,
            )
            multi_lvl_targets.append(target_result)

        return  multi_lvl_targets

    def loss_single(self,
                    heatmap,
                    encoding,
                    offset,
                    targets):

        gt_heatmap = targets['gt_heatmap']
        gt_encoding = targets['gt_encoding']
        gt_offset = targets['gt_offset']

        gt_mask = gt_encoding.ne(0.).sum(1).gt(0).unsqueeze(1).type_as(heatmap)

        num_pos = max(1, gt_mask.sum())

        heatmap_loss = self.losses['hm'](heatmap.sigmoid(),
                                         gt_heatmap,
                                        #  gt_mask,
                                         avg_factor=num_pos)

        offset_loss = self.losses['reg'](offset,
                                       gt_offset,
                                       gt_mask,
                                       avg_factor=num_pos)

        encoding_loss = self.losses['wh'](encoding,
                                        gt_encoding,
                                        gt_mask,
                                        avg_factor=num_pos)

        return heatmap_loss, encoding_loss, offset_loss

    @force_fp32(apply_to=('multi_lvl_heatmap', 'multi_lvl_encoding', 'multi_lvl_offset'))
    def loss(self,
             multi_lvl_heatmap,
             multi_lvl_encoding,
             multi_lvl_offset,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size() for featmap in multi_lvl_heatmap]
        targets = self.get_targets(gt_bboxes, gt_bboxes_ignore, gt_labels,
                                   img_metas, featmap_sizes)
        
        (heatmap_loss, encoding_loss,
         offset_loss) = multi_apply(self.loss_single,
                                              multi_lvl_heatmap,
                                              multi_lvl_encoding,
                                              multi_lvl_offset,
                                              targets)
        
        loss_dict = dict(
            heatmap_loss=heatmap_loss,
            encoding_loss=encoding_loss,
            offset_loss=offset_loss
        )

        return loss_dict

    def decode_heatmap_single(self,
                              heatmap_map,
                              encoding_map,
                              offset_map,
                              num_dets,
                              conf_thr):

        batch, _, height, width = heatmap_map.size()
        heatmap_map = heatmap_map.sigmoid()

        heatmap = get_local_maximum(heatmap_map)
        scores, inds, clses, ys, xs = get_topk_from_heatmap(heatmap, k=num_dets)
        off = transpose_and_gather_feat(offset_map, inds)
        off = off.view(batch, num_dets, 2)
       
        xs = xs.view(batch, num_dets, 1) + off[:, :, 0:1]
        ys = ys.view(batch, num_dets, 1) + off[:, :, 1:2]

        clses = clses.view(batch, num_dets, 1)
        scores = scores.view(batch, num_dets, 1)        

        encodings = transpose_and_gather_feat(encoding_map, inds)
        encodings = encodings.view(batch, num_dets, self.num_encoding_channels)

        detections = polar_decode(encodings, xs, ys, self.num_encoding_channels)     

        index = (scores>conf_thr).squeeze(0).squeeze(1)
        detections = detections[:,index,:]
        scores = scores[:,index,:]
        clses = clses[:,index,:]

        return detections, scores, clses

    def _get_bboxes_single(self,
                         all_lvl_bbox_res,
                         all_lvl_bbox_scores,
                         all_lvl_bbox_clses,
                         all_lvl_feat_sizes,
                         img_meta,
                         rescale=False,
                         with_nms=True):

        img_h, img_w, _ = img_meta.get('pad_shape', None)
        all_lvl_rboxes = []

        for k, bbox_res in enumerate(all_lvl_bbox_res):
            feat_w, feat_h = all_lvl_feat_sizes[k][2:]
            stride_h, stride_w = img_h / feat_h, img_w / feat_w
            # convert the coord from heatmap to ori image
            bbox_res[..., 0::2] = bbox_res[..., 0::2] * stride_w
            bbox_res[..., 1::2] = bbox_res[..., 1::2] * stride_h

            if bbox_res.dim() == 3:
                poly_rboxes = []
                for batch_id in range(len(bbox_res)):
                    if bbox_res[batch_id].size(0) > 0:
                        poly = min_area_polygons(bbox_res[batch_id])
                    else:
                        poly = torch.zeros(0,8,device=bbox_res.device, dtype=bbox_res.dtype)
                    poly_rboxes.append(poly)
                poly_rboxes = torch.stack(poly_rboxes)
            elif bbox_res.dim() == 2:
                poly_rboxes = min_area_polygons(bbox_res)
            else:
                raise 'incorrect dim for input box res'
            det_rboxes = poly2obb(poly_rboxes, version=self.version)[None]
            if rescale:
                img_shape = img_meta['img_shape'][:2]
                ori_shape = img_meta['ori_shape'][:2]
                stride_h = img_shape[0] / ori_shape[0]
                stride_w = img_shape[1] / ori_shape[1]
                stride_l = np.sqrt(stride_h * stride_w)
                det_rboxes[..., :4] = det_rboxes[..., :4] /\
                                    det_rboxes.new_tensor([stride_w, stride_h, stride_l, stride_l])
            all_lvl_rboxes.append(det_rboxes)
        det_rboxes = torch.cat(all_lvl_rboxes, dim=1)[0]
        det_scores = torch.cat(all_lvl_bbox_scores, dim=1)[0]
        det_clses = torch.cat(all_lvl_bbox_clses, dim=1)[0]
        
        if det_rboxes.size(0) < 1:
            # add a default empty result for eval
            # TODO:if there are multiple classes, this should be changed
            det_rboxes = det_rboxes.new_zeros((1,5))
            det_clses = det_clses.new_zeros((1,1))
            det_scores = det_scores.new_zeros((1,1)) * -1.

        if with_nms:
            _nms_cfg = self.test_cfg.get('nms', dict(type='rnms', iou_thr=0.05))
            _nms_cfg = Config(_nms_cfg)
            padding = det_scores.new_zeros(det_scores.shape[0], 1)
            det_scores = torch.cat([det_scores, padding], dim=1)
            det_rboxes, det_clses = multiclass_nms_rotated(det_rboxes, det_scores, self.test_cfg.score_thr, _nms_cfg, self.test_cfg.max_per_img)
        
            return det_rboxes, det_clses
        
        else:
            return det_rboxes, det_scores

    def get_bboxes(self,
                   multi_lvl_heatmap,
                   multi_lvl_encoding,
                   multi_lvl_offset,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        featmap_sizes = [heatmap.size() for heatmap in multi_lvl_heatmap]


        result_list = []
        multi_lvl_box_res, multi_lvl_scores, multi_lvl_clses = multi_apply(
            self.decode_heatmap_single,
            multi_lvl_heatmap,
            multi_lvl_encoding,
            multi_lvl_offset,
            [self.num_dets for _ in range(len(multi_lvl_heatmap))],
            [self.conf_thr for _ in range(len(multi_lvl_heatmap))]
        )

        for img_id in range(len(img_metas)):
            all_lvl_bbox_res = [kpts[img_id][None] for kpts in multi_lvl_box_res]
            all_lvl_bbox_scores = [scores[img_id][None] for scores in multi_lvl_scores]
            all_lvl_bbox_clses = [clses[img_id][None] for clses in multi_lvl_clses]
            results = self._get_bboxes_single(all_lvl_bbox_res,
                                              all_lvl_bbox_scores,
                                              all_lvl_bbox_clses,
                                              featmap_sizes,
                                              img_metas[img_id],
                                              rescale=rescale,
                                              with_nms=with_nms)
            result_list.append(results)

        return result_list
