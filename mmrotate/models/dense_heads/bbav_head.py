import math
import numpy as np
import cv2
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmcv.cnn import ConvModule, xavier_init
from mmcv.runner import force_fp32
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from mmdet.core import multi_apply
from inspect import signature
from mmrotate.core.bbox.transforms import poly2obb
from mmrotate.core import (multiclass_nms_rotated,
                           aug_multiclass_nms_rotated,
                           bbox_mapping_back)
from ..builder import ROTATED_HEADS, build_loss
from ..utils import (get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat)
from mmcv import Config

try:
    from mmcv.ops import diff_iou_rotated_2d
except:  # noqa: E722
    diff_iou_rotated_2d = None

@ROTATED_HEADS.register_module()
class BBAVHead(BaseDenseHead):

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
                            out_ch=10,
                            loss=dict(
                                type='SmoothL1Loss', 
                                beta=1.0, 
                                loss_weight=1)),
                        dict(type='reg', 
                            out_ch=2,
                            loss=dict(
                                type='SmoothL1Loss', 
                                beta=1.0, 
                                loss_weight=1)),
                        dict(type='cls_theta', 
                            out_ch=1,
                            loss=dict(
                                type='CrossEntropyLoss', 
                                use_sigmoid=True, 
                                loss_weight=1))],
                train_cfg=None,
                test_cfg=None,
                norm_cfg=None,
                init_cfg=None):
        super().__init__(init_cfg)
        if diff_iou_rotated_2d is None:
            raise ImportError('Please install mmcv-full >= 1.5.0.')

        self.in_channels = in_channels
        self.feat_channels = feat_channels
        # build losses
        self.losses = dict()
        for branch in head_branches:
            branch_type = branch['type']
            loss_dict = branch['loss']
            self.losses[branch_type] = None if loss_dict == None else build_loss(loss_dict)
        self.head_branches = head_branches
        self.num_classes = head_branches[0].get('out_ch', None)
        self.num_dets = test_cfg.get('num_dets', 500)
        self.conf_thr = test_cfg.get('conf_thr', 0.18)
        self.version = test_cfg.get('version', 'oc')
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):

        act_cfg = dict(type='ReLU')
        for branch in self.head_branches:
            branch_type = branch['type']
            out_channels = branch['out_ch']
            if branch_type == 'wh':
                ksize=3
                psize=1
            else:
                ksize=1
                psize=0
            convs = nn.ModuleList([
            ConvModule(self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        act_cfg=act_cfg,
                        norm_cfg=self.norm_cfg),
            ConvModule(self.feat_channels,
                        out_channels,
                        kernel_size=ksize,
                        padding=psize,
                        act_cfg=None,
                        norm_cfg=None)])
            self.__setattr__(branch_type, convs)

    def init_weights(self):
        super().init_weights()
        for branch in self.head_branches:
            branch_type = branch['type']
            for m in self.__getattr__(branch_type):
                if isinstance(m.conv, nn.Conv2d):
                    xavier_init(m.conv)     
                    nn.init.constant_(m.conv.bias, 0.)   
            if branch_type == 'hm':    
                self.__getattr__(branch_type)[-1].conv.bias.data.fill_(-2.19)

    def forward(self, x):

        unpacked_x = x[0]

        feat = [unpacked_x for _ in range(len(self.head_branches))]

        for k, branch in enumerate(self.head_branches):
            branch_type = branch['type']
            layers = self.__getattr__(branch_type)
            for layer in layers:
                feat[k] = layer(feat[k]) 
            feat[k] = [feat[k]]

        return tuple(feat)  

    def __reorder_pts(self, tt, rr, bb, ll):

        pts = torch.stack([tt, rr, bb, ll])
        l_ind = torch.argmin(pts[:,0])
        r_ind = torch.argmax(pts[:,0])
        t_ind = torch.argmin(pts[:,1])
        b_ind = torch.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new,rr_new,bb_new,ll_new        

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
        gt_rbox = gt_bboxes.new_zeros((10, feat_h, feat_w))
        gt_offset = gt_bboxes.new_zeros((2, feat_h, feat_w))
        gt_theta = gt_bboxes.new_zeros((1, feat_h, feat_w))

        scaled_gt_bboxes = gt_bboxes.clone()
        # rescale the gt_bboxes to fit the feature map size
        scaled_gt_bboxes[:, 0] = scaled_gt_bboxes[:, 0] / stride_w
        scaled_gt_bboxes[:, 1] = scaled_gt_bboxes[:, 1] / stride_h
        scaled_gt_bboxes[:, 2:4] = scaled_gt_bboxes[:, 2:4] / np.sqrt(stride_h * stride_w)        

        for gt_bbox, gt_label in zip(scaled_gt_bboxes, gt_labels):
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
            a = a * 180 / np.pi
            pts = cv2.boxPoints(((x, y), (w, h), a))
            pts = gt_rbox.new_tensor(pts)
            bl = pts[0,:]
            tl = pts[1,:]
            tr = pts[2,:]
            br = pts[3,:]          
            tt = (tl+tr)/2
            rr = (tr+br)/2
            bb = (bl+br)/2
            ll = (tl+bl)/2
            ct = gt_rbox.new_tensor([x, y])
            if a in [-90.0, -0.0, 0.0]:
                tt,rr,bb,ll = self.__reorder_pts(tt,rr,bb,ll)

            gt_rbox[0:2, ct_y, ct_x] = tt - ct
            gt_rbox[2:4, ct_y, ct_x] = rr - ct
            gt_rbox[4:6, ct_y, ct_x] = bb - ct
            gt_rbox[6:8, ct_y, ct_x] = ll - ct

            # calculate box w, h
            x1 = torch.min(pts[:,0])
            x2 = torch.max(pts[:,0])
            y1 = torch.min(pts[:,1])
            y2 = torch.max(pts[:,1])
            wh = torch.stack([x2-x1, y2-y1]) * 1.
            gt_rbox[8:10, ct_y, ct_x] = wh
            # gt_theta
            hbox = gt_bbox.new_tensor([[[x, y, wh[0], wh[1], 0]]])
            rbox = gt_bbox[None][None]
            iou = diff_iou_rotated_2d(hbox, rbox)
            if iou < 0.95:
                gt_theta[0, ct_y, ct_x] = 1

        return gt_heatmap, gt_rbox, gt_offset, gt_theta

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
            batched_gt_rbox = torch.stack(res[1], dim=0)
            batched_gt_offset = torch.stack(res[2], dim=0)
            batched_gt_theta = torch.stack(res[3], dim=0)

            target_result = dict(
                gt_heatmap = batched_gt_heatmap,
                gt_rbox = batched_gt_rbox,
                gt_offset = batched_gt_offset,
                gt_theta = batched_gt_theta
            )
            multi_lvl_targets.append(target_result)

        return  multi_lvl_targets

    def loss_single(self,
                    heatmap,
                    rbox,
                    offset,
                    theta,
                    targets):

        gt_heatmap = targets['gt_heatmap']
        gt_rbox = targets['gt_rbox']
        gt_offset = targets['gt_offset']
        gt_theta = targets['gt_theta']

        gt_mask = gt_rbox.ne(0.).sum(1).gt(0).unsqueeze(1).type_as(heatmap)

        num_pos = max(1, gt_mask.sum())

        heatmap_loss = self.losses['hm'](heatmap.sigmoid(),
                                         gt_heatmap,
                                        #  gt_mask,
                                         avg_factor=num_pos)

        offset_loss = self.losses['reg'](offset,
                                       gt_offset,
                                       gt_mask,
                                       avg_factor=num_pos)

        rbox_loss = self.losses['wh'](rbox,
                                   gt_rbox,
                                   gt_mask,
                                   avg_factor=num_pos)

        theta_loss = self.losses['cls_theta'](theta.sigmoid(),
                                     gt_theta,
                                     gt_mask,
                                     avg_factor=num_pos)

        return heatmap_loss, offset_loss, rbox_loss, theta_loss

    @force_fp32(apply_to=('multi_lvl_heatmap', 'multi_lvl_rbox', 'multi_lvl_offset', 'multi_lvl_theta'))
    def loss(self,
             multi_lvl_heatmap,
             multi_lvl_rbox,
             multi_lvl_offset,
             multi_lvl_theta,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):

        featmap_sizes = [featmap.size() for featmap in multi_lvl_heatmap]
        targets = self.get_targets(gt_bboxes, gt_bboxes_ignore, gt_labels,
                                   img_metas, featmap_sizes)
        
        (heatmap_loss, offset_loss,
         rbox_loss, theta_loss) = multi_apply(self.loss_single,
                                              multi_lvl_heatmap,
                                              multi_lvl_rbox,
                                              multi_lvl_offset,
                                              multi_lvl_theta,
                                              targets)
        
        loss_dict = dict(
            heatmap_loss=heatmap_loss,
            offset_loss=offset_loss,
            rbox_loss=rbox_loss,
            theta_loss=theta_loss
        )

        return loss_dict

    def decode_heatmap_single(self,
                              heatmap_map,
                              rbox_map,
                              offset_map,
                              theta_map,
                              num_dets,
                              conf_thr):

        batch, _, height, width = heatmap_map.size()
        heatmap_map = heatmap_map.sigmoid()
        theta_map = theta_map.sigmoid()

        heatmap = get_local_maximum(heatmap_map)
        scores, inds, clses, ys, xs = get_topk_from_heatmap(heatmap, k=num_dets)
        off = transpose_and_gather_feat(offset_map, inds)
        off = off.view(batch, num_dets, 2)
       
        xs = xs.view(batch, num_dets, 1) + off[:, :, 0:1]
        ys = ys.view(batch, num_dets, 1) + off[:, :, 1:2]

        clses = clses.view(batch, num_dets, 1)
        scores = scores.view(batch, num_dets, 1)        

        rbox = transpose_and_gather_feat(rbox_map, inds)
        rbox = rbox.view(batch, num_dets, 10)

        theta = transpose_and_gather_feat(theta_map, inds)
        theta = theta.view(batch, num_dets, 1)
        mask = (theta>0.8).float().view(batch, num_dets, 1)

        #
        tt_x = (xs+rbox[..., 0:1])*mask + (xs)*(1.-mask)
        tt_y = (ys+rbox[..., 1:2])*mask + (ys-rbox[..., 9:10]/2)*(1.-mask)
        rr_x = (xs+rbox[..., 2:3])*mask + (xs+rbox[..., 8:9]/2)*(1.-mask)
        rr_y = (ys+rbox[..., 3:4])*mask + (ys)*(1.-mask)
        bb_x = (xs+rbox[..., 4:5])*mask + (xs)*(1.-mask)
        bb_y = (ys+rbox[..., 5:6])*mask + (ys+rbox[..., 9:10]/2)*(1.-mask)
        ll_x = (xs+rbox[..., 6:7])*mask + (xs-rbox[..., 8:9]/2)*(1.-mask)
        ll_y = (ys+rbox[..., 7:8])*mask + (ys)*(1.-mask)
        #
        detections = torch.cat([xs,                      # cen_x
                                ys,                      # cen_y
                                tt_x,
                                tt_y,
                                rr_x,
                                rr_y,
                                bb_x,
                                bb_y,
                                ll_x,
                                ll_y],
                               dim=2)        

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
            cen_pt = bbox_res[...,0:2]
            tt = bbox_res[...,2:4]
            rr = bbox_res[...,4:6]
            bb = bbox_res[...,6:8]
            ll = bbox_res[...,8:10]
            tl = tt + ll - cen_pt
            bl = bb + ll - cen_pt
            tr = tt + rr - cen_pt
            br = bb + rr - cen_pt
            poly_rboxes = torch.cat([tl, bl, tr, br], dim=-1)
            # convert the coord from heatmap to ori image
            poly_rboxes[..., 0::2] = poly_rboxes[..., 0::2] * stride_w
            poly_rboxes[..., 1::2] = poly_rboxes[..., 1::2] * stride_h
            for k, poly_rbox in enumerate(poly_rboxes[0]):
                tl, bl, tr, br = self.__reorder_pts(poly_rbox[0:2], poly_rbox[2:4], poly_rbox[4:6], poly_rbox[6:8])
                poly_rboxes[0, k] = torch.cat([tl, bl, tr, br])
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
                   multi_lvl_rbox,
                   multi_lvl_offset,
                   multi_lvl_theta,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):

        featmap_sizes = [heatmap.size() for heatmap in multi_lvl_heatmap]


        result_list = []
        multi_lvl_box_res, multi_lvl_scores, multi_lvl_clses = multi_apply(
            self.decode_heatmap_single,
            multi_lvl_heatmap,
            multi_lvl_rbox,
            multi_lvl_offset,
            multi_lvl_theta,
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

    def aug_test(self, feats, img_metas, rescale=False):
        """Test det bboxes with test time augmentation, can be applied in
        DenseHead except for ``RPNHead`` and its variants, e.g., ``GARPNHead``,
        etc.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.
        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is ``bboxes`` with shape (n, 6),
                where 6 represent (x, y, w, h, a, score).
                The shape of the second tensor in the tuple is ``labels``
                with shape (n,). The length of list should always be 1.
        """
        # check with_nms argument
        gb_sig = signature(self.get_bboxes)
        gb_args = [p.name for p in gb_sig.parameters.values()]
        gbs_sig = signature(self._get_bboxes_single)
        gbs_args = [p.name for p in gbs_sig.parameters.values()]
        assert ('with_nms' in gb_args) and ('with_nms' in gbs_args), \
            f'{self.__class__.__name__}' \
            ' does not support test-time augmentation'

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.forward(x)
            bbox_outputs = self.get_bboxes(
                *outs,
                img_metas=img_meta,
                rescale=False,
                with_nms=False)[0]
            aug_bboxes.append(bbox_outputs[0])
            aug_scores.append(bbox_outputs[1])

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_bboxes(
            aug_bboxes, aug_scores, img_metas)

        merged_scores, merged_labels = torch.max(merged_scores, dim=1)
        merged_bboxes = torch.cat([merged_bboxes, merged_scores[:, None]], -1)
        if merged_bboxes.numel() == 0:
            return [
                (merged_bboxes, merged_labels),
            ]

        det_bboxes, det_labels = aug_multiclass_nms_rotated(
            merged_bboxes, merged_labels, self.test_cfg.score_thr,
            self.test_cfg.nms, self.test_cfg.max_per_img, self.num_classes)

        if rescale:
            # angle should not be rescaled
            merged_bboxes[:, :4] *= merged_bboxes.new_tensor(
                img_metas[0][0]['scale_factor'])

        return [
            (det_bboxes, det_labels),
        ]

    def merge_aug_bboxes(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple[Tensor]: ``bboxes`` with shape (n,4), where
            4 represent (tl_x, tl_y, br_x, br_y)
            and ``scores`` with shape (n,).
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            flip_direction = img_info[0]['flip_direction']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip,
                                       flip_direction)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores
