from inspect import signature
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmdet.models.plugins import DropBlock
import cv2

from mmrotate.core import (multiclass_nms_rotated,
                           aug_multiclass_nms_rotated,
                           bbox_mapping_back)
from ..builder import ROTATED_HEADS, build_loss
from mmdet.core import multi_apply
from ..utils import (gen_gaussian_targetR, get_local_maximum,
                     get_topk_from_heatmap, gather_feat,
                     transpose_and_gather_feat, keypoints2rbboxes,
                     sort_valid_gt_bboxes, get_target_map,
                     generate_ec_from_corner_pts,
                     generate_center_pointer_map2,
                     set_offset2)


INF = 1e8
SMALL_NUM = 1e-6
# multi-thread for saving heatmaps
import mmcv
import os
import queue
from threading import Thread

def ThreadSaveCacheFile(dq, test_cache_cfg):

    if test_cache_cfg is None:
        test_cache_cfg = dict()
    TeO_root = test_cache_cfg.get('root', './TestCache')
    path_dict = dict(
        TeO=TeO_root
    )

    while True:
        data = dq.get()
        save_folder = path_dict[data['type']]
        save_d = data['data']
        img_metas = data['img_metas']
        num_img = len(img_metas)
        save_b = data.get('boxes', None)
        for i in range(num_img):
            img_meta = img_metas[i]
            save_dict = dict()
            for key, _ in save_d.items():
                save_dict[key] = []
            for key, value in save_d.items():
                temp = value
                for j in range(len(temp)):
                    temp_feat = temp[j][i].clone().detach().cpu()[None]
                    if img_meta.get('flip', False):
                        temp_feat = torch.flip(temp_feat, [3])
                    save_dict[key].append(temp_feat)
            filename = img_meta['filename']
            filename = os.path.basename(filename).split('.')[0]
            saved_file = os.path.join(save_folder, filename)
            mmcv.dump(save_dict, saved_file + '.pkl')

        if save_b is not None:
            for key, values in save_b.items():
                for k, value in enumerate(values):
                    save_b[key][k] = value.cpu()
            img_meta = img_metas[0]
            filename = img_meta['filename']
            filename = os.path.basename(filename).split('.')[0]
            saved_file = os.path.join(save_folder, filename)
            mmcv.dump(save_b, saved_file + '_boxes.pkl')

        dq.task_done()


@ROTATED_HEADS.register_module()
class KeyShipHead(BaseDenseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 longside_center_cfg,
                 shortside_center_cfg,
                 target_center_cfg,
                 center_pointer_cfg,
                 ec_offset_cfg,
                 regress_ratio=((-1, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1)),
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1                     
                    ),
                 loss_pointer=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1
                 ),
                 loss_offsets=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                ):
        super(KeyShipHead, self).__init__()
        # self.head_modules = []
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.regress_ratios = regress_ratio
        # init losses
        self.loss_heatmap = None if loss_heatmap == None else build_loss(loss_heatmap)
        self.loss_pointer = None if loss_pointer == None else build_loss(loss_pointer)
        self.loss_offsets = None if loss_offsets == None else build_loss(loss_offsets)
        self.train_cfg = train_cfg if train_cfg is not None else dict()
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.sigma_ratio = self.train_cfg.get('gaussioan_sigma_ratio', (0.1, 0.1))
        self.lc_ptr_sigma = self.test_cfg.get('lc_ptr_sigma', 0.1)
        self.sc_ptr_sigma = self.test_cfg.get('sc_ptr_sigma', 0.01)
        self.test_feat_ind = self.test_cfg.get('test_feat_ind', [-1])
        self.circular_target = self.train_cfg.get('circular_target', False)
        self.rigid_edge_pointer = self.train_cfg.get('rigid_edge_pointer', False)
        self.kpt_enable = self.train_cfg.get('kpt_enable', True)
        self.refine_enable = self.train_cfg.get('refine_enable', True)

        if not self.kpt_enable:
            self.lc_ptr_sigma = 1e-9
            self.sc_ptr_sigma = 1e-9

        self.test_cache_cfg = self.test_cfg.get('cache_cfg', None)
        self.__debug = False

        self.longside_center_cfg = longside_center_cfg
        self.shortside_center_cfg = shortside_center_cfg
        self.target_center_cfg = target_center_cfg
        self.center_pointer_cfg = center_pointer_cfg
        self.ec_offset_cfg = ec_offset_cfg

        self._init_layers()

        # test cfg init
        self.list_num_pkts_per_lvl = self.test_cfg.get('num_kpts_per_lvl', None)
        self.ec_conf_thr = self.test_cfg.get('ec_conf_thr', 0.01)
        self.tc_conf_thr = self.test_cfg.get('tc_conf_thr', 0.1)
        self.valid_size_range = self.test_cfg.get('valid_size_range', None) 
        self.list_num_dets_per_lvl = self.test_cfg.get('num_dets_per_lvl', None)

        if self.test_cache_cfg is not None:
            if not os.path.exists(self.test_cache_cfg['root']):
                os.mkdir(self.test_cache_cfg['root'])
            self.save_test_o = True
            self.__debug = True

        if self.__debug:
            self.data_queue = queue.Queue()
            for _ in range(4):
                t = Thread(
                    target=ThreadSaveCacheFile,
                    args=(self.data_queue, self.test_cache_cfg)
                )
                t.daemon = True
                t.start()

    def _make_layers(self, layer_cfgs, in_channels=256):
        # layer_cfgs: tuple(tuple(layer_type, tuple|list(layer_mode, param)))
        convs = nn.ModuleList()
        in_ch = in_channels
        for cfg in layer_cfgs:
            layer_type, layer_param = cfg
            if layer_type == 'conv':
                assert isinstance(layer_param, (tuple, list))
                mode, out_ch = layer_param[:2]
                conv_cfg = None
                norm_cfg = self.norm_cfg
                act_cfg = dict(type='ReLU')
                if mode == 'DCNv2':
                    conv_cfg = dict(type='DCNv2')
                elif mode == 'out':
                    act_cfg = None
                    norm_cfg = None
                elif mode == 'LReLU':
                    act_cfg = dict(type='LeakyReLU')
                conv = ConvModule(
                    in_ch,
                    out_ch,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg
                )
                if mode == 'drop':
                    if len(layer_param) > 2:
                        warmup = layer_param[2]
                    else:
                        warmup = 5000
                    dropblock = DropBlock(0.25, 5, warmup)
                    convs.append(dropblock)
                in_ch = out_ch
                convs.append(conv)
            elif layer_type == 'upsample':
                assert isinstance(layer_param, (tuple, list))
                mode, s_factor = layer_param[:2]
                up_sample = nn.Upsample(scale_factor=s_factor, mode=mode, align_corners=False)
                convs.append(up_sample)
            else:
                raise NotImplementedError

        return convs

    def _init_extreme_pts_layers(self):

        self.longside_center = self._make_layers(self.longside_center_cfg, self.in_channels)
        self.shortside_center = self._make_layers(self.shortside_center_cfg, self.in_channels)
        self.target_center = self._make_layers(self.target_center_cfg, self.in_channels)

    def _init_center_pointer_layers(self):

        self.center_pointer = self._make_layers(self.center_pointer_cfg, self.in_channels)

    def _init_ec_offset_layers(self):

        self.sc_offset = self._make_layers(self.ec_offset_cfg, 256)
        self.lc_offset = self._make_layers(self.ec_offset_cfg, 256)

    def _init_layers(self):

        self._init_extreme_pts_layers()
        self._init_center_pointer_layers()
        self._init_ec_offset_layers()

    def init_weights(self):

        bias_init = bias_init_with_prob(0.1)

        non_init_classes = (nn.Upsample, DropBlock)

        for m in self.longside_center:
            if isinstance(m, non_init_classes):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.longside_center[-1].conv.bias.data.fill_(bias_init)

        for m in self.shortside_center:
            if isinstance(m, non_init_classes):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.shortside_center[-1].conv.bias.data.fill_(bias_init)

        for m in self.target_center:
            if isinstance(m, non_init_classes):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.target_center[-1].conv.bias.data.fill_(bias_init)       

        for m in self.center_pointer:
            if isinstance(m, non_init_classes):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.center_pointer[-1].conv.bias.data.fill_(bias_init)      

        # init ec offset
        for m in self.sc_offset:
            if isinstance(m, non_init_classes):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.sc_offset[-1].conv.bias.data.fill_(bias_init) 

        for m in self.lc_offset:
            if isinstance(m, non_init_classes):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.lc_offset[-1].conv.bias.data.fill_(bias_init)
        
    def forward_single(self, x):
        '''
            during training, all_cls_scores, all_bbox_preds, all_center_dets are replaced with lc, sc, tc
        '''
        lc, sc, tc = x, x, x

        if not self.kpt_enable:
            lc = lc.detach()
            sc = sc.detach()

        for layer in self.longside_center[:-1]:
            lc = layer(lc)
        for layer in self.shortside_center[:-1]:
            sc = layer(sc)

        lc_res = self.longside_center[-1](lc)
        sc_res = self.shortside_center[-1](sc)    

        if not self.refine_enable:
            lc = lc.detach()
            sc = sc.detach()

        off_lc = self.lc_offset[-1](lc)
        off_sc = self.sc_offset[-1](sc)

        for layer in self.target_center[:-1]:
            tc = layer(tc)

        for layer in self.center_pointer:
            ctx_ptr = layer(tc)
        tc = self.target_center[-1](tc)

        offset = torch.cat([off_sc, off_lc], dim=1)

        result_list = [lc_res, sc_res, tc, ctx_ptr, offset]

        return result_list

    def forward(self, feats):

        if self.training:
            return multi_apply(self.forward_single, feats) 
        else:
            test_feats = []
            for ind in self.test_feat_ind:
                test_feats.append(feats[ind])
            return multi_apply(self.forward_single, test_feats) 

    def _get_targets_single(self, 
                            center_pointer,
                            gt_bboxes,
                            gt_labels,
                            img_meta,
                            reg_range,
                            feat_shape):
        """
        center_pointer: [8, feat_h, feat_w], first dim: (sc_x, sc_y, ..., lc_x, lc_y, ...)
        ec_offset: [4, feat_h, feat_w], first dim: (sc_off_x, sc_off_y, lc_off_x, lc_off_y)
        """
        _, _, feat_h, feat_w = feat_shape
        img_h, img_w, _ = img_meta.get('pad_shape', None) if not None else img_meta['img_shape']
        # TODO: support dynamic feature size
        stride_h, stride_w = 4, 4#img_h / feat_h, img_w / feat_w
        # gt heatmaps: 0:target center 1: shortside center 2:longside center
        gt_heatmaps = [gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))  for _ in range(3)]
        gt_offsets = gt_bboxes.new_zeros((self.ec_offset_cfg[-1][1][1]*2, feat_h, feat_w),
                                  dtype=torch.float32, device=gt_bboxes.device)         
        # gt center pointer
        gt_center_pointer = gt_bboxes.new_zeros((8, feat_h, feat_w))
        # sort valid gt bboxes for this level
        reg_min, reg_max = reg_range[0] * np.sqrt(img_h * img_w),\
                           reg_range[1] * np.sqrt(img_h * img_w)
        valid_index = sort_valid_gt_bboxes(gt_bboxes, (reg_min, reg_max))        
        scaled_gt_bboxes = gt_bboxes.clone()
        # rescale the gt_bboxes to fit the feature map size
        scaled_gt_bboxes[:, 0] = scaled_gt_bboxes[:, 0] / stride_w
        scaled_gt_bboxes[:, 1] = scaled_gt_bboxes[:, 1] / stride_h
        scaled_gt_bboxes[:, 2:4] = scaled_gt_bboxes[:, 2:4] / np.sqrt(stride_h * stride_w)

        # seperate scaled valid and unvalid gt_bbox
        valid_gt_boxes = scaled_gt_bboxes[valid_index]
        valid_gt_labels = gt_labels[valid_index]
        unvalid_gt_boxes = scaled_gt_bboxes[~valid_index]

        # generate target map
        target_map = get_target_map(unvalid_gt_boxes, (feat_w, feat_h), device=gt_bboxes.device)
    
        # get corner points for each rbox
        corner_pts = []
        train_gt_boxes = []

        for gt_bbox, gt_label in zip(valid_gt_boxes, valid_gt_labels):
            x, y, w, h, a = float(gt_bbox[0]), float(gt_bbox[1]), float(gt_bbox[2]), float(gt_bbox[3]), float(gt_bbox[4])
            a = a * 180 / np.pi
            pts = cv2.boxPoints(((x, y), (w, h), a))
            if w < 1 or h < 1:
                continue
            corner_pts.append(pts)
            train_gt_boxes.append((gt_bbox, gt_label))

        # calculate edge centers for each rbox
        corner_pts = torch.tensor(corner_pts, dtype=torch.float32, device=gt_bboxes.device)
        if corner_pts.shape[0] is not 0:
            long_side_center, short_side_center, target_center, num_box = generate_ec_from_corner_pts(corner_pts)

        for k, gt_bbox_label in enumerate(train_gt_boxes):
            # set guassion maps for center points
            gt_bbox, gt_label = gt_bbox_label
            tc = target_center[k],
            sc = short_side_center[k], short_side_center[k + num_box]
            lc = long_side_center[k], long_side_center[k + num_box]
            w, h, a = gt_bbox[2], gt_bbox[3], gt_bbox[4]
            # ablation study of circular target
            if self.circular_target:
                m_e = min(w, h)
                w, h = m_e , m_e
            for v, centers in enumerate((tc, sc, lc)):
                for center in centers:
                    x, y = center[0], center[1]
                    if x < 0 or y < 0 or x >= feat_w or y >=feat_h:
                        continue
                    cat = int(gt_label)
                    gt_heatmaps[v][cat,...] = gen_gaussian_targetR(gt_heatmaps[v][cat,...],
                                                              x, y, w, h, a, self.sigma_ratio)
                    if v > 0:
                        gt_offsets[v*2-2: v*2] =  set_offset2(center, gt_offsets[v*2-2: v*2], w, h, a, self.sigma_ratio)

            gt_center_pointer = generate_center_pointer_map2(tc[0], sc, lc, center_pointer, gt_center_pointer, w, h, a, self.sigma_ratio, self.rigid_edge_pointer)

        target_center_heat = gt_heatmaps[0]
        shortside_center_heat = gt_heatmaps[1]
        longside_center_heat = gt_heatmaps[2]

        return target_center_heat, shortside_center_heat, longside_center_heat, target_map, gt_center_pointer, gt_offsets

    def get_targets(self,
                    center_pointer,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    feat_shapes):

        multi_lvl_feat_targets = []
        if gt_masks_list == None:
            gt_masks_list = [None for _ in range(len(gt_bboxes_list))]
        for k, feat_shape in enumerate(feat_shapes):
            batched_feat_targets = []
            target = multi_apply(self._get_targets_single,
                                center_pointer[k],
                                gt_bboxes_list,
                                gt_labels_list,
                                img_metas,
                                reg_range=self.regress_ratios[k],
                                feat_shape=feat_shape)
            for t in target:
                batched_feat_target = torch.stack(t, dim=0)
                batched_feat_targets.append(batched_feat_target)
            target_result = dict(
                gt_target_center = batched_feat_targets[0],
                gt_shortside_center = batched_feat_targets[1],
                gt_longside_center = batched_feat_targets[2],
                target_map = batched_feat_targets[3],
                gt_center_pointer = batched_feat_targets[4],
                gt_offsets = batched_feat_targets[5]
            )
            multi_lvl_feat_targets.append(target_result)

        return multi_lvl_feat_targets

    def loss_heat_single(self, lc, sc, tc, targets):
        
        target_map = targets['target_map']
        target_index = (target_map == 1).unsqueeze(1)
        gt_longside_center = targets['gt_longside_center']
        gt_shortside_center = targets['gt_shortside_center']
        gt_target_center = targets['gt_target_center']

        gt_longside_center = gt_longside_center[target_index]
        gt_shortside_center = gt_shortside_center[target_index]
        gt_target_center = gt_target_center[target_index]

        lc = lc[target_index]
        sc = sc[target_index]
        tc = tc[target_index]


        # detection loss
        lc_det_loss = self.loss_heatmap(
            lc.sigmoid(),
            gt_longside_center,
            avg_factor = max(1, gt_longside_center.eq(1).sum())
        )
        sc_det_loss = self.loss_heatmap(
            sc.sigmoid(),
            gt_shortside_center,
            avg_factor = max(1, gt_shortside_center.eq(1).sum())
        )
        tc_det_loss = self.loss_heatmap(
            tc.sigmoid(),
            gt_target_center,
            avg_factor = max(1, gt_target_center.eq(1).sum())
        )

        return lc_det_loss / 3, sc_det_loss / 3, tc_det_loss / 3

    def loss_reg_single(self, ctx_ptr, ec_offset, targets):

        gt_center_pointer = targets['gt_center_pointer']
        gt_offsets = targets['gt_offsets']
        gt_center_pointer_mask = gt_center_pointer.ne(0).sum(1).gt(0).unsqueeze(1).type_as(ctx_ptr)
        gt_offset_mask = gt_offsets.ne(0).type_as(ec_offset)
        pointer_loss = self.loss_pointer(ctx_ptr,
                                        gt_center_pointer,
                                        gt_center_pointer_mask,
                                        avg_factor=max(1, gt_center_pointer_mask.sum()))
        offset_loss = self.loss_offsets(ec_offset,
                                        gt_offsets,
                                        gt_offset_mask,
                                        avg_factor=max(1, gt_center_pointer_mask.sum()))
        return pointer_loss, offset_loss

    def loss(self,
             longside_center_heats,
             shortside_center_heats,
             target_center_heats,
             center_pointer,
             ec_offset,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes_ignore=None):

        targets = self.get_targets( center_pointer, gt_bboxes, gt_labels, gt_masks, img_metas,
                                    [feat.shape for feat in target_center_heats])

        lc_det_loss, sc_det_loss, tc_det_loss = multi_apply(self.loss_heat_single,
                                longside_center_heats,
                                shortside_center_heats,
                                target_center_heats,
                                targets)

        pointer_loss, offset_loss= multi_apply(self.loss_reg_single,
                                    center_pointer,
                                    ec_offset,
                                    targets)

        loss_dict = dict(lc_det_loss=lc_det_loss,
                         sc_det_loss=sc_det_loss,
                         tc_det_loss=tc_det_loss,
                         offset_loss=offset_loss,
                         pointer_loss=pointer_loss)

        return loss_dict

    def decode_heatmap_single(self,
                            tc_heat,
                            lc_heat,
                            sc_heat,
                            ctx_ptr,
                            ec_offset,
                            k_pts,
                            num_dets,
                            ec_conf_thr,
                            tc_conf_thr,
                            kernel=3,
                            **kwargs):
        # TODO:support multi-class detection

        batch, _, height, width = tc_heat.size()
        if k_pts < 1:
            return tc_heat.new_zeros((batch, 0, 8)), tc_heat.new_zeros((batch, 0, 1)), tc_heat.new_zeros((batch, 0, 1))

        lc_heat, sc_heat, tc_heat = lc_heat.sigmoid(), sc_heat.sigmoid(), tc_heat.sigmoid()

        tc_heat = get_local_maximum(tc_heat, kernel=kernel)
        sc_heat = get_local_maximum(sc_heat, kernel=kernel)
        lc_heat = get_local_maximum(lc_heat, kernel=kernel)

        tc_scores, tc_inds, tc_clses, tc_ys, tc_xs = get_topk_from_heatmap(tc_heat, k=k_pts)
        sc_scores, _, _, sc_ys, sc_xs = get_topk_from_heatmap(sc_heat, k=k_pts)
        lc_scores, _, _, lc_ys, lc_xs = get_topk_from_heatmap(lc_heat, k=k_pts)

        tc_pos = torch.stack((tc_xs, tc_ys), dim=-1) + 0.5
        sc_pos = torch.stack((sc_xs, sc_ys), dim=-1) + 0.5
        lc_pos = torch.stack((lc_xs, lc_ys), dim=-1) + 0.5

        center_pointers = transpose_and_gather_feat(ctx_ptr, tc_inds)
        center_pointers = center_pointers.view(batch, -1, 4, 2)
        tc_pointer_res = tc_pos.unsqueeze(-2).repeat(1,1,4,1) + center_pointers

        tc_pointer_res_rpt = tc_pointer_res.unsqueeze(-1).repeat(1,1,1,1,k_pts).transpose(-1,-2)
        sc_pos_rpt = sc_pos.unsqueeze(1).unsqueeze(1).repeat(1,k_pts,2,1,1)
        lc_pos_rpt = lc_pos.unsqueeze(1).unsqueeze(1).repeat(1,k_pts,2,1,1)
        sc_ptr_l2_dist = torch.norm(tc_pointer_res_rpt[:,:,:2,:,:] - sc_pos_rpt, dim=-1)
        lc_ptr_l2_dist = torch.norm(tc_pointer_res_rpt[:,:,2:,:,:] - lc_pos_rpt, dim=-1)

        # normalize l2 dist
        sc_ptr_l2_dist = sc_ptr_l2_dist / np.math.sqrt(width * height)
        lc_ptr_l2_dist = lc_ptr_l2_dist / np.math.sqrt(width * height)

        # generate scores
        sc_scores_rpt = sc_scores.unsqueeze(1).unsqueeze(1).repeat(1,k_pts,2,1)
        lc_scores_rpt = lc_scores.unsqueeze(1).unsqueeze(1).repeat(1,k_pts,2,1)
        # calculate final score using gaussian function
        # final_score = score * e ^ -(x^2 / 2*ptr_dist^2)
        sc_scores_rpt_final = sc_scores_rpt * torch.exp(-sc_ptr_l2_dist.pow(2) / (2*ctx_ptr.new_tensor(self.sc_ptr_sigma).pow(2)))
        lc_scores_rpt_final = lc_scores_rpt * torch.exp(-lc_ptr_l2_dist.pow(2) / (2*ctx_ptr.new_tensor(self.lc_ptr_sigma).pow(2)))
        sc_kpt_score, sc_kpt_ind = sc_scores_rpt_final.max(dim=-1, keepdim=True)
        lc_kpt_score, lc_kpt_ind = lc_scores_rpt_final.max(dim=-1, keepdim=True)
        sc_kpt_pos = sc_pos_rpt.gather(3, sc_kpt_ind.unsqueeze(-1).repeat(1,1,1,1,2)).squeeze(3)
        lc_kpt_pos = lc_pos_rpt.gather(3, lc_kpt_ind.unsqueeze(-1).repeat(1,1,1,1,2)).squeeze(3)
        # select key point or regressed edge center based on score
        sc_kpt_pos = torch.where(sc_kpt_score > ec_conf_thr, sc_kpt_pos, tc_pointer_res[:,:,:2,:])
        lc_kpt_pos = torch.where(lc_kpt_score > ec_conf_thr, lc_kpt_pos, tc_pointer_res[:,:,2:,:])
        sc_kpt_score = torch.where(sc_kpt_score > ec_conf_thr, sc_kpt_score, sc_kpt_score.new_full(sc_kpt_score.shape, ec_conf_thr))
        lc_kpt_score = torch.where(lc_kpt_score > ec_conf_thr, lc_kpt_score, lc_kpt_score.new_full(lc_kpt_score.shape, ec_conf_thr))

        # sample offset using grid_sample
        # normalize kpt pos to [-1, 1]
        pos_norm_scaler = sc_kpt_pos.new_tensor([width // 2, height // 2])
        sc_kpt_pos_norm = (sc_kpt_pos - pos_norm_scaler) / pos_norm_scaler
        lc_kpt_pos_norm = (lc_kpt_pos - pos_norm_scaler) / pos_norm_scaler
        sc_kpt_off = F.grid_sample(ec_offset[:,:2,...], sc_kpt_pos_norm, 'bilinear', 'zeros', align_corners=False)
        lc_kpt_off = F.grid_sample(ec_offset[:,2:,...], lc_kpt_pos_norm, 'bilinear', 'zeros', align_corners=False)
        sc_kpt_pos = sc_kpt_pos.floor() + 0.5 
        lc_kpt_pos = lc_kpt_pos.floor() + 0.5 
        if self.refine_enable:
            sc_kpt_pos = sc_kpt_pos + sc_kpt_off.permute(0,2,3,1)
            lc_kpt_pos = lc_kpt_pos + lc_kpt_off.permute(0,2,3,1)

        # generate final results
        scores = (tc_scores * 2 + sc_kpt_score.squeeze(-1).sum(-1) + lc_kpt_score.squeeze(-1).sum(-1)) / 6

        # reject box has low center score
        low_score_ind = tc_scores < tc_conf_thr
        scores -= low_score_ind.float()

        bboxes = torch.cat((sc_kpt_pos, lc_kpt_pos), dim=2)
        clses = tc_clses.view(batch, -1, 1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = bboxes.view(batch, -1, 8)
        bboxes = gather_feat(bboxes, inds)

        clses = gather_feat(clses, inds).float()
        # normalize bbox predictions to range (0, 1)
        bboxes[..., 0::2] = bboxes[..., 0::2] / width
        bboxes[..., 1::2] = bboxes[..., 1::2] / height

        return bboxes.clamp(0., 1.), scores, clses

    def _get_bboxes_single(self,
                            all_lvl_bbox_kpts,
                            all_lvl_bbox_scores,
                            all_lvl_bbox_clses,
                            img_meta,
                            valid_size_range,
                            rescale=False,
                            with_nms=True):

        num_lvl = len(all_lvl_bbox_kpts)
        
        all_lvl_rboxes_sc = []
        # all_lvl_rboxes_lc = []
        for bbox_kpts in all_lvl_bbox_kpts:
            bbox_kpts = bbox_kpts.clamp(0, 1)
            rboxes_sc = keypoints2rbboxes(bbox_kpts, sc_first=True)
            # rboxes_lc = keypoints2rbboxes(bbox_kpts, sc_first=False)
            all_lvl_rboxes_sc.append(rboxes_sc)
            # all_lvl_rboxes_lc.append(rboxes_lc)

        if valid_size_range is not None:
            for i in range(num_lvl):
                valid_ind = (all_lvl_rboxes_sc[i][...,2] > valid_size_range[i][0]) & \
                            (all_lvl_rboxes_sc[i][...,2] < valid_size_range[i][1])
                all_lvl_rboxes_sc[i] = all_lvl_rboxes_sc[i][valid_ind][None]
                # all_lvl_rboxes_lc[i] = all_lvl_rboxes_lc[i][valid_ind][None]
                all_lvl_bbox_scores[i] = all_lvl_bbox_scores[i][valid_ind][None]
                all_lvl_bbox_clses[i] = all_lvl_bbox_clses[i][valid_ind][None]

        all_lvl_rboxes = all_lvl_rboxes_sc #+ all_lvl_rboxes_lc
        all_lvl_bbox_scores = all_lvl_bbox_scores #+ [score * 0.99 for score in all_lvl_bbox_scores]
        all_lvl_bbox_clses = all_lvl_bbox_clses #+ all_lvl_bbox_clses
        det_rboxes = torch.cat(all_lvl_rboxes, dim=1)
        det_scores = torch.cat(all_lvl_bbox_scores, dim=1)
        det_clses = torch.cat(all_lvl_bbox_clses, dim=1)

        valid_score = det_scores > 0
        keep_ind = valid_score[...,0]
        # # keep boxes based on ratio
        valid_ratio = ((det_rboxes[...,2] / det_rboxes[...,3]) < 15 ) & \
                    ((det_rboxes[...,2] / det_rboxes[...,3]) > 0.8 )
        keep_ind =keep_ind & valid_ratio
        det_rboxes = det_rboxes[keep_ind]
        det_scores = det_scores[keep_ind]
        det_clses = det_clses[keep_ind]

        img_shape = img_meta.get('pad_shape', None) if not None else img_meta['img_shape']
        # TODO: May contain error here!! 
        img_h, img_w, _ = img_shape
        det_rboxes[..., 0] = det_rboxes[..., 0] * img_w
        det_rboxes[..., 1] = det_rboxes[..., 1] * img_h
        det_rboxes[..., 2:4] = det_rboxes[..., 2:4] * np.sqrt(img_h * img_w)
        if rescale:
            img_shape = img_meta['img_shape'][:2]
            ori_shape = img_meta['ori_shape'][:2]
            stride_h = img_shape[0] / ori_shape[0]
            stride_w = img_shape[1] / ori_shape[1]
            stride_l = np.sqrt(stride_h * stride_w)
            det_rboxes[..., :4] = det_rboxes[..., :4] /\
                                  det_rboxes.new_tensor([stride_w, stride_h, stride_l, stride_l])

        if det_rboxes.size(0) < 1:
            # add a default empty result for eval
            # TODO:if there are multiple classes, this should be changed
            det_rboxes = det_rboxes.new_zeros((1,5))
            det_clses = det_clses.new_zeros((1,1))
            det_scores = det_scores.new_zeros((1,1)) * -1.

        if with_nms:
            _nms_cfg = self.test_cfg.get('nms', dict(type='rnms', iou_thr=0.05))
            padding = det_scores.new_zeros(det_scores.shape[0], 1)
            det_scores = torch.cat([det_scores, padding], dim=1)
            det_rboxes, det_clses = multiclass_nms_rotated(det_rboxes, det_scores, self.test_cfg.score_thr, _nms_cfg, self.test_cfg.max_per_img)
        
            return det_rboxes, det_clses
        
        else:
            return det_rboxes, det_scores

    def get_bboxes(self,
                    list_longside_center,
                    list_shortside_center,
                    list_target_center,
                    list_center_pointer,
                    list_ec_offset,
                    img_metas,
                    rescale=False,
                    with_nms=True):
        
        result_list = []
        if self.test_cache_cfg is not None:
            data_dict=dict(
                # Test Outputs
                type='TeO',
                data=dict(
                    lc=list_longside_center,
                    sc=list_shortside_center,
                    tc=list_target_center,
                    cp=list_center_pointer,
                    off=list_ec_offset,
                ),
                img_metas=img_metas
            )

            self.data_queue.put(data_dict)

        list_num_pkts_per_lvl = [self.list_num_pkts_per_lvl[-1]]
        list_num_dets_per_lvl = [self.list_num_dets_per_lvl[-1]]
        valid_size_range = [self.valid_size_range[-1]]

        multi_lvl_bbox_kpts, multi_lvl_scores, multi_lvl_clses = multi_apply(
            self.decode_heatmap_single,
            list_target_center,
            list_longside_center,
            list_shortside_center,
            list_center_pointer,
            list_ec_offset,
            list_num_pkts_per_lvl,
            list_num_dets_per_lvl,
            ec_conf_thr=self.ec_conf_thr,
            tc_conf_thr=self.tc_conf_thr,
        )
        for img_id in range(len(img_metas)):
            all_lvl_bbox_kpts = [kpts[img_id][None] for kpts in multi_lvl_bbox_kpts]
            all_lvl_bbox_scores = [scores[img_id][None] for scores in multi_lvl_scores]
            all_lvl_bbox_clses = [clses[img_id][None] for clses in multi_lvl_clses]
            results = self._get_bboxes_single(all_lvl_bbox_kpts,
                                            all_lvl_bbox_scores,
                                            all_lvl_bbox_clses,
                                            img_metas[img_id],
                                            valid_size_range,
                                            rescale=rescale,
                                            with_nms=with_nms)
            result_list.append(results)
        return result_list



    def forward_train(self,
                        x,
                        img_metas,
                        gt_bboxes,
                        gt_labels=None,
                        gt_masks=None,
                        gt_bboxes_ignore=None,
                        proposal_cfg=None,
                        **kwargs):
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_labels, gt_masks, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(*outs[:6], img_metas, cfg=proposal_cfg, extra_data=outs[6:])
            return losses, proposal_list
    
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
