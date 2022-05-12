import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
import cv2

from mmrotate.core import (aug_multiclass_nms_rotated, bbox_mapping_back,
                           build_assigner, build_bbox_coder,
                           build_prior_generator, build_sampler,
                           multiclass_nms_rotated, obb2hbb,
                           rotated_anchor_inside_flags)
from ..builder import ROTATED_HEADS, build_loss
from mmdet.core import multi_apply
from ..utils import (gen_gaussian_targetR, get_local_maximum,
                     get_topk_from_heatmap, gather_feat,
                     transpose_and_gather_feat, keypoints2rbboxes,
                     sort_valid_gt_bboxes, get_target_map,
                     set_offset, set_centripetal_shifts,
                     generate_self_conjugate_data, 
                     generate_cross_paired_data)

INF = 1e8
SMALL_NUM = 1e-6
# multi-thread for saving heatmaps
import mmcv
import os
import queue
from threading import Thread

def ThreadSaveCacheFile(dq, train_cache_cfg, test_cache_cfg):

    if test_cache_cfg is None:
        test_cache_cfg = dict()
    TeO_root = test_cache_cfg.get('root', '/home/gejunyao/ramdisk/TestCache')
    path_dict = dict(
        TeO=TeO_root
    )
    if train_cache_cfg is not None:
        train_root = train_cache_cfg.get('root', '/home/gejunyao/ramdisk/TrainCache')
        TrT_root = os.path.join(train_root, 'TrainTargets')
        TrO_root = os.path.join(train_root, 'TrainOutputs')
        path_dict.update(TrT=TrT_root)
        path_dict.update(TrO=TrO_root)

    while True:
        data = dq.get()
        save_folder = path_dict[data['type']]
        save_d = data['data']
        img_metas = data['img_metas']
        num_img = len(img_metas)
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


        dq.task_done()

@ROTATED_HEADS.register_module()
class ExtremeHeadV2(BaseDenseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 longside_center_cfg,
                 shortside_center_cfg,
                 target_center_cfg,
                 offset_cfg,
                 centipital_shift_cfg,
                 centipital_shift_channels=2,
                 regress_ratio=((-1, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1)),
                 offset_types = ['sc', 'lc', 'tc'],
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1                     
                    ),
                 loss_offsets=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1
                 ),
                 loss_centripetal_shift=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1),
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                ):
        super(ExtremeHeadV2, self).__init__()
        # self.head_modules = []
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.centipital_shift_channels = centipital_shift_channels
        self.offset_types = offset_types
        self.regress_ratios = regress_ratio
        # init losses
        self.loss_heatmap = None if loss_heatmap == None else build_loss(loss_heatmap)
        self.loss_offsets = None if loss_offsets == None else build_loss(loss_offsets)
        self.loss_centripetal_shift = None if loss_offsets == None else build_loss(loss_centripetal_shift)

        self.train_cfg = train_cfg if train_cfg is not None else dict()
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg
        self.sigma_ratio = self.train_cfg.get('gaussioan_sigma_ratio', (0.1, 0.1))

        self.train_cache_cfg = self.train_cfg.get('cache_cfg', None)
        self.test_cache_cfg = self.test_cfg.get('cache_cfg', None)
        self.__debug = False

        self.longside_center_cfg = longside_center_cfg
        self.shortside_center_cfg = shortside_center_cfg
        self.target_center_cfg = target_center_cfg
        self.offset_cfg = offset_cfg
        self.centipital_shift_cfg = centipital_shift_cfg

        self._init_layers()

        # test cfg init
        self.list_num_pkts_per_lvl = self.test_cfg.get('num_kpts_per_lvl', None)
        self.ec_conf_thr = self.test_cfg.get('ec_conf_thr', 0.01)
        self.tc_conf_thr = self.test_cfg.get('tc_conf_thr', 0.1)
        # self.valid_size_range = [(-1, 0.2), (0.05, 0.4), (0.1, 0.8), (0.2, 1), (0.4, 2)] 
        self.valid_size_range = self.test_cfg.get('valid_size_range', None) 
        self.list_num_dets_per_lvl = self.test_cfg.get('num_dets_per_lvl', None)

        if self.train_cache_cfg is not None:
            root = self.train_cache_cfg.get('root', '/home/gejunyao/ramdisk/TrainCache')
            if not os.path.exists(root):
                os.mkdir(root)
            if self.train_cache_cfg.get('save_target', False):
                save_folder = 'TrainTargets'
                if not os.path.exists(os.path.join(root, save_folder)):
                    os.mkdir(os.path.join(root, save_folder))
                self.save_target = True
            if self.train_cache_cfg.get('save_output', False):
                save_folder = 'TrainOutputs'
                if not os.path.exists(os.path.join(root, save_folder)):
                    os.mkdir(os.path.join(root, save_folder))
                self.save_train_o = True      
            self.__debug = True

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
                    args=(self.data_queue, self.train_cache_cfg, self.test_cache_cfg)
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
                conv = ConvModule(
                    in_ch,
                    out_ch,
                    3,
                    padding=1,
                    norm_cfg=norm_cfg,
                    conv_cfg=conv_cfg,
                    act_cfg=act_cfg
                )
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
        # self.head_modules.append('longside_center')
        self.shortside_center = self._make_layers(self.shortside_center_cfg, self.in_channels)
        # self.head_modules.append('shortside_center')
        self.target_center = self._make_layers(self.target_center_cfg, self.in_channels)
        # self.head_modules.append('target_center')

    def _init_offset_layers(self):

        self.offsets = self._make_layers(self.offset_cfg, self.in_channels)
        # self.head_modules.append('offsets')

    def _init_centipital_shift_layers(self):

        self.lc_centipital_shift = self._make_layers(self.centipital_shift_cfg, self.in_channels)
        # self.head_modules.append('lc_centipital_shift')
        self.sc_centipital_shift = self._make_layers(self.centipital_shift_cfg, self.in_channels)
        # self.head_modules.append('sc_centipital_shift')

    def _init_layers(self):

        self._init_extreme_pts_layers()
        self._init_offset_layers()
        self._init_centipital_shift_layers()
    
    def init_weights(self):
        bias_init = bias_init_with_prob(0.1)

        for m in self.longside_center:
            if isinstance(m, nn.Upsample):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.longside_center[-1].conv.bias.data.fill_(bias_init)

        for m in self.shortside_center:
            if isinstance(m, nn.Upsample):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.shortside_center[-1].conv.bias.data.fill_(bias_init)

        for m in self.target_center:
            if isinstance(m, nn.Upsample):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.target_center[-1].conv.bias.data.fill_(bias_init)

        for m in self.offsets:
            if isinstance(m, nn.Upsample):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.offsets[-1].conv.bias.data.fill_(bias_init)

        for m in self.lc_centipital_shift:
            if isinstance(m, nn.Upsample):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.lc_centipital_shift[-1].conv.bias.data.fill_(bias_init)

        for m in self.sc_centipital_shift:
            if isinstance(m, nn.Upsample):
                continue
            elif isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)
        self.sc_centipital_shift[-1].conv.bias.data.fill_(bias_init)
        
    def forward_single(self, x):

        lc, sc, tc, offset, lc_centi_shift, sc_centi_shift = x, x, x, x, x, x

        for layer in self.longside_center:
            lc = layer(lc)
        for layer in self.shortside_center:
            sc = layer(sc)
        for layer in self.target_center:
            tc = layer(tc)
        for layer in self.offsets:
            offset = layer(offset)
        for layer in self.lc_centipital_shift:
            lc_centi_shift = layer(lc_centi_shift)
        for layer in self.sc_centipital_shift:
            sc_centi_shift = layer(sc_centi_shift)

        result_list = [lc, sc, tc, offset, lc_centi_shift, sc_centi_shift]

        return  result_list

    def forward(self, feats):

        return multi_apply(self.forward_single, feats)


    def get_targets(self,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    feat_shapes,
                    img_shape):

        multi_lvl_feat_targets = []
        if gt_masks_list == None:
            gt_masks_list = [None for _ in range(len(gt_bboxes_list))]
        for k, feat_shape in enumerate(feat_shapes):
            batched_feat_targets = []
            target = multi_apply(self._get_targets_single,
                                gt_bboxes_list,
                                gt_labels_list,
                                gt_masks_list,
                                reg_range=self.regress_ratios[k],
                                feat_shape=feat_shape,
                                img_shape=img_shape)
            for t in target[:7]:
                batched_feat_target = torch.stack(t, dim=0)
                batched_feat_targets.append(batched_feat_target)
            target_result = dict(
                gt_target_center = batched_feat_targets[0],
                gt_shortside_center = batched_feat_targets[1],
                gt_longside_center = batched_feat_targets[2],
                target_map = batched_feat_targets[3],
                gt_offsets = batched_feat_targets[4],
                gt_sc_centripetal_shift = batched_feat_targets[5],
                gt_lc_centripetal_shift = batched_feat_targets[6]
            )
            multi_lvl_feat_targets.append(target_result)

        return multi_lvl_feat_targets

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            reg_range,
                            feat_shape,
                            img_shape):
            
        _, _, feat_h, feat_w = feat_shape
        img_h, img_w, _ = img_shape
        stride_h, stride_w = img_h / feat_h, img_w / feat_w
        # gt heatmaps: 0:target center 1: shortside center 2:longside center
        gt_heatmaps = [gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))  for _ in range(3)]
        # gt offsets: the first demension is the same as gt heatmaps
        # in the second demension index 0->x, 1->y
        # BUG: *2 在 len 里面
        gt_offsets = gt_bboxes.new_zeros((len(self.offset_types*2), feat_h, feat_w),
                                  dtype=torch.float32, device=gt_bboxes.device) 
        gt_centripetals = [gt_bboxes.new_zeros((self.centipital_shift_channels, feat_h, feat_w)) for _ in range(2)]
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
            d_12 = torch.square(corner_pts[:,0,0]-corner_pts[:,1,0]) + \
                torch.square(corner_pts[:,0,1]-corner_pts[:,1,1])
            d_23 = torch.square(corner_pts[:,1,0]-corner_pts[:,2,0]) + \
                torch.square(corner_pts[:,1,1]-corner_pts[:,2,1])
            is_d23_longer = d_12 < d_23
            num_box = len(is_d23_longer)
            box_index = torch.arange(num_box, device=gt_bboxes.device)
            box_index = torch.hstack((box_index, box_index)) * 4
            long_side_start_index = torch.as_tensor(is_d23_longer, dtype=torch.int, device=gt_bboxes.device)
            long_side_start_index = torch.hstack((long_side_start_index, long_side_start_index + 2))
            long_side_end_index = (long_side_start_index + 1) % 4
            short_side_start_index = long_side_end_index
            short_side_end_index = (short_side_start_index + 1) % 4
            corner_pts = corner_pts.reshape(-1, 2)
            long_side_center = (corner_pts[box_index + long_side_start_index,:] +
                                corner_pts[box_index + long_side_end_index,:]) / 2
            short_side_center = (corner_pts[box_index + short_side_start_index,:] +
                                corner_pts[box_index + short_side_end_index,:]) / 2 
            target_center = (corner_pts[box_index[:num_box], :] + 
                            corner_pts[box_index[:num_box] + 2, :]) / 2
        for k, gt_bbox_label in enumerate(train_gt_boxes):
            # set guassion maps for center points
            gt_bbox, gt_label = gt_bbox_label
            tc = target_center[k],
            sc = short_side_center[k], short_side_center[k + num_box]
            lc = long_side_center[k], long_side_center[k + num_box]
            w, h, a = gt_bbox[2], gt_bbox[3], gt_bbox[4]
            for v, centers in enumerate((tc, sc, lc)):
                for center in centers:
                    x, y = center[0], center[1]
                    if x < 0 or y < 0 or x >= feat_w or y >=feat_h:
                        continue
                    cat = int(gt_label)
                    gt_heatmaps[v][cat,...] = gen_gaussian_targetR(gt_heatmaps[v][cat,...],
                                                              x, y, w, h, a, self.sigma_ratio)
                    offset_cat = ('tc', 'sc', 'lc')[v]
                    if offset_cat in self.offset_types:
                        off_idx = self.offset_types.index(offset_cat)
                        gt_offsets[off_idx*2:off_idx*2+2,...] = set_offset(gt_offsets[off_idx*2:off_idx*2+2,...], x, y)
            # set gt centripetal shifts
            for v, centers in enumerate((sc, lc)):
                for center in centers:
                    x, y = center[0], center[1]
                    if x < 0 or y < 0 or x >= feat_w or y >=feat_h:
                        continue
                    gt_centripetals[v] = set_centripetal_shifts(gt_centripetals[v], x, y, tc[0][0], tc[0][1])

        target_center_heat = gt_heatmaps[0]
        shortside_center_heat = gt_heatmaps[1]
        longside_center_heat = gt_heatmaps[2]
        target_map = target_map[None]
        sc_centripetal_shift = gt_centripetals[0]
        lc_centripetal_shift = gt_centripetals[1]

        return (target_center_heat, 
                shortside_center_heat, 
                longside_center_heat, 
                target_map,
                gt_offsets,
                sc_centripetal_shift,
                lc_centripetal_shift)

    def loss_heat_single(self, lc, sc, tc, offsets, lc_centri_shift, sc_centri_shift, targets):
        
        target_map = targets['target_map']
        target_index = (target_map == 1)
        gt_longside_center = targets['gt_longside_center']
        gt_shortside_center = targets['gt_shortside_center']
        gt_target_center = targets['gt_target_center']
        gt_offsets = targets['gt_offsets']
        gt_sc_centripetal_shift = targets['gt_sc_centripetal_shift']
        gt_lc_centripetal_shift = targets['gt_lc_centripetal_shift']

        offset_mask = gt_offsets.ne(0).type_as(gt_longside_center)
        # none_zero = gt_offsets.nonzero()
        # longside_offset_pos = gt_longside_center.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_longside_center)
        # shortside_offset_pos = gt_shortside_center.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_shortside_center)
        # target_offset_pos = gt_target_center.eq(1).sum(1).gt(0).unsqueeze(1).type_as(gt_target_center)

        # offset_mask = torch.cat([target_offset_pos, 
        #                          target_offset_pos,
        #                          shortside_offset_pos,
        #                          shortside_offset_pos,
        #                          longside_offset_pos,
        #                          longside_offset_pos], dim=1)

        sc_centri_shift_mask = gt_sc_centripetal_shift.ne(0).sum(1).gt(0).unsqueeze(1).type_as(gt_longside_center)
        lc_centri_shift_mask = gt_lc_centripetal_shift.ne(0).sum(1).gt(0).unsqueeze(1).type_as(gt_longside_center)

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
        # offset loss
        offset_loss = self.loss_offsets(
            offsets,
            gt_offsets,
            offset_mask,
            avg_factor=max(1, offset_mask.sum())
        )

        sc_centri_shift_loss = self.loss_centripetal_shift(
            sc_centri_shift,
            gt_sc_centripetal_shift,
            sc_centri_shift_mask,
            avg_factor=max(1, sc_centri_shift_mask.sum())
        )

        lc_centri_shift_loss = self.loss_centripetal_shift(
            lc_centri_shift,
            gt_lc_centripetal_shift,
            lc_centri_shift_mask,
            avg_factor=max(1, sc_centri_shift_mask.sum())
        )

        return lc_det_loss / 3, sc_det_loss / 3, tc_det_loss / 3, offset_loss, (sc_centri_shift_loss + lc_centri_shift_loss)/2

    def loss(self,
             longside_center_heats,
             shortside_center_heats,
             target_center_heats,
             offsets,
             lc_centri_shift, 
             sc_centri_shift,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes_ignore=None):

        targets = self.get_targets( gt_bboxes, gt_labels, gt_masks,
                                    [feat.shape for feat in target_center_heats],
                                    img_metas[0]['pad_shape'])

        lc_det_loss, sc_det_loss, tc_det_loss, offset_loss, centripetal_loss = multi_apply(self.loss_heat_single,
                                longside_center_heats,
                                shortside_center_heats,
                                target_center_heats,
                                offsets,
                                lc_centri_shift, 
                                sc_centri_shift,
                                targets)

        loss_dict = dict(lc_det_loss=lc_det_loss,
                         sc_det_loss=sc_det_loss,
                         tc_det_loss=tc_det_loss,
                         offset_loss=offset_loss,
                         centripetal_loss=centripetal_loss)

        if self.train_cache_cfg.get('save_target', False):
            data_dict = dict(
                # Training Targets
                # BUG:targets are list(dict) instead of dict(list)
                type='TrT',
                data=targets,
                img_metas=img_metas
            )
            self.data_queue.put(data_dict)

        if self.train_cache_cfg.get('save_output', False):
            data_dict = dict(
                # Training Outputs
                type='TrO',
                data=dict(
                    lc=longside_center_heats,
                    sc=shortside_center_heats,
                    tc=target_center_heats,
                    off=offsets,
                ),
                img_metas=img_metas,
            )
            self.data_queue.put(data_dict)

        return loss_dict

    def decode_heatmap_single(self,
                            tc_heat,
                            lc_heat,
                            sc_heat,
                            lc_shift,
                            sc_shift,
                            offset,
                            k_pts,
                            num_dets,
                            ec_conf_thr,
                            tc_conf_thr,
                            kernel=3,
                            with_centripetal_shift=True):
        batch, _, height, width = tc_heat.size()

        lc_heat, sc_heat, tc_heat = lc_heat.sigmoid(), sc_heat.sigmoid(), tc_heat.sigmoid()

        lc_heat = get_local_maximum(lc_heat, kernel=kernel)
        sc_heat = get_local_maximum(sc_heat, kernel=kernel)

        lc_scores, lc_inds, lc_clses, lc_ys, lc_xs = get_topk_from_heatmap(lc_heat, k=k_pts)
        sc_scores, sc_inds, sc_clses, sc_ys, sc_xs = get_topk_from_heatmap(sc_heat, k=k_pts)

        # offset = torch.zeros((tc_heat.size(0), 4, tc_heat.size(2), tc_heat.size(3)), device=tc_heat.device)

        sc_off = transpose_and_gather_feat(offset[:,0:2], sc_inds)
        lc_off = transpose_and_gather_feat(offset[:,2:4], lc_inds)

        if with_centripetal_shift:
            lc_centripetal_shift = transpose_and_gather_feat(lc_shift, lc_inds)
            sc_centripetal_shift = transpose_and_gather_feat(sc_shift, sc_inds)

            lc_ctxs = lc_xs + lc_centripetal_shift[..., 0]
            lc_ctys = lc_ys + lc_centripetal_shift[..., 1]
            sc_ctxs = sc_xs + sc_centripetal_shift[..., 0]
            sc_ctys = sc_ys + sc_centripetal_shift[..., 1]
            lc_xs_pair_raw = generate_self_conjugate_data(lc_xs, batch)
            lc_ys_pair_raw = generate_self_conjugate_data(lc_ys, batch)
            sc_xs_pair_raw = generate_self_conjugate_data(sc_xs, batch)
            sc_ys_pair_raw = generate_self_conjugate_data(sc_ys, batch)
            lc_xs_pair_raw, sc_xs_pair_raw = generate_cross_paired_data(lc_xs_pair_raw, sc_xs_pair_raw, batch)
            lc_ys_pair_raw, sc_ys_pair_raw = generate_cross_paired_data(lc_ys_pair_raw, sc_ys_pair_raw, batch)
            # generate unrepeated pairs
            # lc_ctxs_pair, lc_ctys_pair, sc_ctxs_pair, sc_ctys_pair = generate_paired_data(lc_ctxs, lc_ctys, sc_ctxs, sc_ctys, batch, K)
            lc_ctxs_pair = generate_self_conjugate_data(lc_ctxs, batch)
            lc_ctys_pair = generate_self_conjugate_data(lc_ctys, batch)
            sc_ctxs_pair = generate_self_conjugate_data(sc_ctxs, batch)
            sc_ctys_pair = generate_self_conjugate_data(sc_ctys, batch)
            lc_ctxs_pair, sc_ctxs_pair = generate_cross_paired_data(lc_ctxs_pair, sc_ctxs_pair, batch)
            lc_ctys_pair, sc_ctys_pair = generate_cross_paired_data(lc_ctys_pair, sc_ctys_pair, batch)

        # add offsets
        lc_xs = lc_xs + lc_off[..., 0]
        lc_ys = lc_ys + lc_off[..., 1]
        sc_xs = sc_xs + sc_off[..., 0]
        sc_ys = sc_ys + sc_off[..., 1]   
        # generate unrepeated pairs
        lc_xs_pair = generate_self_conjugate_data(lc_xs, batch)
        lc_ys_pair = generate_self_conjugate_data(lc_ys, batch)
        sc_xs_pair = generate_self_conjugate_data(sc_xs, batch)
        sc_ys_pair = generate_self_conjugate_data(sc_ys, batch)
        lc_xs_pair, sc_xs_pair = generate_cross_paired_data(lc_xs_pair, sc_xs_pair, batch)
        lc_ys_pair, sc_ys_pair = generate_cross_paired_data(lc_ys_pair, sc_ys_pair, batch)

        lc_clses_pair = generate_self_conjugate_data(lc_clses, batch)
        sc_clses_pair = generate_self_conjugate_data(sc_clses, batch)
        lc_clses_pair, sc_clses_pair = generate_cross_paired_data(lc_clses_pair, sc_clses_pair, batch)

        lc_scores_pair = generate_self_conjugate_data(lc_scores, batch)
        sc_scores_pair = generate_self_conjugate_data(sc_scores, batch)
        lc_scores_pair, sc_scores_pair = generate_cross_paired_data(lc_scores_pair, sc_scores_pair, batch)

        num_data_pair = sc_scores_pair.size(1)

        # collision detection
        lc_ly = lc_ys_pair[..., 0] - lc_ys_pair[..., 1]
        lc_lx = lc_xs_pair[..., 0] - lc_xs_pair[..., 1]
        sc_ly = sc_ys_pair[..., 0] - sc_ys_pair[..., 1]
        sc_lx = sc_xs_pair[..., 0] - sc_xs_pair[..., 1]

        collision_px = ((lc_xs_pair.sum(-1) + sc_xs_pair.sum(-1) + 1) / 4).long().clamp(0, width-1)
        collision_py = ((lc_ys_pair.sum(-1) + sc_ys_pair.sum(-1) + 1) / 4).long().clamp(0, width-1)

        colli_inds = lc_clses_pair[...,0].long() * (height * width) + \
                    collision_py * width + \
                    collision_px
        colli_inds = colli_inds.view(batch, -1)
        colli_heat = tc_heat.view(batch, -1, 1)
        colli_scores = gather_feat(colli_heat, colli_inds)
        colli_scores = colli_scores.view(batch, num_data_pair, num_data_pair)

        scores = (lc_scores_pair.sum(-1) + sc_scores_pair.sum(-1) + 2*colli_scores) / 6

        # reject boxes based on classes
        cls_inds = (lc_clses_pair[..., 0] != lc_clses_pair[..., 1]) + (sc_clses_pair[...,0] != sc_clses_pair[...,1]) + \
                (lc_clses_pair[..., 0] != sc_clses_pair[..., 0])
        cls_inds = (cls_inds > 0)    

        # reject boxes based on none-collision
        # denom [batch, num_lc_pair, num_sc_pair]
        if with_centripetal_shift != True:
            denom = lc_lx * sc_ly - sc_lx * lc_ly

            none_collision = ((-SMALL_NUM < denom) & (denom < SMALL_NUM)).long()

            denom_pos = denom > 0
            lcsc_ly = lc_ys_pair[..., 1] - sc_ys_pair[..., 1]
            lcsc_lx = lc_xs_pair[..., 1] - sc_xs_pair[..., 1]

            s_numer = lc_lx * lcsc_ly - lc_ly * lcsc_lx
            none_collision = none_collision + ((s_numer < 0) == denom_pos).long()
            t_numer = sc_lx * lcsc_ly - sc_ly * lcsc_lx
            none_collision = none_collision + ((t_numer < 0) == denom_pos).long()
            none_collision = none_collision + \
                (s_numer.abs() > denom.abs()).long() +\
                (t_numer.abs() > denom.abs()).long()
            none_colli_inds = (none_collision > 0)
            scores = scores - none_colli_inds.float()

        # reject boxes based on scores
        low_scor_inds = (lc_scores_pair[..., 0] < ec_conf_thr) + (lc_scores_pair[..., 1] < ec_conf_thr) + \
                        (sc_scores_pair[..., 0] < ec_conf_thr) + (sc_scores_pair[..., 1] < ec_conf_thr) + \
                        (colli_scores < tc_conf_thr)
        low_scor_inds = (low_scor_inds > 0)

        scores = scores - cls_inds.float()
        scores = scores - low_scor_inds.float()

        if with_centripetal_shift:
            # Reject boxes based on centripetal shift.
            # To simplify the code, we reject the points outside the boxes 
            # formed by four centers of each edge center.
            none_centripetal_inds = torch.zeros_like(scores, device=scores.device, dtype=torch.int32)
            # centripetal_region_x1 = (lc_xs_pair + sc_xs_pair) / 2
            # centripetal_region_x2 = (torch.flip(lc_xs_pair, [-1]) + sc_xs_pair) / 2
            # centripetal_region_y1 = (lc_ys_pair + sc_ys_pair) / 2
            # centripetal_region_y2 = (torch.flip(lc_ys_pair, [-1]) + sc_ys_pair) / 2
            centripetal_region_x1 = lc_xs_pair_raw
            centripetal_region_x2 = sc_xs_pair_raw
            centripetal_region_y1 = lc_ys_pair_raw
            centripetal_region_y2 = sc_ys_pair_raw    
            centripetal_region_xs = torch.cat([centripetal_region_x1, centripetal_region_x2], -1)
            centripetal_region_ys = torch.cat([centripetal_region_y1, centripetal_region_y2], -1)
            centripetal_region_zs = torch.zeros_like(centripetal_region_ys, dtype=centripetal_region_ys.dtype)
            centripetal_region_pos = torch.stack([centripetal_region_xs, centripetal_region_ys, centripetal_region_zs], dim=-1)
            shifted_xs = torch.cat([lc_ctxs_pair, sc_ctxs_pair], dim=-1)
            shifted_ys = torch.cat([lc_ctys_pair, sc_ctys_pair], dim=-1)
            shifted_zs = torch.zeros_like(shifted_ys, dtype=shifted_ys.dtype)
            shifted_pos = torch.stack([shifted_xs, shifted_ys, shifted_zs], dim=-1)
            for i in range(4):
                vec_pp = centripetal_region_pos - shifted_pos[..., i, :].unsqueeze(3)
                vec_1 = centripetal_region_pos[..., 2:,:] - centripetal_region_pos[..., 0,:].unsqueeze(3)
                vec_2 = centripetal_region_pos[..., 2:,:] - centripetal_region_pos[..., 1,:].unsqueeze(3)
                AreaA = torch.cross(vec_1[...,0,:], vec_1[...,1,:], dim=-1)[..., -1].abs() + \
                        torch.cross(vec_2[...,0,:], vec_2[...,1,:], dim=-1)[..., -1].abs()
                AreaB = torch.cross(vec_pp[..., 0, :], vec_pp[..., 2, :], dim=-1)[..., -1].abs() + \
                        torch.cross(vec_pp[..., 2, :], vec_pp[..., 1, :], dim=-1)[..., -1].abs() + \
                        torch.cross(vec_pp[..., 1, :], vec_pp[..., 3, :], dim=-1)[..., -1].abs() + \
                        torch.cross(vec_pp[..., 3, :], vec_pp[..., 0, :], dim=-1)[..., -1].abs()
                none_centripetal_inds += (AreaA != AreaB)
            none_centripetal_inds = none_centripetal_inds > 0
            scores = scores - none_centripetal_inds.float()

        scores = scores.view(batch, -1)
        scores, inds = torch.topk(scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = torch.stack((lc_xs_pair[..., 0], lc_ys_pair[..., 0],
                            lc_xs_pair[..., 1], lc_ys_pair[..., 1],
                            sc_xs_pair[..., 0], sc_ys_pair[..., 0],
                            sc_xs_pair[..., 1], sc_ys_pair[..., 1]), dim=-1)

        bboxes = bboxes.view(batch, -1, 8)
        bboxes = gather_feat(bboxes, inds)

        clses = lc_clses_pair[..., -1].contiguous().view(batch, -1, 1)
        clses = gather_feat(clses, inds).float()

        # normalize bbox predictions to range (0, 1)
        bboxes[:, 0::2] = bboxes[:, 0::2] / width
        bboxes[:, 1::2] = bboxes[:, 1::2] / height

        return bboxes, scores, clses

    def _get_bboxes_single(self,
                            all_lvl_bbox_kpts,
                            all_lvl_bbox_scores,
                            all_lvl_bbox_clses,
                            img_meta,
                            valid_size_range,
                            with_centripetal_shift=True,
                            rescale=False,
                            with_nms=True):

        num_lvl = len(all_lvl_bbox_kpts)
        
        all_lvl_rboxes = []
        for bbox_kpts in all_lvl_bbox_kpts:
            rboxes = keypoints2rbboxes(bbox_kpts)
            all_lvl_rboxes.append(rboxes)

        if valid_size_range is not None:
            for i in range(num_lvl):
                valid_ind = (all_lvl_rboxes[i][...,2] > valid_size_range[i][0]) & \
                            (all_lvl_rboxes[i][...,2] < valid_size_range[i][1])
                all_lvl_rboxes[i] = all_lvl_rboxes[i][valid_ind][None]
                all_lvl_bbox_scores[i] = all_lvl_bbox_scores[i][valid_ind][None]
                all_lvl_bbox_clses[i] = all_lvl_bbox_clses[i][valid_ind][None]

        det_rboxes = torch.cat(all_lvl_rboxes, dim=1)
        det_scores = torch.cat(all_lvl_bbox_scores, dim=1)
        det_clses = torch.cat(all_lvl_bbox_clses, dim=1)

        valid_score = det_scores > 0
        keep_ind = valid_score[...,0]
        # # keep boxes based on ratio
        if with_centripetal_shift != True:
            valid_ratio = ((det_rboxes[...,2] / det_rboxes[...,3]) < 8 ) & \
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

        if with_nms:
            _nms_cfg = self.test_cfg.get('nms_cfg', dict(type='rnms', iou_thr=0.05))
            padding = det_scores.new_zeros(det_scores.shape[0], 1)
            det_scores = torch.cat([det_scores, padding], dim=1)
            det_rboxes, det_clses = multiclass_nms_rotated(det_rboxes, det_scores, self.test_cfg.score_thr, _nms_cfg, self.test_cfg.max_per_img)
        else:
            det_rboxes = torch.cat([det_rboxes, det_scores], dim=-1)
        
        if det_rboxes.size(0) < 1:
            # add a default empty result for eval
            det_rboxes = det_rboxes.new_zeros((1,6))
            det_clses = det_clses.new_zeros((1,))

        return det_rboxes, det_clses


    def get_bboxes(self,
                    list_longside_center,
                    list_shortside_center,
                    list_target_center,
                    list_offset,
                    list_lc_centri_shift,
                    list_sc_centri_shift,
                    # list_lc_feat,
                    # list_sc_feat,
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
                    off=list_offset,
                    lc_shift=list_lc_centri_shift,
                    sc_shift=list_sc_centri_shift,
                    # sc_feat = list_sc_feat,
                    # lc_feat = list_lc_feat
                ),
                img_metas=img_metas
            )
            # if extra_data is not None:
            #     for k, arg in enumerate(extra_data):
            #         data_dict['data']['extra_{}'.format(k)] = arg
            self.data_queue.put(data_dict)

        assert len(list_longside_center) == len(list_shortside_center)  and \
            len(list_shortside_center) == len(list_target_center) and \
            len(list_longside_center) == len(list_offset) and \
            len(list_offset) == len(list_lc_centri_shift) and \
            len(list_lc_centri_shift) == len(list_sc_centri_shift) and \
            len(list_offset) == len(self.list_num_pkts_per_lvl) and \
            len(list_offset) == len(self.list_num_dets_per_lvl)

        if self.valid_size_range is not None:
            assert len(self.valid_size_range) == len(list_target_center)

        multi_lvl_bbox_kpts, multi_lvl_scores, multi_lvl_clses = multi_apply(
            self.decode_heatmap_single,
            list_target_center,
            list_longside_center,
            list_shortside_center,
            list_lc_centri_shift,
            list_sc_centri_shift,
            list_offset,
            self.list_num_pkts_per_lvl,
            self.list_num_dets_per_lvl,
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
                                            self.valid_size_range,
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