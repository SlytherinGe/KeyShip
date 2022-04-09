import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, bias_init_with_prob, normal_init
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
import cv2
from mmcv.runner.base_module import BaseModule
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmcv.cnn import xavier_init
from mmcv.cnn import Linear, build_activation_layer
from mmcv.cnn.bricks.transformer import FFN, build_positional_encoding

from mmrotate.core import (aug_multiclass_nms_rotated, bbox_mapping_back,
                           build_assigner, build_bbox_coder,
                           build_prior_generator, build_sampler,
                           multiclass_nms_rotated, obb2hbb,
                           rotated_anchor_inside_flags)
from ..builder import ROTATED_HEADS, build_loss
from mmdet.core import multi_apply
from mmdet.core.bbox.match_costs import build_match_cost
from ..utils import (gen_gaussian_targetR, get_local_maximum,
                     get_topk_from_heatmap, gather_feat,
                     transpose_and_gather_feat, keypoints2rbboxes,
                     sort_valid_gt_bboxes, get_target_map,
                     generate_self_conjugate_data, 
                     generate_cross_paired_data)

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

INF = 1e8
SMALL_NUM = 1e-6
# multi-thread for saving heatmaps
import mmcv
import os
import queue
from threading import Thread

class ClusFormer(BaseModule):

    def __init__(self, num_classes,
                       encoder=None, 
                       decoder=None, 
                       act_cfg=dict(type='ReLU', inplace=True),
                       init_cfg=None):
        super().__init__(init_cfg)
        self.num_classes = num_classes
        self.encoder = build_transformer_layer_sequence(encoder)
        self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims
        self.act_cfg = act_cfg
        self.activation = build_activation_layer(self.act_cfg)
        self._init_layers()

    def _init_layers(self):
        '''Initialize layers of the clusformer head'''
        self.fc_cls = Linear(256, self.num_classes+1)
        self.ffn_reg = FFN(act_cfg=self.act_cfg)
        self.fc_reg = Linear(256, 8)

    def init_weights(self):

        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True   
        return super().init_weights()         

    def forward(self, ec_vecs, tc_vecs, ec_pos_embed=None, tc_pos_embed=None):
        """Forward function for `Transformer`.

        Args:
            ec_vecs (Tensor): Input ec center query with shape
                [2*num_queries, batch_size, embed_dims] where
            tc_vecs (Tensor): Input ec center query with shape
                [num_queries, batch_size, embed_dims] where
            ec_pos_embed (Tensor): Sampled positional embedding for ec
                with the shape of [2*num_queries, batch_size, embed_dims].
            ec_pos_embed (Tensor): Sampled positional embedding for ec
                with the shape of [num_queries, batch_size, embed_dims].

        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.

                - cls_score: Class output from decoder. 0 means background object.
                      with shape [batch_size, num_queries, num_class+1]
                - bbox_pred: Normalized bbox prediction from decoder, with shape \
                      [batch_size, num_queries, 8]. The last dims contains the 
                      following pos info: [SC1_x, SC1_y, SC2_x, SC2_y, LC1_x, LC1_y,
                      LC2_x, LC2_y]
        """
        ec_vec_embeded = ec_vecs
        tc_vec_embeded = tc_vecs

        memory = self.encoder(ec_vec_embeded, 
                              key=None, 
                              value=None, 
                              query_pos=ec_pos_embed)
        out_dec = self.decoder(query=tc_vec_embeded, 
                               key=memory, 
                               value=memory, 
                               key_pos=ec_pos_embed, 
                               query_pos=tc_pos_embed)
        out_dec = out_dec.permute(1,0,2)
        all_cls_scores = self.fc_cls(out_dec)
        all_bbox_preds = self.fc_reg(self.activation(self.ffn_reg(out_dec))).sigmoid()

        return all_cls_scores, all_bbox_preds

class HungarianAssigner(object):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of one components:
    regression L2 cost. The targets don't include the no_object, so generally
    there are more predictions than targets. After the one-to-one matching,
    the un-matched are treated as backgrounds. Thus each query prediction will
    be assigned with `0` or a positive integer indicating the ground truth index:

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt

    Args:
        reg_cost (dict, optional): The config dict for key points matching
    """

    def __init__(self,
                 reg_cost=dict(type='KeypointL2Cost', weight=1.0)):

        self.reg_cost = build_match_cost(reg_cost)

    def assign(self,
               kpt_pred,
               gt_kpts,
               eps=1e-7):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred (Tensor): Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            img_meta (dict): Meta information for current image.
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`. Default None.
            eps (int | float, optional): A value added to the denominator for
                numerical stability. Default 1e-7.

        Returns:
            :obj:`AssignResult`: The assigned result.
        """

        num_gts, num_bboxes = gt_kpts.size(0), kpt_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = kpt_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
                return assigned_gt_inds

        # 2. compute the weighted costs
        # regression L1 cost
        reg_cost = self.reg_cost(kpt_pred, gt_kpts)
        # weighted sum of above three costs
        cost = reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        if linear_sum_assignment is None:
            raise ImportError('Please run "pip install scipy" '
                              'to install scipy first.')
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            kpt_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            kpt_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        return assigned_gt_inds

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
class ExtremeHeadV3(BaseDenseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 longside_center_cfg,
                 shortside_center_cfg,
                 target_center_cfg,
                 clusformer_cfg=None,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True),
                 regress_ratio=((-1, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 1)),
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1                     
                    ),
                 loss_clusformer_cls=dict(
                                type='CrossEntropyLoss',
                                use_sigmoid=False,
                                loss_weight=1.0),
                 loss_clusformer_reg=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=5.0), 
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                ):
        super(ExtremeHeadV3, self).__init__()
        # self.head_modules = []
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.regress_ratios = regress_ratio
        # init losses
        self.loss_heatmap = None if loss_heatmap == None else build_loss(loss_heatmap)
        self.loss_clusformer_cls = None if loss_clusformer_cls == None else build_loss(loss_clusformer_cls)
        self.loss_clusformer_reg = None if loss_clusformer_reg == None else build_loss(loss_clusformer_reg)

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

        self._init_layers()

        # init Clusformer
        self.clusformer = ClusFormer(num_classes,
                                     clusformer_cfg.get('encoder_cfg', None),
                                     clusformer_cfg.get('decoder_cfg', None))
        self.num_queries = clusformer_cfg.get('num_queries', 60)
        self.positional_encoder = build_positional_encoding(positional_encoding)

        # init assigner
        self.assigner = HungarianAssigner()

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

    def _init_layers(self):

        self._init_extreme_pts_layers()
    
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
        
    def forward_single(self, x):

        lc, sc, tc = x, x, x,

        for layer in self.longside_center:
            lc = layer(lc)
        for layer in self.shortside_center:
            sc = layer(sc)
        for layer in self.target_center:
            tc = layer(tc)

        # clusformer forward
        batch, _, feat_h, feat_w = x.size()
        mask = x.new_zeros((batch, feat_h, feat_w))
        pos_embedding = self.positional_encoder(mask).to(x.device).expand_as(x).detach()
        lc_peak = get_local_maximum(lc)
        sc_peak = get_local_maximum(sc)
        tc_peak = get_local_maximum(tc)
        _, topk_inds_tc, _, topk_ys_tc, topk_xs_tc = get_topk_from_heatmap(tc_peak, self.num_queries)
        _, topk_inds_sc, _, _, _ = get_topk_from_heatmap(sc_peak, self.num_queries)
        _, topk_inds_lc, _, _, _ = get_topk_from_heatmap(lc_peak, self.num_queries)

        # peak_feat shape [batch_size, num_peak, num_feat]
        peak_feat_tc = transpose_and_gather_feat(x, topk_inds_tc)
        peak_feat_sc = transpose_and_gather_feat(x, topk_inds_sc)
        peak_feat_lc = transpose_and_gather_feat(x, topk_inds_lc)
        # get the embeded position for each peak
        peak_pos_tc = transpose_and_gather_feat(pos_embedding, topk_inds_tc)
        peak_pos_sc = transpose_and_gather_feat(pos_embedding, topk_inds_sc)
        peak_pos_lc = transpose_and_gather_feat(pos_embedding, topk_inds_lc)

        batched_feat_ec = torch.cat([peak_feat_lc, peak_feat_sc], dim=1).permute(1,0,2)
        batched_pos_ec = torch.cat([peak_pos_lc, peak_pos_sc], dim=1).permute(1,0,2)
        batched_feat_tc = peak_feat_tc.permute(1,0,2)
        batched_pos_tc = peak_pos_tc.permute(1,0,2)

        all_center_dets = (torch.stack([topk_xs_tc, topk_ys_tc], dim=-1) + 0.5) / x.new_tensor([feat_w, feat_h])

        all_cls_scores, all_bbox_preds = self.clusformer(batched_feat_ec, batched_feat_tc, batched_pos_ec, batched_pos_tc)
        # fomat result
        result_list = [lc, sc, tc, all_cls_scores, all_bbox_preds, all_center_dets]

        return  result_list

    def forward(self, feats):

        return multi_apply(self.forward_single, feats)


    def get_targets(self,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    all_cls_scores_list, 
                    all_bbox_preds_list,
                    all_center_dets_list,
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
                                all_cls_scores_list[k], 
                                all_bbox_preds_list[k],
                                all_center_dets_list[k],
                                reg_range=self.regress_ratios[k],
                                feat_shape=feat_shape,
                                img_shape=img_shape)
            for t in target[:6]:
                batched_feat_target = torch.stack(t, dim=0)
                batched_feat_targets.append(batched_feat_target)
            target_result = dict(
                gt_target_center = batched_feat_targets[0],
                gt_shortside_center = batched_feat_targets[1],
                gt_longside_center = batched_feat_targets[2],
                target_map = batched_feat_targets[3],
                clusformer_ec_target = batched_feat_targets[4],
                clusformer_label_target = batched_feat_targets[5]
            )
            multi_lvl_feat_targets.append(target_result)

        return multi_lvl_feat_targets

    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            all_cls_scores, 
                            all_bbox_preds,
                            all_center_dets,
                            reg_range,
                            feat_shape,
                            img_shape):
            
        _, _, feat_h, feat_w = feat_shape
        img_h, img_w, _ = img_shape
        stride_h, stride_w = img_h / feat_h, img_w / feat_w
        # gt heatmaps: 0:target center 1: shortside center 2:longside center
        gt_heatmaps = [gt_bboxes.new_zeros((self.num_classes, feat_h, feat_w))  for _ in range(3)]
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

        # generate targets for clusformer
        long_side_center = long_side_center.view(2, num_box, 2).permute(1,0,2).contiguous()
        short_side_center = short_side_center.view(2, num_box, 2).permute(1,0,2).contiguous()
        assigned_gt_inds = self.assigner.assign(all_center_dets, target_center)
        clusformer_label_target = assigned_gt_inds.gt(0).long()
        clusformer_ec_target = torch.zeros_like(all_bbox_preds)
        pos_inds = torch.nonzero(assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        valid_bbox_pred = all_bbox_preds[pos_inds].view(-1, 4, 2)
        # determine the relation between a pair of ec
        valid_bbox_lc = long_side_center[assigned_gt_inds[pos_inds]-1]
        valid_bbox_sc = short_side_center[assigned_gt_inds[pos_inds]-1]
        valid_bbox_lc_r = torch.flip(valid_bbox_lc, [1])
        valid_bbox_sc_r = torch.flip(valid_bbox_sc, [1])
        gt_lc0_dist = torch.norm(valid_bbox_pred[:,:2,:]-valid_bbox_lc, dim=-1).sum(-1)
        gt_lc0_dist_r = torch.norm(valid_bbox_pred[:,:2,:]-valid_bbox_lc_r, dim=-1).sum(-1)
        gt_sc0_dist = torch.norm(valid_bbox_pred[:,2:,:]-valid_bbox_sc, dim=-1).sum(-1)
        gt_sc0_dist_r = torch.norm(valid_bbox_pred[:,2:,:]-valid_bbox_sc_r, dim=-1).sum(-1)
        target_lc = torch.where((gt_lc0_dist < gt_lc0_dist_r).view(-1,1,1), valid_bbox_lc, valid_bbox_lc_r)
        target_sc = torch.where((gt_sc0_dist < gt_sc0_dist_r).view(-1,1,1), valid_bbox_sc, valid_bbox_sc_r)            
        target_bboxes = torch.cat([target_lc, target_sc], dim=1)

        clusformer_ec_target[pos_inds] = target_bboxes.view(-1, 8)

        target_center_heat = gt_heatmaps[0]
        shortside_center_heat = gt_heatmaps[1]
        longside_center_heat = gt_heatmaps[2]
        target_map = target_map[None]

        return (target_center_heat, 
                shortside_center_heat, 
                longside_center_heat, 
                target_map,
                clusformer_ec_target,
                clusformer_label_target)

    def loss_heat_single(self, lc, sc, tc, targets):
        
        target_map = targets['target_map']
        target_index = (target_map == 1)
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
    
    def loss_clusformer_single(self, all_cls_scores, all_bbox_preds, target):

        clusformer_ec_target = target['clusformer_ec_target']
        clusformer_label_target = target['clusformer_label_target']

        valid_inds = clusformer_label_target.gt(0).float()
        avg_factor = valid_inds.sum()
        clusformer_reg_loss = self.loss_clusformer_reg(
            all_bbox_preds,
            clusformer_ec_target,
            valid_inds.unsqueeze(-1),
            avg_factor
        )
        clusformer_cls_loss = self.loss_clusformer_cls(
            all_cls_scores.view(-1, 2),
            clusformer_label_target.view(-1),
            avg_factor=avg_factor
        )

        return clusformer_reg_loss / 2, clusformer_cls_loss / 2

    def loss(self,
             longside_center_heats,
             shortside_center_heats,
             target_center_heats,
             all_cls_scores, 
             all_bbox_preds,
             all_center_dets,
             gt_bboxes,
             gt_labels,
             gt_masks,
             img_metas,
             gt_bboxes_ignore=None):

        targets = self.get_targets( gt_bboxes, gt_labels, gt_masks,
                                    all_cls_scores, 
                                    all_bbox_preds,
                                    all_center_dets,
                                    [feat.shape for feat in target_center_heats],
                                    img_metas[0]['pad_shape'])

        lc_det_loss, sc_det_loss, tc_det_loss = multi_apply(self.loss_heat_single,
                                longside_center_heats,
                                shortside_center_heats,
                                target_center_heats,
                                targets)

        clusformer_reg_loss, clusformer_cls_loss = multi_apply(self.loss_clusformer_single,
                                all_cls_scores, 
                                all_bbox_preds,
                                targets)

        loss_dict = dict(lc_det_loss=lc_det_loss,
                         sc_det_loss=sc_det_loss,
                         tc_det_loss=tc_det_loss,
                         clusformer_cls_loss=clusformer_cls_loss,
                         clusformer_reg_loss=clusformer_reg_loss)

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
                ),
                img_metas=img_metas,
            )
            self.data_queue.put(data_dict)

        return loss_dict

    def decode_clusformer_single(self,
                                all_cls_scores,
                                all_bbox_preds,
                                all_center_dets,
                                num_dets):

        batch = all_bbox_preds.size(0)
        all_cls_scores = all_cls_scores.softmax(dim=-1)
        # valid_ind = all_cls_scores[...,1] > 0.5
        valid_bbox_preds = all_bbox_preds    
        valid_bbox_scores = all_cls_scores[..., 1]
        valid_bbox_clses = torch.zeros_like(valid_bbox_scores)

        scores, inds = torch.topk(valid_bbox_scores, num_dets)
        scores = scores.unsqueeze(2)

        bboxes = valid_bbox_preds.view(batch, -1, 8)
        bboxes = gather_feat(bboxes, inds)

        clses = valid_bbox_clses.unsqueeze(-1)
        clses = gather_feat(clses, inds).float()

        return bboxes, scores, clses

    def _get_bboxes_single(self,
                            all_lvl_bbox_kpts,
                            all_lvl_bbox_scores,
                            all_lvl_bbox_clses,
                            img_meta,
                            valid_size_range,
                            rescale=False,
                            with_nms=True):

        num_lvl = len(all_lvl_bbox_kpts)
        
        all_lvl_rboxes = []
        for bbox_kpts in all_lvl_bbox_kpts:
            rboxes = keypoints2rbboxes(bbox_kpts, sc_first=True)
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
                    list_all_cls_scores,
                    list_all_bbox_preds,
                    list_all_center_dets,
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
                ),
                boxes=dict(
                    bbox_pred=list_all_bbox_preds,
                    box_cls=list_all_cls_scores                    
                ),
                img_metas=img_metas
            )
            # if extra_data is not None:
            #     for k, arg in enumerate(extra_data):
            #         data_dict['data']['extra_{}'.format(k)] = arg
            self.data_queue.put(data_dict)

        assert len(list_longside_center) == len(list_shortside_center)  and \
            len(list_shortside_center) == len(list_target_center) and \
            len(list_longside_center) == len(list_all_cls_scores) and \
            len(list_all_cls_scores) == len(list_all_bbox_preds) and \
            len(list_all_bbox_preds) == len(list_all_center_dets) and \
            len(list_all_center_dets) == len(self.list_num_pkts_per_lvl)

        if self.valid_size_range is not None:
            assert len(self.valid_size_range) == len(list_target_center)

        multi_lvl_bbox_kpts, multi_lvl_scores, multi_lvl_clses = multi_apply(
            self.decode_clusformer_single,
            list_all_cls_scores,
            list_all_bbox_preds,
            list_all_center_dets,
            self.list_num_pkts_per_lvl,
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