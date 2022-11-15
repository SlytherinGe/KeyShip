import torch
import torch.nn as nn
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmcv.cnn import ConvModule
from mmdet.core import multi_apply
from mmrotate.core.bbox.transforms import obb2poly, poly2obb
from mmcv.ops import min_area_polygons
from mmrotate.core import (multiclass_nms_rotated)
from ..builder import ROTATED_HEADS, build_loss
from ..utils import (get_local_maximum, get_topk_from_heatmap, transpose_and_gather_feat)
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
class StrongScatteringHead(BaseDenseHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 num_feats=256,
                 up_sample_rate=16,
                 loss_heatmap=dict(
                     type='GaussianFocalLoss',
                     alpha=2.0,
                     gamma=4.0,
                     loss_weight=1                     
                    ),
                 loss_embedding=dict(
                     type='SmoothL1Loss', beta=1.0, loss_weight=1
                 ),
                 train_cfg=None,
                 test_cfg=None,
                 norm_cfg=None,
                 init_cfg=dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01)
                ):
        super(StrongScatteringHead, self).__init__(init_cfg)
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_feats = num_feats
        self.up_sample_rate = up_sample_rate
        self.loss_heatmap = build_loss(loss_heatmap) if loss_heatmap is not None else None
        self.loss_embedding = build_loss(loss_embedding) if loss_embedding is not None else None
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()
        
        # init cache saving
        self.train_cache_cfg = self.train_cfg.get('cache_cfg', None)
        self.test_cache_cfg = self.test_cfg.get('cache_cfg', None)
        self.__debug = False
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

    def _init_layers(self):

        # init Ship Attention Layers
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.ConvB = ConvModule(self.in_channels,
                                self.num_feats,
                                kernel_size=3,
                                padding=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='ReLU'))
        self.ConvC = ConvModule(self.in_channels,
                                self.num_feats,
                                kernel_size=3,
                                padding=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='ReLU'))
        self.ConvD = ConvModule(self.in_channels,
                                self.num_feats,
                                kernel_size=3,
                                padding=1,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=dict(type='ReLU'))
        self.ConvE = ConvModule(self.num_feats,
                                self.num_feats,
                                kernel_size=1,
                                padding=0,
                                norm_cfg=dict(type='BN', requires_grad=True),
                                act_cfg=None)

        # init Detection and Grouping heads
        self.ConvStem = ConvModule( self.num_feats,                               
                                self.num_feats,
                                kernel_size=3,
                                padding=1,
                                norm_cfg=self.norm_cfg,
                                act_cfg=dict(type='ReLU'))
        self.ConvHeatmap = nn.ModuleList(  [ ConvModule(self.num_feats,
                                                self.num_feats,
                                                kernel_size=3,
                                                padding=1,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=dict(type='ReLU')),
                                            ConvModule(self.num_feats,
                                                self.num_classes,
                                                kernel_size=1,
                                                padding=0,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=None),
                                            nn.Upsample(scale_factor=self.up_sample_rate, 
                                                        mode='bilinear', 
                                                        align_corners=False)])
        self.ConvEmbedding = nn.ModuleList( [ConvModule(self.num_feats,
                                                self.num_feats,
                                                kernel_size=3,
                                                padding=1,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=dict(type='ReLU')),
                                            ConvModule(self.num_feats,
                                                1,
                                                kernel_size=1,
                                                padding=0,
                                                norm_cfg=self.norm_cfg,
                                                act_cfg=None),
                                            nn.Upsample(scale_factor=self.up_sample_rate, 
                                                        mode='bilinear', 
                                                        align_corners=False)])


    def forward(self, inputs):

        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]
        else:
            x = inputs
        batch, num_feats, feat_h, feat_w = x.size()
        # forward precedure for Ship Attention Module
        feat_D = self.ConvD(x)
        feat_C = self.ConvC(x)
        feat_B = self.ConvB(x)
        feat_E = self.ConvE(feat_D)
        feat_B_f = feat_B.view(batch, num_feats, -1).transpose(1,2)
        feat_C_f = feat_C.view(batch, num_feats, -1)
        feat_F = torch.bmm(feat_B_f, feat_C_f)
        feat_F = torch.softmax(feat_F, dim=1)
        feat_F = torch.bmm(feat_D.view(batch, num_feats, -1), feat_F)

        feat_G = feat_F.view(batch, num_feats, feat_h, feat_w)*self.alpha +\
                 feat_E * self.beta + x
        feat_G = torch.relu(feat_G)

        # forward precedure for Detection and Grouping Heads
        feat_stem = feat_G + self.ConvStem(feat_G)

        feat_heatmap, feat_embedding = feat_stem, feat_stem
        for layer in self.ConvHeatmap:
            feat_heatmap = layer(feat_heatmap)
        for layer in self.ConvEmbedding:
            feat_embedding = layer(feat_embedding)

        return [feat_heatmap], [feat_embedding]
    
    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels=None,
                      gt_masks=None,
                      gt_bboxes_ignore=None,
                      proposal_cfg=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): Features from FPN.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            proposal_cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used

        Returns:
            tuple:
                losses: (dict[str, Tensor]): A dictionary of loss components.
                proposal_list (list[Tensor]): Proposals of each image.
        """
        outs = self(x)
        if gt_labels is None:
            loss_inputs = outs + (gt_bboxes, gt_masks , img_metas)
        else:
            loss_inputs = outs + (gt_bboxes, gt_masks , gt_labels, img_metas)
        losses = self.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        if proposal_cfg is None:
            return losses
        else:
            proposal_list = self.get_bboxes(
                *outs, img_metas=img_metas, cfg=proposal_cfg)
            return losses, proposal_list
        
    def get_targets(self,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_masks_list,
                    img_metas,
                    feat_shapes):
    
        for feat_shape in feat_shapes:
            target_heatmaps, target_instances = multi_apply(self._get_targets_single,
                                                            gt_bboxes_list,
                                                            gt_labels_list,
                                                            gt_masks_list,
                                                            img_metas,
                                                            feat_shape=feat_shape)
            target_results = dict(
                target_heatmaps = torch.stack(target_heatmaps, dim=0),
                target_instances = target_instances
            )                           
        
        return target_results
        
    def _get_targets_single(self,
                            gt_bboxes,
                            gt_labels,
                            gt_masks,
                            img_meta,
                            feat_shape):
        
        _, _, feat_h, feat_w = feat_shape
        img_h, img_w, _ = img_meta.get('pad_shape', None) if not None else img_meta['img_shape']
        stride_h, stride_w = img_h / feat_h, img_w / feat_w
        scaled_gt_masks = gt_masks.to_tensor(dtype=torch.int32, device=gt_bboxes.device)
        instance_inds = []
        heatmap_target = scaled_gt_masks.sum(0).ge(1).type_as(gt_bboxes)
        num_targets = len(gt_bboxes)
        for i in range(num_targets):
            inds = scaled_gt_masks[i].nonzero()
            inds = inds[:,0] * feat_w + inds[:,1]
            if len(inds)>0:
                instance_inds.append(inds)
        
        return  heatmap_target, instance_inds
        
    def loss(self,
                heatmaps,
                embeddings,
                gt_bboxes,
                gt_masks,
                gt_labels,
                img_metas,
                gt_bboxes_ignore=None):
        
        targets = self.get_targets(gt_bboxes, gt_labels, gt_masks,img_metas,
                                   [feat.shape for feat in heatmaps])
        
        # loss for heatmap
        target_heatmaps = targets['target_heatmaps']
        heatmap_loss = self.loss_heatmap(heatmaps[-1].sigmoid(),
                                         target_heatmaps,
                                         avg_factor = max(1, target_heatmaps.eq(1).sum()))
        # loss for embeddings
        embedding_instances = targets['target_instances']
        embedding_pull_loss, embedding_push_loss = self.loss_embedding(embeddings[-1],
                                                                    embedding_instances)
        
        loss_dict = dict(
            heatmap_loss=heatmap_loss,
            embedding_push_loss=embedding_push_loss,
            embedding_pull_loss=embedding_pull_loss
        )
        
        return loss_dict
    
    def get_bboxes(self,
                    list_feat_heatmap,
                    list_feat_embedding,
                    img_metas,
                    rescale=False,
                    with_nms=True):
        
        result_list = []
        if self.test_cache_cfg is not None:
            data_dict=dict(
                # Test Outputs
                type='TeO',
                data=dict(
                    hm=list_feat_heatmap,
                    embed=list_feat_embedding
                ),
                img_metas=img_metas
            )
            self.data_queue.put(data_dict)

        # result_list = []
        result_list = [(torch.zeros((1, 6), device=list_feat_heatmap[-1].device, dtype=torch.float32),
                        torch.zeros((1,), device=list_feat_heatmap[-1].device, dtype=torch.int64))]

        return result_list