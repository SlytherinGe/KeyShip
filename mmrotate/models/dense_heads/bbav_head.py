from turtle import forward
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmdet.models.dense_heads.base_dense_head import BaseDenseHead
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS, build_loss

@ROTATED_HEADS.register_module()
class BBAVHead(BaseDenseHead):

    def __init__(self,  num_classes,
                        in_channels,
                        feat_channels=256,
                        loss_heatmap=dict(
                            type='GaussianFocalLoss',
                            alpha=2.0,
                            gamma=4.0,
                            loss_weight=1                               
                        ),
                        loss_offset=dict(
                            type='SmoothL1Loss', beta=1.0, loss_weight=1
                        ),
                        loss_rbox=dict(
                            type='SmoothL1Loss', beta=1.0, loss_weight=1
                        ),
                        loss_theta=dict(
                            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1
                        ),
                        train_cfg=None,
                        test_cfg=None,
                        norm_cfg=None,
                        init_cfg=None):
        super().__init__(init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.loss_heatmap = None if loss_heatmap == None else build_loss(loss_heatmap)
        self.loss_offset = None if loss_offset == None else build_loss(loss_offset)
        self.loss_rbox = None if loss_rbox == None else build_loss(loss_rbox)
        self.loss_theta = None if loss_theta == None else build_loss(loss_theta)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.norm_cfg = norm_cfg

        self._init_layers()

    def _init_layers(self):

        act_cfg = dict(type='ReLU')
        self.heatmap_layers = nn.ModuleList([
            ConvModule(self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        act_cfg=act_cfg,
                        norm_cfg=self.norm_cfg),
            ConvModule(self.feat_channels,
                        self.num_classes,
                        kernel_size=1,
                        act_cfg=None,
                        norm_cfg=None)
        ])
        self.offset_layers = nn.ModuleList([
            ConvModule(self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        act_cfg=act_cfg,
                        norm_cfg=self.norm_cfg),
            ConvModule(self.feat_channels,
                        2,
                        kernel_size=1,
                        act_cfg=None,
                        norm_cfg=None)
        ])
        self.rbox_layers = nn.ModuleList([
            ConvModule(self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        act_cfg=act_cfg,
                        norm_cfg=self.norm_cfg),
            ConvModule(self.feat_channels,
                        10,
                        kernel_size=3,
                        padding=1,
                        act_cfg=None,
                        norm_cfg=None)
        ])
        self.theta_layers = nn.ModuleList([
            ConvModule(self.in_channels,
                        self.feat_channels,
                        kernel_size=3,
                        padding=1,
                        bias=True,
                        act_cfg=act_cfg,
                        norm_cfg=self.norm_cfg),
            ConvModule(self.feat_channels,
                        1,
                        kernel_size=1,
                        act_cfg=None,
                        norm_cfg=None)
        ])

    def init_weights(self):
        super().init_weights()

        for m in self.heatmap_layers:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)            
        self.heatmap_layers[-1].conv.bias.data.fill_(-2.19)

        for m in self.offset_layers:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)            
        self.offset_layers[-1].conv.bias.data.fill_(0.)

        for m in self.rbox_layers:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)            
        self.rbox_layers[-1].conv.bias.data.fill_(0.)

        for m in self.theta_layers:
            if isinstance(m.conv, nn.Conv2d):
                normal_init(m.conv, std=0.01)            
        self.theta_layers[-1].conv.bias.data.fill_(0.)

    def forward(self, x):

        heat, off, rbox, theta = x, x, x, x

        for layer in self.heatmap_layers:
            heat = layer(heat)
        
        for layer in self.offset_layers:
            off = layer(off)

        for layer in self.rbox_layers:
            rbox = layer(rbox)

        for layer in self.theta_layers:
            theta = layer(theta)     

        return [heat], [off], [rbox], [theta]  

    def get_targets(self, gt_bboxes_list:list,
                          gt_labels_list:list,
                          img_metas:list,
                          feat_shape:list):

        num_imgs = len(img_metas)
        

        return

    @force_fp32()
    def loss(self,
             heatmap,
             offset,
             rbox,
             theta,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):


        return