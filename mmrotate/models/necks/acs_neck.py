import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.runner import BaseModule
from mmcv.cnn import ConvModule
from ..builder import ROTATED_NECKS


@ROTATED_NECKS.register_module()
class ACSNeck(BaseModule):

    def __init__(self, 
                in_channels,
                out_channels,      
                dilations=(1, 3, 6, 9, 1),
                init_cfg=dict(type='Kaiming', layer='Conv2d')):
        super().__init__(init_cfg)
        assert dilations[-1] == 1
        self.stem_conv = ConvModule(in_channels, out_channels, 3, 1, 1,
                                    act_cfg=dict(type='ReLU'),
                                    norm_cfg=dict(type='BN', requires_grad=True))
        self.aspp = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for dilation in dilations:
            kernel_size = 3 if dilation > 1 else 1
            padding = dilation if dilation > 1 else 0
            conv = nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                dilation=dilation,
                padding=padding,
                bias=True)
            self.aspp.append(conv)
            self.batch_norms.append(nn.BatchNorm2d(out_channels))
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, inputs):
        if isinstance(inputs, (list, tuple)):
            x = inputs[-1]
        else:
            x = inputs
        x = self.stem_conv(x)
        avg_x = self.gap(x)
        out = []
        for aspp_idx in range(len(self.aspp) - 1):
            out.append(self.batch_norms[aspp_idx](self.aspp[aspp_idx](x)) + x)
        # compute avg layer
        out.append(F.relu(self.batch_norms[-1](self.aspp[-1](avg_x))))
        out[-1] = out[-1].expand_as(out[-2])
        out_feat = torch.zeros_like(out[-2])
        for o in out:
            out_feat = out_feat + o
        return [out_feat]


