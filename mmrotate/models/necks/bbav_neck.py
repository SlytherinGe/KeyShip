from turtle import forward
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.runner import BaseModule, auto_fp16

from ..builder import ROTATED_NECKS

class CombinationModule(nn.Module):
    def __init__(self, c_low, c_up, batch_norm=False, group_norm=False, instance_norm=False):
        super(CombinationModule, self).__init__()
        if batch_norm:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.BatchNorm2d(c_up),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.BatchNorm2d(c_up),
                                           nn.ReLU(inplace=True))
        elif group_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.GroupNorm(num_groups=32, num_channels=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.GroupNorm(num_groups=32, num_channels=c_up),
                                          nn.ReLU(inplace=True))
        elif instance_norm:
            self.up = nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                    nn.InstanceNorm2d(num_features=c_up),
                                    nn.ReLU(inplace=True))
            self.cat_conv = nn.Sequential(nn.Conv2d(c_up * 2, c_up, kernel_size=1, stride=1),
                                          nn.InstanceNorm2d(num_features=c_up),
                                          nn.ReLU(inplace=True))
        else:
            self.up =  nn.Sequential(nn.Conv2d(c_low, c_up, kernel_size=3, padding=1, stride=1),
                                     nn.ReLU(inplace=True))
            self.cat_conv =  nn.Sequential(nn.Conv2d(c_up*2, c_up, kernel_size=1, stride=1),
                                           nn.ReLU(inplace=True))

    def forward(self, x_low, x_up):
        x_low = self.up(F.interpolate(x_low, x_up.shape[2:], mode='bilinear', align_corners=False))
        return self.cat_conv(torch.cat((x_up, x_low), 1))

@ROTATED_NECKS.register_module()
class BBAVNeck(BaseModule):

    def __init__(self, 
                in_channels,
                out_channels,               
                init_cfg = dict(
                type='Xavier', layer='Conv2d', distribution='uniform')):
        super().__init__(init_cfg)

        assert isinstance(in_channels, list)

        self.num_ins = len(in_channels)
        assert self.num_ins > 1

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.dec_convs = nn.ModuleList()

        for i in range(self.num_ins-1):
            conv_module = CombinationModule(self.in_channels[-1-i],
                                            self.in_channels[-2-i])
            self.dec_convs.append(conv_module)

    def forward(self, inputs):

        num_imputs = len(inputs)
        assert num_imputs == self.num_ins

        res = self.dec_convs[0](inputs[-1], inputs[-2])

        for i in range(num_imputs - 2):
            res = self.dec_convs[i+1](res, inputs[-3-i])

        return res


