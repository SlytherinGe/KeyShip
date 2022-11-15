
import torch
from mmdet.models.losses.utils import weight_reduce_loss
from mmdet.models.losses import SmoothL1Loss
from mmrotate.core.bbox.iou_calculators import build_iou_calculator
from ..builder import ROTATED_LOSSES
from mmrotate.models.utils import polar_decode
from mmrotate.core.bbox.transforms import poly2obb
from mmcv.ops import min_area_polygons

@ROTATED_LOSSES.register_module()
class IOUWeightedSmoothL1Loss(SmoothL1Loss):

    def __init__(self, beta=1, 
                       gamma=1, 
                       iou_calculator=dict(type='RBboxOverlaps2D'),
                       reduction='mean', 
                       loss_weight=1):
        super().__init__(beta, reduction, 1)
        self.gamma = gamma
        self.iou_func = build_iou_calculator(iou_calculator) if iou_calculator is not None else None
        self.eps = torch.finfo(torch.float32).eps
        self.loss_weight_iou = loss_weight

    def forward(self, 
                pred, 
                target, 
                weight=None, 
                avg_factor=None, 
                reduction_override=None, 
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        gt_mask_inds = weight.squeeze(1).flatten().gt(0)
        valid_gt_encoding = target.permute(0, 2, 3, 1).contiguous()
        valid_gt_encoding = valid_gt_encoding.view(-1, valid_gt_encoding.size(-1))[gt_mask_inds,:]
        valid_encoding_pred = pred.permute(0, 2, 3, 1).contiguous()
        valid_encoding_pred = valid_encoding_pred.view(-1, valid_encoding_pred.size(-1))[gt_mask_inds,:]
        loss_res = super().forward( valid_encoding_pred,
                                    valid_gt_encoding,
                                    weight=None,
                                    avg_factor=None,
                                    reduction_override='none')
        
        with torch.no_grad():
            valid_gt_poly = polar_decode(valid_encoding_pred[None], None, None, 8)
            valid_gt_poly = min_area_polygons(valid_gt_poly[0])
            valid_gt_rbox = poly2obb(valid_gt_poly)
            valid_poly_pred = polar_decode(valid_encoding_pred[None], None, None, 8)
            valid_poly_pred = min_area_polygons(valid_poly_pred[0])
            valid_rbox_pred = poly2obb(valid_poly_pred)
            ious = self.iou_func(valid_rbox_pred, valid_gt_rbox).diag()
            iou_weights = (1 - self.gamma*ious.log()/(loss_res.sum(-1).abs() + self.eps))

        return self.loss_weight_iou * weight_reduce_loss(loss_res.sum(-1), iou_weights, reduction, avg_factor)

    