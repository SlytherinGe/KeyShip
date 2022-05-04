# Copyright (c) OpenMMLab. All rights reserved.
import torch

from ..builder import ROTATED_BBOX_ASSIGNERS
from mmdet.core.bbox.assigners.assign_result import AssignResult
from mmdet.core.bbox.assigners.base_assigner import BaseAssigner


@ROTATED_BBOX_ASSIGNERS.register_module()
class Point2MaskAssigner(BaseAssigner):
    """Assign a corresponding gt rbbox or background to each point.

    Each proposals will be assigned with `0`, or a positive integer
    indicating the ground truth index.

    - 0: negative sample, no assigned gt
    - positive integer: positive sample, index (1-based) of assigned gt
    """

    def __init__(self, **args):
        pass

    def assign(self,
               cls_pred,
               ctx_pred,
               mask_pred,
               gt_labels,
               gt_kpts,
               gt_mask,
               img_meta,
               gt_bboxes_ignore=None,
               eps=1e-7):
        """Assign gt to points.

        This method assign a gt bbox to every points set, each points set
        will be assigned with  the background_label (-1), or a label number.
        -1 is background, and semi-positive number is the index (0-based) of
        assigned gt.
        The assignment is done in following steps, the order matters.

        1. assign every points to the background_label (-1)
        2. A point is assigned to some gt bbox if
            (i) the point is in the gt rbox mask

        Args:
            ctx_pred (Tensor): the center points detected, shape(n, 2) while last
                dimension stands for normalized (x, y).
            gt_mask (Tensor): Groundtruth rboxe masks, shape (k, pad_h, pad_w).
            gt_bboxes_ignore (Tensor, optional): Ground truth bboxes that are
                labelled as `ignored`, e.g., crowd boxes in COCO.
                NOTE: currently unused.
            gt_labels (Tensor, optional): Label of gt_bboxes, shape (k, ).

        Returns:
            :obj:`AssignResult`: The assign result.
        """
        num_points = ctx_pred.shape[0]
        num_gts, mask_h, mask_w = gt_mask.shape

        if num_gts == 0 or num_points == 0:
            # If no truth assign everything to the background
            assigned_gt_inds = ctx_pred.new_full((num_points, ),
                                               0,
                                               dtype=torch.long)
            if gt_labels is None:
                assigned_labels = None
            else:
                assigned_labels = ctx_pred.new_full((num_points, ),
                                                  -1,
                                                  dtype=torch.long)
            return AssignResult(
                num_gts, assigned_gt_inds, None, labels=assigned_labels)

        ctx_scaler = ctx_pred.new_tensor([mask_w, mask_h])
        points_xy = (ctx_pred * ctx_scaler).long()
        # stores the assigned gt index of each point
        assigned_gt_inds = ctx_scaler.new_zeros((num_points, ), dtype=torch.long)
        assigned_gt_one_hot = gt_mask[:, points_xy[:,1], points_xy[:,0]]
        val, ind = assigned_gt_one_hot.max(dim=-1)
        none_bg_ind = val > 0
        assigned_gt_inds[none_bg_ind] = ind[none_bg_ind] + 1

        if gt_labels is not None:
            assigned_labels = assigned_gt_inds.new_full((num_points, ), -1)
            pos_inds = torch.nonzero(
                assigned_gt_inds > 0, as_tuple=False).squeeze()
            if pos_inds.numel() > 0:
                assigned_labels[pos_inds] = gt_labels[
                    assigned_gt_inds[pos_inds] - 1]
        else:
            assigned_labels = None

        return AssignResult(
            num_gts, assigned_gt_inds, None, labels=assigned_labels)

