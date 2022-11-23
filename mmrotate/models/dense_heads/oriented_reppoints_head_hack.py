from mmrotate.models.dense_heads import OrientedRepPointsHead
from cv2 import minAreaRect
import torch
from mmcv.ops import chamfer_distance
from mmrotate.core import obb2poly, poly2obb, multiclass_nms_rotated
import numpy as np
from ..builder import ROTATED_HEADS


def ChamferDistance2D(point_set_1,
                      point_set_2,
                      distance_weight=0.05,
                      eps=1e-12):
    """Compute the Chamfer distance between two point sets.

    Args:
        point_set_1 (torch.tensor): point set 1 with shape (N_pointsets,
                                    N_points, 2)
        point_set_2 (torch.tensor): point set 2 with shape (N_pointsets,
                                    N_points, 2)

    Returns:
        dist (torch.tensor): chamfer distance between two point sets
                             with shape (N_pointsets,)
    """
    assert point_set_1.dim() == point_set_2.dim()
    assert point_set_1.shape[-1] == point_set_2.shape[-1]
    assert point_set_1.dim() <= 3
    dist1, dist2, _, _ = chamfer_distance(point_set_1, point_set_2)
    dist1 = torch.sqrt(torch.clamp(dist1, eps))
    dist2 = torch.sqrt(torch.clamp(dist2, eps))
    dist = distance_weight * (dist1.mean(-1) + dist2.mean(-1)) / 2.0

    return dist

def min_area_polygons(pointsets: torch.Tensor) -> torch.Tensor:
    """Find the smallest polygons that surrounds all points in the point sets.

    Args:
        pointsets (Tensor): point sets with shape  (N, 18).

    Returns:
        torch.Tensor: Return the smallest polygons with shape (N, 8).
    """
    rects = []
    n_sets, n_pts = pointsets.size()[:2]
    n_pts = n_pts // 2
    pointsets_np = pointsets.view(n_sets, -1, 2).cpu().numpy()
    for pointset in pointsets_np:
        rect = minAreaRect(pointset)
        rect_ = [rect[0][0], rect[0][1], rect[1][0], rect[1][1], rect[2] /180 * np.pi]
        rects.append(rect_)
    rects = pointsets.new_tensor(rects)
    polygons = obb2poly(rects, 'oc')
    
    return polygons

@ROTATED_HEADS.register_module()
class OrientedRepPointsHeadHack(OrientedRepPointsHead):
    
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)
        
    def pointsets_quality_assessment(self, pts_features, cls_score,
                                     pts_pred_init, pts_pred_refine, label,
                                     bbox_gt, label_weight, bbox_weight,
                                     pos_inds):
        """Assess the quality of each point set from the classification,
        localization, orientation, and point-wise correlation based on
        the assigned point sets samples.
        Args:
            pts_features (torch.tensor): points features with shape (N, 9, C)
            cls_score (torch.tensor): classification scores with
                        shape (N, class_num)
            pts_pred_init (torch.tensor): initial point sets prediction with
                        shape (N, 9*2)
            pts_pred_refine (torch.tensor): refined point sets prediction with
                        shape (N, 9*2)
            label (torch.tensor): gt label with shape (N)
            bbox_gt(torch.tensor): gt bbox of polygon with shape (N, 8)
            label_weight (torch.tensor): label weight with shape (N)
            bbox_weight (torch.tensor): box weight with shape (N)
            pos_inds (torch.tensor): the  inds of  positive point set samples

        Returns:
            qua (torch.tensor) : weighted quality values for positive
                                 point set samples.
        """
        if len(pos_inds) == 0:
            print('!!!!!!!!!')
            return cls_score.new_zeros(100),
        device = cls_score.device
        pos_scores = cls_score[pos_inds]
        pos_pts_pred_init = pts_pred_init[pos_inds]
        pos_pts_pred_refine = pts_pred_refine[pos_inds]
        pos_pts_refine_features = pts_features[pos_inds]
        pos_bbox_gt = bbox_gt[pos_inds]
        pos_label = label[pos_inds]
        pos_label_weight = label_weight[pos_inds]
        pos_bbox_weight = bbox_weight[pos_inds]

        # quality of point-wise correlation
        qua_poc = self.poc_qua_weight * self.feature_cosine_similarity(
            pos_pts_refine_features)

        qua_cls = self.loss_cls(
            pos_scores,
            pos_label,
            pos_label_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        polygons_pred_init = min_area_polygons(pos_pts_pred_init)
        polygons_pred_refine = min_area_polygons(pos_pts_pred_refine)
        sampling_pts_pred_init = self.sampling_points(
            polygons_pred_init, 10, device=device)
        sampling_pts_pred_refine = self.sampling_points(
            polygons_pred_refine, 10, device=device)
        sampling_pts_gt = self.sampling_points(pos_bbox_gt, 10, device=device)

        # quality of orientation
        qua_ori_init = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_init)
        qua_ori_refine = self.ori_qua_weight * ChamferDistance2D(
            sampling_pts_gt, sampling_pts_pred_refine)

        # quality of localization
        qua_loc_init = self.loss_bbox_refine(
            pos_pts_pred_init,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')
        qua_loc_refine = self.loss_bbox_refine(
            pos_pts_pred_refine,
            pos_bbox_gt,
            pos_bbox_weight,
            avg_factor=self.loss_cls.loss_weight,
            reduction_override='none')

        # quality of classification
        qua_cls = qua_cls.sum(-1)

        # weighted inti-stage and refine-stage
        qua = qua_cls + self.init_qua_weight * (
            qua_loc_init + qua_ori_init) + (1.0 - self.init_qua_weight) * (
                qua_loc_refine + qua_ori_refine) + qua_poc

        return qua,    

    def _get_bboxes_single(self,
                           cls_score_list,
                           point_pred_list,
                           mlvl_priors,
                           img_meta,
                           cfg,
                           rescale=False,
                           with_nms=True,
                           **kwargs):
        """Transform outputs of a single image into bbox predictions.
        Args:
            cls_score_list (list[Tensor]): Box scores from all scale
                levels of a single image, each item has shape
                (num_priors * num_classes, H, W).
            bbox_pred_list (list[Tensor]): Box energies / deltas from
                all scale levels of a single image, each item has shape
                (num_priors * 4, H, W).
            score_factor_list (list[Tensor]): Score factor from all scale
                levels of a single image. RepPoints head does not need
                this value.
            mlvl_priors (list[Tensor]): Each element in the list is
                the priors of a single level in feature pyramid, has shape
                (num_priors, 2).
            img_meta (dict): Image meta info.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.
        Returns:
            tuple[Tensor]: Results of detected bboxes and labels. If with_nms
                is False and mlvl_score_factor is None, return mlvl_bboxes and
                mlvl_scores, else return mlvl_bboxes, mlvl_scores and
                mlvl_score_factor. Usually with_nms is False is used for aug
                test. If with_nms is True, then return the following format

                - det_bboxes (Tensor): Predicted bboxes with shape \
                    [num_bboxes, 5], where the first 4 columns are bounding \
                    box positions (cx, cy, w, h, a) and the 5-th \
                    column are scores between 0 and 1.
                - det_labels (Tensor): Predicted labels of the corresponding \
                    box with shape [num_bboxes].
        """

        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_score_list) == len(point_pred_list)
        scale_factor = img_meta['scale_factor']

        mlvl_bboxes = []
        mlvl_scores = []
        for level_idx, (cls_score, points_pred, points) in enumerate(
                zip(cls_score_list, point_pred_list, mlvl_priors)):
            assert cls_score.size()[-2:] == points_pred.size()[-2:]

            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            points_pred = points_pred.permute(1, 2, 0).reshape(
                -1, 2 * self.num_points)
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < scores.shape[0]:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                points_pred = points_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            pts_pred = points_pred.reshape(-1, self.num_points, 2)
            pts_pred_offsety = pts_pred[:, :, 0::2]
            pts_pred_offsetx = pts_pred[:, :, 1::2]
            pts_pred = torch.cat([pts_pred_offsetx, pts_pred_offsety],
                                 dim=2).reshape(-1, 2 * self.num_points)

            pts_pos_center = points[:, :2].repeat(1, self.num_points)
            pts = pts_pred * self.point_strides[level_idx] + pts_pos_center

            polys = min_area_polygons(pts)
            bboxes = poly2obb(polys, self.version)

            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes[..., :4].new_tensor(
                scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms_rotated(
                mlvl_bboxes, mlvl_scores, cfg.score_thr, cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            raise NotImplementedError