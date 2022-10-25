import torch
from mmdet.core import images_to_levels, multi_apply, unmap
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.runner import force_fp32

from ..builder import ROTATED_HEADS
from .rotated_retina_head import RotatedRetinaHead

from mmrotate.core import (obb2hbb, rotated_anchor_inside_flags, hbb2obb)


@ROTATED_HEADS.register_module()
class SoftRetinaHead(RotatedRetinaHead):
    r"""An anchor-based head used in `RotatedRetinaNet
    <https://arxiv.org/pdf/1708.02002.pdf>`_.

    The head contains two subnetworks. The first classifies anchor boxes and
    the second regresses deltas for the anchors.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int, optional): Number of stacked convolutions.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Config dict for normalization layer.
            Default: None.
        anchor_generator (dict): Config dict for anchor generator
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """  # noqa: W605
    def __init__(self, num_classes,
                       in_channels, 
                       stacked_convs=4, 
                       conv_cfg=None, 
                       norm_cfg=None, 
                       anchor_generator=dict(
                            type='AnchorGenerator',
                            octave_base_scale=4,
                            scales_per_octave=3,
                            ratios=[0.5, 1.0, 2.0],
                            strides=[8, 16, 32, 64, 128]),
                       init_cfg=dict(
                            type='Normal',
                            layer='Conv2d',
                            std=0.01,
                            override=dict(
                                type='Normal',
                                name='retina_cls',
                                std=0.01,
                                bias_prob=0.01)),
                       **kwargs):
        super(SoftRetinaHead, self).__init__(num_classes, 
                         in_channels, 
                         stacked_convs, 
                         conv_cfg, 
                         norm_cfg, 
                         anchor_generator, 
                         init_cfg, 
                         **kwargs)
        self.angle_version = self.bbox_coder.angle_range

    # def get_anchors(self, featmap_sizes, img_metas, device='cuda'):
    #     """Get anchors according to feature map sizes.

    #     Args:
    #         featmap_sizes (list[tuple]): Multi-level feature map sizes.
    #         img_metas (list[dict]): Image meta info.
    #         device (torch.device | str): Device for returned tensors

    #     Returns:
    #         tuple (list[Tensor]):

    #             - anchor_list (list[Tensor]): Anchors of each image.
    #             - valid_flag_list (list[Tensor]): Valid flags of each image.
    #     """
    #     num_imgs = len(img_metas)

    #     # since feature map sizes of all images are the same, we only compute
    #     # anchors for one time
    #     multi_level_anchors = self.anchor_generator.grid_priors(
    #         featmap_sizes, device)
    #     for i in range(len(multi_level_anchors)):
    #         multi_level_anchors[i] = hbb2obb(multi_level_anchors[i][...,:-1], self.angle_version)
    #     anchor_list = [multi_level_anchors for _ in range(num_imgs)]

    #     # for each image, we compute valid flags of multi level anchors
    #     valid_flag_list = []
    #     for img_id, img_meta in enumerate(img_metas):
    #         multi_level_flags = self.anchor_generator.valid_flags(
    #             featmap_sizes, img_meta['pad_shape'], device)
    #         valid_flag_list.append(multi_level_flags)

    #     return anchor_list, valid_flag_list


# @ROTATED_HEADS.register_module()
# class SoftRetinaHead(RotatedRetinaHead):

#     def __init__(self, num_classes,
#                        in_channels, 
#                        stacked_convs=4, 
#                        conv_cfg=None, 
#                        norm_cfg=None, 
#                        anchor_generator=dict(
#                             type='AnchorGenerator',
#                             octave_base_scale=4,
#                             scales_per_octave=3,
#                             ratios=[0.5, 1.0, 2.0],
#                             strides=[8, 16, 32, 64, 128]),
#                        init_cfg=dict(
#                             type='Normal',
#                             layer='Conv2d',
#                             std=0.01,
#                             override=dict(
#                                 type='Normal',
#                                 name='retina_cls',
#                                 std=0.01,
#                                 bias_prob=0.01)),
#                        **kwargs):
#         super(SoftRetinaHead, self).__init__(num_classes, 
#                          in_channels, 
#                          stacked_convs, 
#                          conv_cfg, 
#                          norm_cfg, 
#                          anchor_generator, 
#                          init_cfg, 
#                          **kwargs)

#     @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
#     def loss(self,
#              cls_scores,
#              bbox_preds,
#              gt_bboxes,
#              gt_labels,
#              img_metas,
#              gt_bboxes_ignore=None):
#         """Compute losses of the head.

#         Args:
#             cls_scores (list[Tensor]): Box scores for each scale level
#                 Has shape (N, num_anchors * num_classes, H, W)
#             bbox_preds (list[Tensor]): Box energies / deltas for each scale
#                 level with shape (N, num_anchors * 5, H, W)
#             gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
#                 shape (num_gts, 5) in [cx, cy, w, h, a] format.
#             gt_labels (list[Tensor]): class indices corresponding to each box
#             img_metas (list[dict]): Meta information of each image, e.g.,
#                 image size, scaling factor, etc.
#             gt_bboxes_ignore (None | list[Tensor]): specify which bounding
#                 boxes can be ignored when computing the loss. Default: None

#         Returns:
#             dict[str, Tensor]: A dictionary of loss components.
#         """
#         featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
#         assert len(featmap_sizes) == self.anchor_generator.num_levels

#         device = cls_scores[0].device

#         anchor_list, valid_flag_list = self.get_anchors(
#             featmap_sizes, img_metas, device=device)
#         label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
#         cls_reg_targets = self.get_targets(
#             anchor_list,
#             valid_flag_list,
#             bbox_preds,
#             gt_bboxes,
#             img_metas,
#             gt_bboxes_ignore_list=gt_bboxes_ignore,
#             gt_labels_list=gt_labels,
#             label_channels=label_channels)
#         if cls_reg_targets is None:
#             return None
#         (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
#          num_total_pos, num_total_neg) = cls_reg_targets
#         num_total_samples = (
#             num_total_pos + num_total_neg if self.sampling else num_total_pos)

#         # anchor number of multi levels
#         num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
#         # concat all level anchors and flags to a single tensor
#         concat_anchor_list = []
#         for i, _ in enumerate(anchor_list):
#             concat_anchor_list.append(torch.cat(anchor_list[i]))
#         all_anchor_list = images_to_levels(concat_anchor_list,
#                                            num_level_anchors)

#         losses_cls, losses_bbox = multi_apply(
#             self.loss_single,
#             cls_scores,
#             bbox_preds,
#             all_anchor_list,
#             labels_list,
#             label_weights_list,
#             bbox_targets_list,
#             bbox_weights_list,
#             num_total_samples=num_total_samples)
#         return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)

#     def get_targets(self, 
#                     anchor_list, 
#                     valid_flag_list, 
#                     bbox_pred_list,
#                     gt_bboxes_list, 
#                     img_metas, 
#                     gt_bboxes_ignore_list=None, 
#                     gt_labels_list=None, 
#                     label_channels=1, 
#                     unmap_outputs=True, 
#                     return_sampling_results=False):
#         """Compute regression and classification targets for anchors in
#         multiple images.

#         Args:
#             anchor_list (list[list[Tensor]]): Multi level anchors of each
#                 image. The outer list indicates images, and the inner list
#                 corresponds to feature levels of the image. Each element of
#                 the inner list is a tensor of shape (num_anchors, 5).
#             valid_flag_list (list[list[Tensor]]): Multi level valid flags of
#                 each image. The outer list indicates images, and the inner list
#                 corresponds to feature levels of the image. Each element of
#                 the inner list is a tensor of shape (num_anchors, )
#             gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
#             img_metas (list[dict]): Meta info of each image.
#             gt_bboxes_ignore_list (list[Tensor]): Ground truth bboxes to be
#                 ignored.
#             gt_labels_list (list[Tensor]): Ground truth labels of each box.
#             label_channels (int): Channel of label.
#             unmap_outputs (bool): Whether to map outputs back to the original
#                 set of anchors.

#         Returns:
#             tuple: Usually returns a tuple containing learning targets.

#                 - labels_list (list[Tensor]): Labels of each level.
#                 - label_weights_list (list[Tensor]): Label weights of each \
#                     level.
#                 - bbox_targets_list (list[Tensor]): BBox targets of each level.
#                 - bbox_weights_list (list[Tensor]): BBox weights of each level.
#                 - num_total_pos (int): Number of positive samples in all \
#                     images.
#                 - num_total_neg (int): Number of negative samples in all \
#                     images.

#             additional_returns: This function enables user-defined returns from
#                 `self._get_targets_single`. These returns are currently refined
#                 to properties at each feature map (i.e. having HxW dimension).
#                 The results will be concatenated after the end
#         """
#         num_imgs = len(img_metas)
#         assert len(anchor_list) == len(valid_flag_list) == num_imgs

#         # anchor number of multi levels
#         num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
#         # concat all level anchors to a single tensor
#         concat_anchor_list = []
#         concat_valid_flag_list = []
#         for i in range(num_imgs):
#             assert len(anchor_list[i]) == len(valid_flag_list[i])
#             concat_anchor_list.append(torch.cat(anchor_list[i]))
#             concat_valid_flag_list.append(torch.cat(valid_flag_list[i]))
#         # reshape the bbox_pred_list into the shape of cancat_anchor_list
#         # list(FPN(tensor[batch_size, num_anchor×5, h, w])) -> list(IMG_INS(tensor[total_anchors, 5]))
#         concat_bbox_pred_list = []
#         for i in range(num_imgs):
#             bbox_pred_temp = []
#             for bbox_pred in bbox_pred_list:
#                 pred = bbox_pred[i].permute(1, 2, 0).reshape(-1, 5)
#                 bbox_pred_temp.append(pred)
#             concat_bbox_pred_list.append(torch.cat(bbox_pred_temp))

#         # compute targets for each image
#         if gt_bboxes_ignore_list is None:
#             gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
#         if gt_labels_list is None:
#             gt_labels_list = [None for _ in range(num_imgs)]
#         results = multi_apply(
#             self._get_targets_single,
#             concat_anchor_list,
#             concat_valid_flag_list,
#             concat_bbox_pred_list,
#             gt_bboxes_list,
#             gt_bboxes_ignore_list,
#             gt_labels_list,
#             img_metas,
#             label_channels=label_channels,
#             unmap_outputs=unmap_outputs)
#         (all_labels, all_label_weights, all_bbox_targets, all_bbox_weights,
#          pos_inds_list, neg_inds_list, sampling_results_list) = results[:7]
#         rest_results = list(results[7:])  # user-added return values
#         # no valid anchors
#         if any([labels is None for labels in all_labels]):
#             return None
#         # sampled anchors of all images
#         num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
#         num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
#         # split targets to a list w.r.t. multiple levels
#         labels_list = images_to_levels(all_labels, num_level_anchors)
#         label_weights_list = images_to_levels(all_label_weights,
#                                               num_level_anchors)
#         bbox_targets_list = images_to_levels(all_bbox_targets,
#                                              num_level_anchors)
#         bbox_weights_list = images_to_levels(all_bbox_weights,
#                                              num_level_anchors)
#         res = (labels_list, label_weights_list, bbox_targets_list,
#                bbox_weights_list, num_total_pos, num_total_neg)
#         if return_sampling_results:
#             res = res + (sampling_results_list, )
#         for i, r in enumerate(rest_results):  # user-added return values
#             rest_results[i] = images_to_levels(r, num_level_anchors)

#         return res + tuple(rest_results)

#     def _get_targets_single(self,
#                             flat_anchors,
#                             valid_flags,
#                             flat_preds,
#                             gt_bboxes,
#                             gt_bboxes_ignore,
#                             gt_labels,
#                             img_meta,
#                             label_channels=1,
#                             unmap_outputs=True):
#         """Compute regression and classification targets for anchors in a
#         single image.

#         Args:
#             flat_anchors (torch.Tensor): Multi-level anchors of the image,
#                 which are concatenated into a single tensor of shape
#                 (num_anchors, 5)
#             valid_flags (torch.Tensor): Multi level valid flags of the image,
#                 which are concatenated into a single tensor of
#                     shape (num_anchors,).
#             gt_bboxes (torch.Tensor): Ground truth bboxes of the image,
#                 shape (num_gts, 5).
#             img_meta (dict): Meta info of the image.
#             gt_bboxes_ignore (torch.Tensor): Ground truth bboxes to be
#                 ignored, shape (num_ignored_gts, 5).
#             img_meta (dict): Meta info of the image.
#             gt_labels (torch.Tensor): Ground truth labels of each box,
#                 shape (num_gts,).
#             label_channels (int): Channel of label.
#             unmap_outputs (bool): Whether to map outputs back to the original
#                 set of anchors.

#         Returns:
#             tuple (list[Tensor]):

#                 - labels_list (list[Tensor]): Labels of each level
#                 - label_weights_list (list[Tensor]): Label weights of each \
#                   level
#                 - bbox_targets_list (list[Tensor]): BBox targets of each level
#                 - bbox_weights_list (list[Tensor]): BBox weights of each level
#                 - num_total_pos (int): Number of positive samples in all images
#                 - num_total_neg (int): Number of negative samples in all images
#         """
#         inside_flags = rotated_anchor_inside_flags(
#             flat_anchors, valid_flags, img_meta['img_shape'][:2],
#             self.train_cfg.allowed_border)
#         if not inside_flags.any():
#             return (None, ) * 7
#         # assign gt and sample anchors
#         anchors = flat_anchors[inside_flags, :]
#         preds = flat_preds[inside_flags, :]

#         if self.assign_by_circumhbbox is not None:
#             gt_bboxes_assign = obb2hbb(gt_bboxes, self.assign_by_circumhbbox)
#             assign_result = self.assigner.assign(
#                 anchors, gt_bboxes_assign, gt_bboxes_ignore,
#                 None if self.sampling else gt_labels)
#         else:
#             assign_result = self.assigner.assign(
#                 anchors, gt_bboxes, gt_bboxes_ignore,
#                 None if self.sampling else gt_labels)

#         sampling_result = self.sampler.sample(assign_result, anchors,
#                                               gt_bboxes)

#         num_valid_anchors = anchors.shape[0]
#         bbox_targets = torch.zeros_like(anchors)
#         bbox_weights = torch.zeros_like(anchors)
#         labels = anchors.new_full((num_valid_anchors, ),
#                                   self.num_classes,
#                                   dtype=torch.long)
#         label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

#         pos_inds = sampling_result.pos_inds
#         neg_inds = sampling_result.neg_inds
#         if len(pos_inds) > 0:
#             if not self.reg_decoded_bbox:
#                 pos_bbox_targets = self.bbox_coder.encode(
#                     sampling_result.pos_bboxes, sampling_result.pos_gt_bboxes)
#             else:
#                 pos_bbox_targets = sampling_result.pos_gt_bboxes
#             bbox_targets[pos_inds, :] = pos_bbox_targets
#             bbox_weights[pos_inds, :] = 1.0
#             if gt_labels is None:
#                 # Only rpn gives gt_labels as None
#                 # Foreground is the first class since v2.5.0
#                 labels[pos_inds] = 0
#             else:
#                 labels[pos_inds] = gt_labels[
#                     sampling_result.pos_assigned_gt_inds]
#             if self.train_cfg.pos_weight <= 0:
#                 label_weights[pos_inds] = 1.0
#             else:
#                 label_weights[pos_inds] = self.train_cfg.pos_weight
#         if len(neg_inds) > 0:
#             label_weights[neg_inds] = 1.0

#         # map up to original set of anchors
#         if unmap_outputs:
#             num_total_anchors = flat_anchors.size(0)
#             labels = unmap(
#                 labels, num_total_anchors, inside_flags,
#                 fill=self.num_classes)  # fill bg label
#             label_weights = unmap(label_weights, num_total_anchors,
#                                   inside_flags)
#             bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
#             bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

#         return (labels, label_weights, bbox_targets, bbox_weights, pos_inds,
#                 neg_inds, sampling_result)