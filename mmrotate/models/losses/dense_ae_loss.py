import mmcv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.utils import transpose_and_gather_feat

from ..builder import ROTATED_LOSSES

@mmcv.jit(derivate=True, coderize=True)
def dense_ae_loss_per_image(feat_pred, insta_inds):
    
    
    num_insta = len(insta_inds)
    if num_insta == 0:
        push_loss = feat_pred.sum() * 0.
        pull_loss = feat_pred.sum() * 0.     
        return pull_loss, push_loss
    me_list = [] 
    pull_loss = 0.
    push_loss = 0.
    for inds in insta_inds:
        feats = transpose_and_gather_feat(feat_pred[None], inds[None]).squeeze(0)
        embed_center = feats.mean(dim=0)
        instance_pull = (feats - embed_center).norm(dim=-1).pow(2).mean()
        pull_loss += instance_pull
        me_list.append(embed_center)
        
    me_list = torch.stack(me_list)
    # N is object number in image, M is dimension of embedding vector
    N, M = me_list.size()    
    margin = 2  # exp setting of CornerNet, details in section 3.3 of paper

    # confusion matrix of push loss
    conf_mat = me_list.expand((N, N, M)).permute(1, 0, 2) - me_list
    conf_weight = 1 - torch.eye(N).type_as(me_list)
    conf_mat = conf_weight * (margin - conf_mat.sum(-1).abs())    
    
    if N > 1:  # more than one object in current image
        push_loss = F.relu(conf_mat).sum() / (N * (N - 1))
    else:
        push_loss = feat_pred.sum() * 0.
    
    return pull_loss, push_loss

@mmcv.jit(derivate=True, coderize=True)
def dense_ae_loss_per_image_2(feat_pred, insta_inds):
    
    """This function not only push the center of the instances away, but the pixels within targets

    Returns:
        _type_: _description_
    """
    
    num_insta = len(insta_inds)
    if num_insta == 0:
        push_loss = feat_pred.sum() * 0.
        pull_loss = feat_pred.sum() * 0.     
        return pull_loss, push_loss
    me_list = [] 
    pull_loss = 0.
    push_loss = 0.
    for inds in insta_inds:
        feats = transpose_and_gather_feat(feat_pred[None], inds[None]).squeeze(0)
        embed_center = feats.mean(dim=0)
        instance_pull = (feats - embed_center).norm(dim=-1).pow(2).mean()
        pull_loss += instance_pull
        me_list.append(embed_center)
        
    me_list = torch.stack(me_list)
    # N is object number in image, M is dimension of embedding vector
    N, M = me_list.size()    
    margin = 2  # exp setting of CornerNet, details in section 3.3 of paper

    # confusion matrix of push loss
    conf_mat = me_list.expand((N, N, M)).permute(1, 0, 2) - me_list
    conf_weight = 1 - torch.eye(N).type_as(me_list)
    conf_mat = conf_weight * (margin - conf_mat.sum(-1).abs())    
    
    if N > 1:  # more than one object in current image
        push_loss = F.relu(conf_mat).sum() / (N * (N - 1))
    else:
        push_loss = feat_pred.sum() * 0.
    
    return pull_loss, push_loss

@ROTATED_LOSSES.register_module()
class DenseAssociativeEmbeddingLoss(nn.Module):
    
    def __init__(self, pull_weight=0.25, push_weight=0.25):
        super(DenseAssociativeEmbeddingLoss, self).__init__()
        self.pull_weight=pull_weight
        self.push_weight=push_weight
        
    def forward(self, pred, inds):
        """forward function for dense ae loss

        Args:
            pred (tensor): tensor of [batch, embedding_channel, feat_h, feat_w]
            inds (list): [[instance_inds],] the outter list contains the 
            data of different batches, the inner list contains the instance for
            the current batch.
        """
        batch = pred.size(0)
        pull_all, push_all = 0.0, 0.0
        for i in range(batch):
            pull, push = dense_ae_loss_per_image(pred[i], inds[i])

            pull_all += self.pull_weight * pull
            push_all += self.push_weight * push

        return pull_all, push_all