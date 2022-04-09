'''
Author: SlytherinGe
LastEditTime: 2022-02-21 10:41:18
'''
import numpy as np
from sympy import false
import torch
import math
import cv2
SMALL_NUM = 1e-6

def gaussian2D_R(w, h, a, sigma_ratio, dtype=torch.float32, device='cpu'):
    # sigma_ratio: tuple:(ratio_w, ratio_h)
    cos_theta = math.cos(a)
    sin_theta = math.sin(a)
    # calculate the min bounding box containing the rotated box
    min_width, min_height = w * cos_theta + abs(h * sin_theta), h * cos_theta + abs(w * sin_theta)
    cx, cy = max(1, int(min_width / 2)), max(1, int(min_height / 2))

    X = torch.arange(-cx, cx+1, dtype=dtype, device=device)
    Y = torch.arange(-cy, cy+1, dtype=dtype, device=device)

    y, x = torch.meshgrid(Y, X)
    scale = torch.ones_like(y)
    pos_matrix = torch.cat((x[None], y[None], scale[None])).permute(1,2,0)

    rotate_matrix = torch.tensor([[cos_theta, -sin_theta, 0],
                                  [sin_theta, cos_theta,0],
                                  [0,0,1]],device=device)
    rotated_matrix = torch.matmul(pos_matrix, rotate_matrix)

    X, Y = rotated_matrix[...,0], rotated_matrix[...,1]
    
    theta_x = w * sigma_ratio[0]
    theta_y = h * sigma_ratio[1]

    G = X * X / (2 * theta_x * theta_x) +\
        Y * Y / (2 * theta_y * theta_y)

    G = (-G).exp()

    G[G < torch.finfo(G.dtype).eps * G.max()] = 0

    return G

def gen_gaussian_targetR(heatmap, cx, cy , w, h, a, sigma_ratio):
    '''
    Input a should be a radius rather than an angle
    '''
    gaussian_kernel = gaussian2D_R(w, h, a,
                                   sigma_ratio=sigma_ratio, 
                                   dtype=heatmap.dtype, 
                                   device=heatmap.device)
    k_h, k_w = gaussian_kernel.shape
    h_h, h_w = heatmap.shape[:2]
    half_k_h, half_k_w = (k_h - 1) // 2, (k_w - 1) // 2    

    cx, cy = int(cx), int(cy)  

    left, right = min(cx, half_k_w), min(h_w - cx, half_k_w + 1)
    top, bottom = min(cy, half_k_h), min(h_h - cy, half_k_h + 1) 

    masked_heatmap = heatmap[cy - top:cy + bottom, cx - left:cx + right]
    masked_gaussian = gaussian_kernel[half_k_h - top:half_k_h + bottom,
                                      half_k_w - left:half_k_w + right]

    out_heatmap = heatmap   
    try:
        torch.max(
            masked_heatmap,
            masked_gaussian,
            out=out_heatmap[cy - top:cy + bottom, cx - left:cx + right])
    except RuntimeError:
        print('cx:', cx, 'cy:', cy)
        print('l, r, t, b:', left, right, top, bottom)
        print('cx, half_k_w, h_w:',cx, half_k_w, h_w)
        print('cy, half_k_h, h_h',cy, half_k_h, h_h)
        print(masked_gaussian.shape)
        print(masked_heatmap.shape)
    
    return out_heatmap

def get_local_maximum(heat, kernel=3):
    """Extract local maximum pixel with given kernel.

    Args:
        heat (Tensor): Target heatmap.
        kernel (int): Kernel size of max pooling. Default: 3.

    Returns:
        heat (Tensor): A heatmap where local maximum pixels maintain its
            own value and other positions are 0.
    """
    pad = (kernel - 1) // 2
    hmax = torch.max_pool2d(heat, kernel, stride=1, padding=pad)
    keep = (hmax == heat).float()
    return heat * keep


def get_topk_from_heatmap(scores, k=20):
    """Get top k positions from heatmap.

    Args:
        scores (Tensor): Target heatmap with shape
            [batch, num_classes, height, width].
        k (int): Target number. Default: 20.

    Returns:
        tuple[torch.Tensor]: Scores, indexes, categories and coords of
            topk keypoint. Containing following Tensors:

        - topk_scores (Tensor): Max scores of each topk keypoint.
        - topk_inds (Tensor): Indexes of each topk keypoint.
        - topk_clses (Tensor): Categories of each topk keypoint.
        - topk_ys (Tensor): Y-coord of each topk keypoint.
        - topk_xs (Tensor): X-coord of each topk keypoint.
    """
    batch, _, height, width = scores.size()
    topk_scores, topk_inds = torch.topk(scores.view(batch, -1), k)
    topk_clses = topk_inds // (height * width)
    topk_inds = topk_inds % (height * width)
    topk_ys = topk_inds // width
    topk_xs = (topk_inds % width).int().float()
    return topk_scores, topk_inds, topk_clses, topk_ys, topk_xs


def gather_feat(feat, ind, mask=None):
    """Gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.
        mask (Tensor | None): Mask of feature map. Default: None.

    Returns:
        feat (Tensor): Gathered feature.
    """
    dim = feat.size(2)
    ind = ind.unsqueeze(2).repeat(1, 1, dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def transpose_and_gather_feat(feat, ind):
    """Transpose and gather feature according to index.

    Args:
        feat (Tensor): Target feature map.
        ind (Tensor): Target coord index.

    Returns:
        feat (Tensor): Transposed and gathered feature.
    """
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = gather_feat(feat, ind)
    return feat

def keypoints2rbboxes(bboxes, using_geo_center=True, sc_first=False):

    # calculate a
    batch = bboxes.size(0)
    sc_offset = 4 * (1 - int(sc_first))
    dy = bboxes[...,sc_offset+3] - bboxes[...,sc_offset+1]
    dx = bboxes[...,sc_offset+2] - bboxes[...,sc_offset]
    a = torch.atan2(dy , dx)

    # calculate w, which is the length between shortside centers
    sc_vec = bboxes[...,sc_offset:sc_offset+2] - bboxes[...,sc_offset+2:sc_offset+4]
    w = torch.norm(sc_vec, dim=-1)

    # calculate h
    if sc_first:
        lc_pts = bboxes[...,4:].view(batch,-1,2,1,2).repeat(1,1,1,2,1)
        sc_pts = bboxes[...,:4].view(batch,-1,1,2,2).repeat(1,1,2,1,1)  
    else:      
        lc_pts = bboxes[...,:4].view(batch,-1,2,1,2).repeat(1,1,1,2,1)
        sc_pts = bboxes[...,4:].view(batch,-1,1,2,2).repeat(1,1,2,1,1)
    vec = sc_pts - lc_pts
    vec_3d = torch.zeros((vec.size(0), vec.size(1), vec.size(2), vec.size(3), 3),
                          device=sc_pts.device,
                          dtype=sc_pts.dtype)
    vec_3d[...,:2] = vec
    area = torch.cross(vec_3d[...,0,:], vec_3d[...,1,:], dim=-1)
    g_vec = sc_pts[...,0,:] - sc_pts[...,1,:]
    g_len = torch.norm(g_vec, dim=-1)
    dist = area[...,2].abs() / g_len
    h = dist.sum(dim=-1)

    # calculate x, y
    if using_geo_center:
        x = bboxes[...,0::2].sum(dim=-1) / 4
        y = bboxes[...,1::2].sum(dim=-1) / 4
    else:
        raise NotImplementedError
        
    rbboxes = torch.stack([x,y,w,h,a],dim=-1)

    return rbboxes

def sort_valid_gt_bboxes(gt_bboxes, reg_range):
    """
    Args:
        gt_bboxes (Tensor[N,5]): N gt bboxes with [x,y,w,h,a] information
            for an image.
        reg_range (Tuple(min, max))
    Returns:
        Tuple(valid_gt_bboxes, unvalid_gt_bboxes)
    """
    min_len, max_len = reg_range
    
    bbox_longside_len = gt_bboxes[:, 2:4].max(dim=1)[0]
    # bbox_shortside_len = gt_bboxes[:, 2:4].min(dim=1)[0]

    valid_index = (bbox_longside_len > min_len) & (bbox_longside_len <= max_len)
    # valid_index = (bbox_shortside_len > min_len) & (bbox_shortside_len <= max_len)

    return valid_index

def get_target_map(unvalid_gt_bboxes, map_shape, device):

    w, h = map_shape
    target_map = np.ones((h, w))
    # multiply the width and height of unvalid gt bboxes by 1.5
    unvalid_gt_bboxes = unvalid_gt_bboxes.clone()
    unvalid_gt_bboxes[:, 2:4] = unvalid_gt_bboxes[:, 2:4] * 1.5

    for box in unvalid_gt_bboxes:
        x, y, w, h, a = float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(box[4])
        a = a * 180 / np.pi
        pts = cv2.boxPoints(((x, y), (w, h), a))  
        pts = np.int0(pts)
        cv2.drawContours(target_map, [pts], -1, 0, -1)

    return torch.tensor(target_map, dtype=torch.int, device=device)    

def set_offset(offset_map, f_pos_x, f_pos_y):
    
    pos_x, pos_y = int(f_pos_x), int(f_pos_y)
    _, off_h ,off_w = offset_map.shape
    if pos_x < off_w and pos_y < off_h:
        offset_map[0, pos_y, pos_x] = f_pos_x - pos_x
        offset_map[1, pos_y, pos_x] = f_pos_y - pos_y 

    return offset_map

def set_centripetal_shifts(centripetal_map, f_ec_x, f_ec_y, f_tc_x, f_tc_y):

    pos_x, pos_y = int(f_ec_x), int(f_ec_y)
    _, off_h ,off_w = centripetal_map.shape
    if pos_x < off_w and pos_x < off_h:
        centripetal_map[0, pos_y, pos_x] = f_tc_x - f_ec_x
        centripetal_map[1, pos_y, pos_x] = f_tc_y - f_ec_y 

    return centripetal_map

# def set_dual_centripetal_shifts(centripetal_map, ori_pos, t0_pos, t1_pos):
#     # t0 is the target itself and t1 is the crowed one

#     pos_x, pos_y = int(ori_pos[0]), int(ori_pos[1])
#     _, off_h ,off_w = centripetal_map.shape

#     if pos_x >= off_w or pos_x >= off_h:
#         return centripetal_map

#     ori_x, ori_y = ori_pos
#     t0_x, t0_y = t0_pos

#     pos_ind = int(t0_x > ori_x)
#     cx0 = (-2) * (float(pos_ind) - 0.5) * (ori_x - t0_x) + SMALL_NUM
#     cx0 = torch.log(cx0)
#     cy0 = t0_y - ori_y

#     centripetal_map[pos_ind*2, pos_y, pos_x] = cx0
#     centripetal_map[pos_ind*2+1, pos_y, pos_x] = cy0 

#     if t1_pos == None:
#         return centripetal_map

#     t1_x, t1_y = t1_pos
#     pos1_ind = int(t1_x > ori_x)
#     cx1 = (-2) * (float(pos1_ind) - 0.5) * (ori_x - t1_x) + SMALL_NUM
#     cx1 = torch.log(cx1)
#     cy1 = t1_y - ori_y
#     centripetal_map[pos1_ind*2, pos_y, pos_x] = cx1
#     centripetal_map[pos1_ind*2+1, pos_y, pos_x] = cy1

#     return centripetal_map

# def set_dual_centripetal_shifts(centripetal_map, ori_pos, t0_pos, t1_pos):
#     # t0 is the target itself and t1 is the crowed one

#     pos_x, pos_y = int(ori_pos[0]), int(ori_pos[1])
#     _, off_h ,off_w = centripetal_map.shape

#     if pos_x >= off_w or pos_x >= off_h:
#         return centripetal_map

#     ori_x, ori_y = ori_pos
#     t0_x, t0_y = t0_pos

#     centripetal_map[0, pos_y, pos_x] = t0_x - ori_x
#     centripetal_map[1, pos_y, pos_x] = t0_y - ori_y

#     if t1_pos == None:
#         return centripetal_map

#     t1_x, t1_y = t1_pos
#     centripetal_map[2, pos_y, pos_x] = t1_x - ori_x
#     centripetal_map[3, pos_y, pos_x] = t1_y - ori_y

#     return centripetal_map

def set_dual_centripetal_shifts(centripetal_map, ori_pos, t0_pos, t1_pos):
    # t0 is the target itself and t1 is the crowed one

    pos_x, pos_y = int(ori_pos[0]), int(ori_pos[1])
    _, off_h ,off_w = centripetal_map.shape

    if pos_x >= off_w or pos_x >= off_h:
        return centripetal_map

    ori_x, ori_y = ori_pos
    t0_x, t0_y = t0_pos

    pos_ind = int(t0_y > ori_y)
    cx0 = t0_x - ori_x
    cy0 = t0_y - ori_y

    centripetal_map[pos_ind*2, pos_y, pos_x] = cx0
    centripetal_map[pos_ind*2+1, pos_y, pos_x] = cy0 

    if t1_pos == None:
        return centripetal_map

    t1_x, t1_y = t1_pos
    pos1_ind = int(t1_y > ori_y)
    cx1 = t1_x - ori_x
    cy1 = t1_y - ori_y
    centripetal_map[pos1_ind*2, pos_y, pos_x] = cx1
    centripetal_map[pos1_ind*2+1, pos_y, pos_x] = cy1

    return centripetal_map

def generate_self_conjugate_data(attr, batch_size):
    """pair attributes unrepeatedly
    Args:
        attr (Tensor): Some attibutes 
            Shape [batch, num_attr_items].
        batch_size (int): batch size of the original feature

    Returns:
        attr_pair (Tensor): Results of paired data.
            Shape [batch, num_conjugate_data, 2]
    """
    num_items = attr.size(-1)
    triu_ind = torch.triu_indices(num_items, num_items, 1, device=attr.device)
    # generate unrepeated pairs
    attr_a = attr.view(batch_size, num_items, 1, 1).expand(batch_size, num_items, num_items, 1)
    attr_b = attr.view(batch_size, 1, num_items, 1).expand(batch_size, num_items, num_items, 1)
    # pair shape: [batch, 2, a, b]
    attr_pair = torch.cat([attr_a, attr_b], dim=-1).permute(0,3,1,2)

    # pair shape : [batch, num_pairs, 2]
    attr_pair = attr_pair[:,:, triu_ind[0], triu_ind[1]].permute(0,2,1)

    return attr_pair

def generate_self_conjugate_data2(attr, batch_size, triu_ind):
    """Pair attributes unrepeatedly. This function needs additional
        triu_ind to slice unrepeated data.
    Args:
        attr (Tensor): Some attibutes 
            Shape [batch, num_attr_items].
        batch_size (int): batch size of the original feature.
        triu_ind (Tensor): ind to slice data.
            Shape [2, num_attr_items].

    Returns:
        attr_pair (Tensor): Results of paired data.
            Shape [batch, num_conjugate_data, 2]
    """
    num_items = attr.size(-1)
    # generate unrepeated pairs
    attr_a = attr.view(batch_size, num_items, 1, 1).expand(batch_size, num_items, num_items, 1)
    attr_b = attr.view(batch_size, 1, num_items, 1).expand(batch_size, num_items, num_items, 1)
    # pair shape: [batch, 2, a, b]
    attr_pair = torch.cat([attr_a, attr_b], dim=-1).permute(0,3,1,2)

    # pair shape : [batch, num_pairs, 2]
    attr_pair = attr_pair[:,:, triu_ind[0], triu_ind[1]].permute(0,2,1)

    return attr_pair

def generate_cross_paired_data(paired_attr_a, paired_attr_b, batch_size):
    """Reshape the input data for further computation.
        The input data should belong to the same attr
    Args:
        paired_attr_a (Tensor): Paired data a 
            Shape [batch, num_conjugate_data, 2]
        paired_attr_b (Tensor): Paired data b 
            Shape [batch, num_conjugate_data, 2]
        batch_size (int): batch size of the original feature

    Returns:
    tuplt[Tensor]: Results of reshaped data
        - a_cross (Tensor): Results of paired data.
            Shape [batch, num_paired_a, num_paired_b, 2]
        - b_cross (Tensor): Results of paired data.
            Shape [batch, num_paired_a, num_paired_b, 2]
    """
    num_a_pair, num_b_pair = paired_attr_a.size(1), paired_attr_b.size(1)
    a_cross = paired_attr_a.view(batch_size, num_a_pair, 1, 2).expand(batch_size, num_a_pair, num_b_pair, 2)
    b_cross = paired_attr_b.view(batch_size, 1, num_b_pair, 2).expand(batch_size, num_a_pair, num_b_pair, 2)
    return a_cross.contiguous(), b_cross.contiguous()
