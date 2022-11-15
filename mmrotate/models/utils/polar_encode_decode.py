import sys
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from mmcv.ops import min_area_polygons

'''encode'''
def arrange_order(tt, rr, bb, ll):
    """
    点顺序定义：
    1. 水平框
    这种情况下，t表示y坐标最大的点，b表示y坐标最小的点，l表示x坐标最大的点，r表示x坐标最小的点
    2. 旋转框
    这种情况下，t表示第一象限点，r表示第二象限点，b表示第三象限点，l表示第四象限点
    """
    pts = torch.stack([tt, rr, bb, ll])
    # 如果有零，说明向量垂直
    if (pts == 0).sum() > 0:
        l_ind = torch.argmin(pts[:,0])
        r_ind = torch.argmax(pts[:,0])
        t_ind = torch.argmin(pts[:,1])
        b_ind = torch.argmax(pts[:,1])
        tt_new = pts[t_ind,:]
        rr_new = pts[r_ind,:]
        bb_new = pts[b_ind,:]
        ll_new = pts[l_ind,:]
        return tt_new, rr_new, bb_new, ll_new       
    else:
        # tt_ind = (pts[:,0] > 0) & (pts[:,1] > 0)
        # rr_ind = (pts[:,0] < 0) & (pts[:,1] > 0)
        # bb_ind = (pts[:,0] < 0) & (pts[:,1] < 0)
        # ll_ind = (pts[:,0] > 0) & (pts[:,1] < 0)
        # tt_new, rr_new, bb_new, ll_new = pts[tt_ind], pts[rr_ind], pts[bb_ind], pts[ll_ind]

        # return tt_new[0], rr_new[0], bb_new[0], ll_new[0]
        return tt, rr, bb, ll

def calculate_point_angle(point):
    """
    返回一个直角坐标系下点的角度，范围是(-pi, pi]
    """
    x = point[0];
    y = point[1];
    return torch.atan2(y, x)

def between_range(pre_angle, nxt_angle, target_angle):
    """
    判断target_angle是否在(pre_angle, nxt_angle]范围内，要考虑(-pi, pi]
    """

    '''
    当a->b->c->d这条角度链可能存在几个情况:
    1. pre < nxt 只要pre和nxt连线不过x负半轴，就不会有问题；
       这种情况就直接判断角度范围即可
    2. pre > nxt:
       pre > 0, nxt < 0: 说明pre在12象限，nxt在34象限，判断这个范围即可，注意x负半轴角度是正pi;
    3. pre == nxt: 不可能发生，如果发生，应该报错

    '''

    if pre_angle < nxt_angle:
        if target_angle > pre_angle and target_angle <= nxt_angle:
            return True
        else:
            return False
    elif pre_angle > nxt_angle:
        assert pre_angle > 0 and nxt_angle < 0
        if (target_angle > pre_angle and target_angle <= np.pi) or (target_angle > - np.pi and target_angle <= nxt_angle):
            return True
        else:
            return False
    else:
        raise("OBB with zero width/height!")

def calculate_distance(t_ag, a1, a2, r1, r2):
    """
    给定极坐标下两点(r1, a1), (r2, a2)，计算角度t_ag的射线与这两点确定的直线的交点距极点的距离
    """
    return r1 * r2 * math.sin(a1 - a2) / (r2 * math.sin(t_ag - a2) + r1 * math.sin(a1 - t_ag))


def polar_encode(pts:torch.tensor, ct:torch.tensor, n=8):
    '''
    该函数只处理一张图中一个目标的标注信息，将之从4点标注转为极坐标距离标注
    pts  (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    ct   (torch.Tensor): [ctx, cty]
    n: 将0~180度分为几个部分，生成几个标注点，如每隔45度一点的话，n=4。
    return: polar_pts_n表示极坐标系下每隔180/n角度，矩形框边界点与原点（中心点）产生的距离，是一个n长list。
    '''

    bl = pts[0:2]
    tl = pts[2:4]
    tr = pts[4:6]
    br = pts[6:8]

    tt = (tl+tr)/2 - ct
    rr = (tr+br)/2 - ct
    bb = (bl+br)/2 - ct
    ll = (tl+bl)/2 - ct

    # 转换为了有序的四个象限坐标
    p1, p2, p3, p4 = arrange_order(tt, rr, bb, ll)
    
    # 得到四个角点
    a_pt, b_pt, c_pt, d_pt = p1 + p2, p2 + p3, p3 + p4, p4 + p1

    # 得到四个角点对应的角度，方便计算交点，注意这里角度范围[0, 2*pi)
    a_ag, b_ag, c_ag, d_ag = calculate_point_angle(a_pt), calculate_point_angle(b_pt), \
                             calculate_point_angle(c_pt), calculate_point_angle(d_pt)

    # 接下来给定一个角度，需要计算这个角度射线与边界上某一条边的交点
    # 获取n个标注点，这些点是对应角度射线与包围盒边界的交点

    # 范围是[0, pi)内取n个点
    delta_angle = np.pi / n
    neighbor_pts_angle = pts.new_tensor([[a_ag, b_ag], [b_ag, c_ag], [c_ag, d_ag], [d_ag, a_ag]])
    anomaly_neighbor_inds = neighbor_pts_angle[:,1] < neighbor_pts_angle[:,0]
    neighbor_pts_angle[anomaly_neighbor_inds,1] = neighbor_pts_angle[anomaly_neighbor_inds,1] + np.pi*2

    corner_pts = torch.stack([a_pt, b_pt, c_pt, d_pt])

    target_angles = torch.arange(n, device=pts.device) * delta_angle
    target_angles_expand = target_angles.unsqueeze(-1).expand(n,4)
    neighbor_pts_angle_expand = neighbor_pts_angle.unsqueeze(0).expand(n,4,2)

    valid_range_ind = (target_angles_expand >= neighbor_pts_angle_expand[...,0]).float() + \
                  (target_angles_expand < neighbor_pts_angle_expand[...,1]).float()
    valid_range_ind_id = valid_range_ind.ge(2)*torch.arange(4, device=target_angles.device)
    valid_range_ind_id = valid_range_ind_id.sum(-1, keepdim=True)
    valid_corner_pt_inds = torch.cat([valid_range_ind_id, valid_range_ind_id+1], dim=-1).fmod(4)
    selected_corner_pt_loc = corner_pts[valid_corner_pt_inds]
    selected_corner_pt_a = torch.atan2(selected_corner_pt_loc[...,1], selected_corner_pt_loc[...,0])
    selected_corner_pt_r = selected_corner_pt_loc.pow(2).sum(-1).sqrt()
    polar_length = selected_corner_pt_r[:,0]*selected_corner_pt_r[:,1]*torch.sin(selected_corner_pt_a[:,0]-selected_corner_pt_a[:,1]) /\
                    (selected_corner_pt_r[:,1]*torch.sin(target_angles-selected_corner_pt_a[:,1])+
                    selected_corner_pt_r[:,0]*torch.sin(selected_corner_pt_a[:,0]-target_angles))
    
    assert torch.isnan(polar_length).any() == False
    
    return polar_length

# from mmrotate.core.bbox.transforms import poly2obb, obb2poly
# cen_x, cen_y = 10., 10.
# bbox_w, bbox_h = 10., 15.
# theta = 0.
# poly = obb2poly(torch.tensor([[cen_x, cen_y, bbox_w, bbox_h, theta]]))
# ctx = torch.tensor([cen_x, cen_y])
# # 将四点标注法转为极坐标距离标注
# res = polar_encode(poly[0], ctx, 8)
# print(res)

# def polar_encode(pts:torch.tensor, ct:torch.tensor, n):
#     '''
#     该函数只处理一张图中一个目标的标注信息，将之从4点标注转为极坐标距离标注
#     pts  (torch.Tensor): [[x0,y0,x1,y1,x2,y2,x3,y3],]
#     ct   (torch.Tensor): [[ctx, cty],]
#     n: 将0~180度分为几个部分，生成几个标注点，如每隔45度一点的话，n=4。
#     return: polar_pts_n表示极坐标系下每隔180/n角度，矩形框边界点与原点（中心点）产生的距离，是一个n长list。
#     '''
#     n_box = len(pts)

#     bl = pts[...,0:2]
#     tl = pts[...,2:4]
#     tr = pts[...,4:6]
#     br = pts[...,6:8]

#     tt = (tl+tr)/2 - ct
#     rr = (tr+br)/2 - ct
#     bb = (bl+br)/2 - ct
#     ll = (tl+bl)/2 - ct

#     # 转换为了有序的四个象限坐标
#     p1, p2, p3, p4 = torch.zeros_like(tt), torch.zeros_like(tt), torch.zeros_like(tt), torch.zeros_like(tt)
#     for i in range(n_box):
#         p1[i], p2[i], p3[i], p4[i] = arrange_order(tt[i], rr[i], bb[i], ll[i])
    
#     # 得到四个角点
#     a_pt, b_pt, c_pt, d_pt = p1 + p2, p2 + p3, p3 + p4, p4 + p1

#     # 得到四个角点对应的角度，方便计算交点，注意这里角度范围[0, 2*pi)
#     a_ag, b_ag, c_ag, d_ag = calculate_point_angle(a_pt), calculate_point_angle(b_pt), \
#                              calculate_point_angle(c_pt), calculate_point_angle(d_pt)

#     # 接下来给定一个角度，需要计算这个角度射线与边界上某一条边的交点
#     # 获取n个标注点，这些点是对应角度射线与包围盒边界的交点

#     # 范围是[0, pi)内取n个点
#     delta_angle = np.pi / n
#     neighbor_pts_angle = torch.stack([torch.stack([a_ag, b_ag], dim=-1), 
#                                       torch.stack([b_ag, c_ag], dim=-1), 
#                                       torch.stack([c_ag, d_ag], dim=-1), 
#                                       torch.stack([d_ag, a_ag], dim=-1)], dim=-2)
#     anomaly_neighbor_inds = neighbor_pts_angle[...,1] < neighbor_pts_angle[...,0]
#     anomaly_nz = anomaly_neighbor_inds.nonzero()
#     neighbor_pts_angle[anomaly_nz[:,0], anomaly_nz[:,1]][1] = neighbor_pts_angle[anomaly_nz[:,0], anomaly_nz[:,1]][1] + np.pi*2

#     corner_pts = torch.stack([a_pt, b_pt, c_pt, d_pt])

#     target_angles = torch.arange(n, device=pts.device) * delta_angle
#     target_angles_expand = target_angles.unsqueeze(-1).expand(n,4).unsqueeze(0).expand(n_box, n, 4)
#     target_angles = target_angles.unsqueeze(0).expand(n_box, n).flatten()
#     neighbor_pts_angle_expand = neighbor_pts_angle.unsqueeze(1).expand(n_box, n,4,2)

#     valid_range_ind = (target_angles_expand >= neighbor_pts_angle_expand[...,0]).float() + \
#                   (target_angles_expand < neighbor_pts_angle_expand[...,1]).float()
#     valid_range_ind_id = valid_range_ind.ge(2)*torch.arange(4, device=target_angles.device)
#     valid_range_ind_id = valid_range_ind_id.sum(-1, keepdim=True)
#     valid_corner_pt_inds = torch.cat([valid_range_ind_id, valid_range_ind_id+1], dim=-1).fmod(4).view(-1, 2)
#     selected_corner_pt_loc = corner_pts.view(-1,2)[valid_corner_pt_inds]
#     selected_corner_pt_a = torch.atan2(selected_corner_pt_loc[...,1], selected_corner_pt_loc[...,0])
#     selected_corner_pt_r = selected_corner_pt_loc.pow(2).sum(-1).sqrt()
#     polar_length = selected_corner_pt_r[...,0]*selected_corner_pt_r[...,1]*torch.sin(selected_corner_pt_a[...,0]-selected_corner_pt_a[...,1]) /\
#                     (selected_corner_pt_r[...,1]*torch.sin(target_angles-selected_corner_pt_a[...,1])+
#                     selected_corner_pt_r[...,0]*torch.sin(selected_corner_pt_a[...,0]-target_angles))
    
#     return polar_length.view(n_box, n)

'''decode'''
def polar_decode(polar_encodings, xs, ys, n=8):
    '''
    Decode input polar encodings into bbavs
    polar_encodings  (torch.Tensor): [batch, num_dets, n]
    xs               (torch.Tensor): [batch, num_dets, 1]
    ys               (torch.Tensor): [batch, num_dets, 1]
    n                (int):     number of encoding vectors 
    
    return:          (torch.Tensor): [batch, num_dets, n*4]
    '''

    batch, num_dets = polar_encodings.size()[:2]

    # 2. 转为直角坐标并补全
    angles = torch.linspace(0, np.pi, n+1, device=polar_encodings.device)[:-1].unsqueeze(0).unsqueeze(0).repeat(1,num_dets,1)
    wh_x = torch.mul(polar_encodings, torch.cos(angles))
    wh_y = torch.mul(polar_encodings, torch.sin(angles))
    wh_x_symmetry = -1 * wh_x
    wh_y_symmetry = -1 * wh_y

    wh_x = wh_x + xs if xs is not None else wh_x
    wh_y = wh_y + ys if ys is not None else wh_y
    wh_x_symmetry = wh_x_symmetry + xs  if xs is not None else wh_x_symmetry
    wh_y_symmetry = wh_y_symmetry + ys  if ys is not None else wh_y_symmetry

    target_pts = torch.stack([wh_x, wh_y, wh_x_symmetry, wh_y_symmetry], dim=-1)
    target_pts = target_pts.view(batch, num_dets,-1).contiguous()

    return target_pts