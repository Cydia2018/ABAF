'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-10-06
@Email: xxxmy@foxmail.com
'''

import torch
import torch.nn as nn
from .config import DefaultConfig
import torch.nn.functional as F
from dataloader.encoder import DataEncoder


def _coords2boxes(coords, offsets):
    '''
    Args
    coords [sum(_h*_w),2]
    offsets [batch_size,sum(_h*_w),4] ltrb
    '''
    x1y1 = coords[None, :, :] - offsets[..., :2]
    x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
    boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
    return boxes


def _reshape_cat_out(inputs, strides):
    '''
    Args
    inputs: list contains five [batch_size,c,_h,_w]
    Returns
    out [batch_size,sum(_h*_w),c]
    coords [sum(_h*_w),2]
    '''
    batch_size = inputs[0].shape[0]
    c = inputs[0].shape[1]
    out = []
    coords = []
    for pred, stride in zip(inputs, strides):
        pred = pred.permute(0, 2, 3, 1)
        coord = coords_fmap2orig(pred, stride).to(device=pred.device)
        pred = torch.reshape(pred, [batch_size, -1, c])
        out.append(pred)
        coords.append(coord)
    return torch.cat(out, dim=1), torch.cat(coords, dim=0)


def _reshape_cat_out2(inputs, strides):
    '''
    Args
    inputs: list contains five [batch_size,c,_h,_w]
    [batch_size,sum(_h*_w),4]
    Returns
    out [batch_size,sum(_h*_w),c]
    coords [sum(_h*_w),2]
    '''
    batch_size = inputs[0].shape[0]
    c = inputs[0].shape[3]
    out = []
    coords = []
    for pred, stride in zip(inputs, strides):
        # pred = pred.permute(0, 2, 3, 1)
        coord = coords_fmap2orig(pred, stride).to(device=pred.device)
        pred = torch.reshape(pred, [batch_size, -1, c])
        out.append(pred)
        coords.append(coord)
    return torch.cat(out, dim=1), torch.cat(coords, dim=0)


def coords_fmap2orig(feature, stride):
    '''
    transfor one fmap coords to orig coords
    Args
    featurn [batch_size,h,w,c]
    stride int
    Returns 
    coords [n,2]
    '''
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super().__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        '''
        inputs  
        [0]list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w]  
        [1]gt_boxes [batch_size,m,4]  FloatTensor  
        [2]classes [batch_size,m]  LongTensor
        Returns
        cls_targets:[batch_size,sum(_h*_w),1]
        cnt_targets:[batch_size,sum(_h*_w),1]
        reg_targets:[batch_size,sum(_h*_w),4]
        '''
        cls_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), reg_targets_all_level

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.0):  # 1.5
        '''
        Args  
        out list contains [[batch_size,class_num,h,w],[batch_size,1,h,w],[batch_size,4,h,w]]  
        gt_boxes [batch_size,m,4]  
        classes [batch_size,m]  
        stride int  
        limit_range list [min,max]  
        Returns  
        cls_targets,cnt_targets,reg_targets
        '''
        cls_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]

        cls_logits = cls_logits.permute(0, 2, 3, 1)  # [batch_size,h,w,class_num]
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)  # [h*w,2]

        cls_logits = cls_logits.reshape((batch_size, -1, class_num))  # [batch_size,h*w,class_num]
        # cnt_logits=cnt_logits.permute(0,2,3,1)
        # cnt_logits=cnt_logits.reshape((batch_size,-1,1))
        # reg_preds=reg_preds.permute(0,2,3,1)
        # reg_preds=reg_preds.reshape((batch_size,-1,4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]

        # l_off1=x[None,:,None]-gt_boxes[...,0][:,None,:]    # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        # t_off1=y[None,:,None]-gt_boxes[...,1][:,None,:]       # int(512/stride)
        # r_off1=gt_boxes[...,2][:,None,:]-x[None,:,None]
        # b_off1=gt_boxes[...,3][:,None,:]-y[None,:,None]

        # new_voc3进行了归一化，且分配合理
        # voc9_nonorm未进行归一化，分配合理
        # voc9_nonorm_sample未进行归一化，分配合理，中心采样
        l_off = (x[None, :, None] - gt_boxes[..., 0][:, None, :])  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        t_off = (y[None, :, None] - gt_boxes[..., 1][:, None, :])
        r_off = (gt_boxes[..., 2][:, None, :] - x[None, :, None])
        b_off = (gt_boxes[..., 3][:, None, :] - y[None, :, None])
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)  # [batch_size,h*w,m,4]

        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])  # [batch_size,h*w,m]

        off_min = torch.min(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]
        off_max = torch.max(ltrb_off, dim=-1)[0]  # [batch_size,h*w,m]

        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio
        # -----------
        radiu_max = stride * 2.0

        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]  # [1,h*w,1]-[batch_size,1,m]-->[batch_size,h*w,m]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]


        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)  # [batch_size,h*w,m,4]
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]

        mask_center = c_off_max < radiu
        # -------
        mask_max = (c_off_max >= radiu) & (c_off_max <= radiu_max)

        mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size,h*w,m]
        # mask_pos = mask_in_gtboxes & mask_in_level
        
        # ---------------
        mask_ignore = mask_in_gtboxes & mask_in_level & mask_max  # [batch_size,h*w,m]
        # ---------------

        areas[~mask_pos] = 99999999
        areas_min_ind = torch.min(areas, dim=-1)[1]  # [batch_size,h*w]
        # reg_targets=ltrb_off[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]#[batch_size*h*w,4]
        reg_targets = ltrb_off[torch.zeros_like(areas, dtype=torch.uint8).scatter_(-1, areas_min_ind.unsqueeze(dim=-1),
                                                                                   1)]  # [batch_size*h*w,4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))  # [batch_size,h*w,4]

        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]  # [batch_size,h*w,m]
        # cls_targets=classes[torch.zeros_like(areas,dtype=torch.bool).scatter_(-1,areas_min_ind.unsqueeze(dim=-1),1)]
        cls_targets = classes[
            torch.zeros_like(areas, dtype=torch.uint8).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))  # [batch_size,h*w,1]

        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])  # [batch_size,h*w]
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)  # [batch_size,h*w,1]

        reg_targets = reg_targets / stride

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        mask_pos_2 = mask_pos.long().sum(dim=-1)  # [batch_size,h*w]
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)

        cls_targets[~mask_pos_2] = 0  # [batch_size,h*w,1]
        cnt_targets[~mask_pos_2] = -1
        reg_targets[~mask_pos_2] = -1

        # ------------------------
        mask_ignore_2 = mask_ignore.long().sum(dim=-1)  # [batch_size,h*w]
        mask_ignore_2 = mask_ignore_2.type(torch.uint8)
        cls_targets[mask_ignore_2] = -1  # [batch_size,h*w,1]
        cnt_targets[mask_ignore_2] = -1  # [batch_size,h*w,1]
        # ------------------------

        return cls_targets, cnt_targets, reg_targets


def compute_cls_loss(preds, targets, mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)  # [batch_size,sum(_h*_w),1]
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, class_num])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)  # [batch_size,sum(_h*_w),class_num]
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]
        target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None,
                      :] == target_pos).float()  # sparse-->onehot
        loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_cls_loss_anchor(preds, targets, mask):
    '''
    Args
    preds: list contains five level pred [batch_size,class_num,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    preds_reshape = []
    class_num = preds[0].shape[1]
    mask = mask.unsqueeze(dim=-1)  # [batch_size,sum(_h*_w),1]
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]

    batch_size = targets.shape[0]
    class_num = preds[0].shape[1]
    targets = targets.unsqueeze(dim=-1)  # [batch_size,sum(_h*_w),1]
    assert preds.shape[:2] == targets.shape[:2]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index]  # [sum(_h*_w),class_num]
        target_pos = targets[batch_index]  # [sum(_h*_w),1]
        target_pos = (torch.arange(1, class_num + 1, device=target_pos.device)[None,
                      :] == target_pos).float()  # sparse-->onehot
        loss.append(focal_loss_from_logits(pred_pos, target_pos).view(1))
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_cnt_loss(preds, targets, mask):
    '''
    Args  
    preds: list contains five level pred [batch_size,1,_h,_w]
    targets: [batch_size,sum(_h*_w),1]
    mask: [batch_size,sum(_h*_w)]
    '''
    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    mask = mask.unsqueeze(dim=-1)
    # mask=targets>-1#[batch_size,sum(_h*_w),1]
    num_pos = torch.sum(mask, dim=[1, 2]).clamp_(min=1).float()  # [batch_size,]
    for pred in preds:
        pred = pred.permute(0, 2, 3, 1)
        pred = torch.reshape(pred, [batch_size, -1, c])
        preds_reshape.append(pred)
    preds = torch.cat(preds_reshape, dim=1)
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),1]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,]
        assert len(pred_pos.shape) == 1
        loss.append(
            nn.functional.binary_cross_entropy_with_logits(input=pred_pos, target=target_pos, reduction='sum').view(1))
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_reg_loss(preds, targets, mask, mode='giou'):
    '''
    Args  
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''

    strides = [8, 16, 32, 64, 128]
    preds, coords = _reshape_cat_out(preds, strides)
    preds = _coords2boxes(coords, preds)  # [batch_size,sum(_h*_w),4]
    targets_new = []
    for target, stride in zip(targets, strides):
        target = target.view(target.shape[0], int(512 / stride), int(512 / stride), 4)
        targets_new.append(target)
    targets = targets_new
    targets, coords = _reshape_cat_out2(targets, strides)
    targets = _coords2boxes(coords, targets)  # 转为xyxy  [batch_size,sum(_h*_w),4]
    # targets = torch.cat(targets, dim=1)

    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    '''
    for pred in preds:
        # pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    '''
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_pos.shape) == 2
        if mode == 'iou':
            loss.append(iou_loss(pred_pos, target_pos).view(1))
        elif mode == 'giou':
            # loss.append(giou_loss_anchor(pred_pos, target_pos).view(1))
            # -----------------
            loss.append(diou_loss(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_reg_loss_iou(preds, targets, mask, mode='giou'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''

    strides = [8, 16, 32, 64, 128]
    preds, coords = _reshape_cat_out(preds, strides)
    preds = _coords2boxes(coords, preds)  # [batch_size,sum(_h*_w),4]
    targets_new = []
    for target, stride in zip(targets, strides):
        target = target.view(target.shape[0], int(512 / stride), int(512 / stride), 4)
        targets_new.append(target)
    targets = targets_new
    targets, coords = _reshape_cat_out2(targets, strides)
    targets = _coords2boxes(coords, targets)  # 转为xyxy  [batch_size,sum(_h*_w),4]
    # targets = torch.cat(targets, dim=1)

    batch_size = targets.shape[0]
    c = targets.shape[-1]
    preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    '''
    for pred in preds:
        # pred=pred.permute(0,2,3,1)
        pred=torch.reshape(pred,[batch_size,-1,c])
        preds_reshape.append(pred)
    preds=torch.cat(preds_reshape,dim=1)
    '''
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_pos.shape) == 2
        if mode == 'iou':
            loss.append(iou_loss(pred_pos, target_pos).view(1))
        elif mode == 'giou':
            # loss.append(giou_loss(pred_pos,target_pos).view(1))
            # -----------------
            loss.append(iou_loss_xyxy(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def compute_reg_loss_anchor(preds, targets, pos, mode='giou'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''
    '''
    batch_size = targets.shape[0]
    # c = targets.shape[-1]
    # preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    assert preds.shape==targets.shape#[batch_size,sum(_h*_w),4]
    loss=[]
    for batch_index in range(batch_size):
        pred_pos=preds[batch_index][mask[batch_index]]#[num_pos_b,4]
        target_pos=targets[batch_index][mask[batch_index]]#[num_pos_b,4]
        assert len(pred_pos.shape)==2
        if mode=='iou':
            loss.append(iou_loss(pred_pos,target_pos).view(1))
        elif mode=='giou':
            loss.append(giou_loss_anchor(pred_pos,target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    return torch.cat(loss,dim=0)/num_pos#[batch_size,]
    '''

    # L1_loss函数
    num_pos = pos.data.long().sum()
    mask = pos.unsqueeze(2).expand_as(preds)  # [N,#anchors,4]
    masked_loc_preds = preds[mask].view(-1, 4)  # [#pos,4]
    masked_loc_targets = targets[mask].view(-1, 4)  # [#pos,4]
    loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
    return loc_loss / num_pos


def compute_reg_loss_anchor_giou(preds, targets, mask, mode='giou'):
    '''
    Args
    preds: list contains five level pred [batch_size,4,_h,_w]
    targets: [batch_size,sum(_h*_w),4]
    mask: [batch_size,sum(_h*_w)]
    '''

    batch_size = targets.shape[0]
    # c = targets.shape[-1]
    # preds_reshape = []
    # mask=targets>-1#[batch_size,sum(_h*_w),4]
    num_pos = torch.sum(mask, dim=1).clamp_(min=1).float()  # [batch_size,]
    assert preds.shape == targets.shape  # [batch_size,sum(_h*_w),4]
    loss = []
    for batch_index in range(batch_size):
        pred_pos = preds[batch_index][mask[batch_index]]  # [num_pos_b,4]
        target_pos = targets[batch_index][mask[batch_index]]  # [num_pos_b,4]
        assert len(pred_pos.shape) == 2
        if mode == 'iou':
            loss.append(iou_loss(pred_pos, target_pos).view(1))
        elif mode == 'giou':
            # test = giou_loss_anchor(pred_pos,target_pos)
            # test = test.view(1)
            # loss.append(giou_loss_anchor(pred_pos, target_pos).view(1))
            # -----------
            loss.append(diou_loss(pred_pos, target_pos).view(1))
            # loss.append(iou_loss_xyxy(pred_pos, target_pos).view(1))
        else:
            raise NotImplementedError("reg loss only implemented ['iou','giou']")
    # tt = torch.stack(loss)/num_pos
    return torch.cat(loss, dim=0) / num_pos  # [batch_size,]


def iou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    lt = torch.min(preds[:, :2], targets[:, :2])
    rb = torch.min(preds[:, 2:], targets[:, 2:])
    wh = (rb + lt).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    iou = overlap / (area1 + area2 - overlap)
    loss = -iou.clamp(min=1e-6).log()
    return loss.sum()


def iou_loss_xyxy(preds,targets):
    xy1 = torch.max(preds[:, :2], targets[:, :2])
    xy2 = torch.min(preds[:, 2:], targets[:, 2:])
    wh = (xy2 - xy1).clamp(min=0)
    overlap = wh[:, 0] * wh[:, 1]
    area1 = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area2 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    iou = overlap / (area1 + area2 - overlap)
    loss = -iou.clamp(min=1e-6).log()
    return loss.sum()

'''
def iou_loss_xyxy(bboxes1, bboxes2):
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols)).cuda()
    if rows * cols == 0:
        # gious = 1. - ious
        return ious.sum()
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows)).cuda()
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
            bboxes2[:, 3] - bboxes2[:, 1])
    # 最大框和最小框
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # inter_max_rb

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # inter_min_lt

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)  # inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    union = area1 + area2 - inter_area

    ious = inter_area / union
    return ious.sum()
'''


def giou_loss(preds, targets):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''

    lt_min = torch.min(preds[:, :2], targets[:, :2])
    rb_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (rb_min + lt_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] + preds[:, 0]) * (preds[:, 3] + preds[:, 1])
    area2 = (targets[:, 2] + targets[:, 0]) * (targets[:, 3] + targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union

    lt_max = torch.max(preds[:, :2], targets[:, :2])
    rb_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (rb_max + lt_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    loss = 1. - giou
    return loss.sum()

    '''
    bboxes1 = preds
    bboxes2 = targets
    area1 = (bboxes1[:, 2] + bboxes1[:, 0]) * (
            bboxes1[:, 3] + bboxes1[:, 1])
    area2 = (bboxes2[:, 2] + bboxes2[:, 0]) * (
            bboxes2[:, 3] + bboxes2[:, 1])
    # 最大框和最小框
    inter_min_rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])    # rb_min

    inter_max_lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])    # lt_max

    out_max_rb = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])      # rb_max

    out_min_lt = torch.min(bboxes1[:, :2], bboxes2[:, :2])      # lt_min
    inter = torch.clamp((inter_min_rb + out_min_lt), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_rb + inter_max_lt), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area
    
    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious, min=-1.0, max=1.0)
    gious = 1. - ious
    return gious.sum()
    '''


def giou_loss_anchor(bboxes1, bboxes2):
    '''
    Args:
    preds: [n,4] ltrb
    targets: [n,4]
    '''
    '''
    # xy1_p = torch.cat((preds[:, 0].unsqueeze(-1), preds[:, 2].unsqueeze(-1)), dim=1)
    # xy2_p = torch.cat((preds[:, 1].unsqueeze(-1), preds[:, 3].unsqueeze(-1)), dim=1)
    # xy1_t = torch.cat((targets[:, 0].unsqueeze(-1), targets[:, 2].unsqueeze(-1)), dim=1)
    # xy2_t = torch.cat((targets[:, 1].unsqueeze(-1), targets[:, 3].unsqueeze(-1)), dim=1)
    xy1_min = torch.min(preds[:, :2], targets[:, :2])
    xy2_min = torch.min(preds[:, 2:], targets[:, 2:])
    wh_min = (xy2_min - xy1_min).clamp(min=0)
    overlap = wh_min[:, 0] * wh_min[:, 1]  # [n]
    area1 = (preds[:, 2] - preds[:, 0]) * (preds[:, 3] - preds[:, 1])
    area2 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
    union = (area1 + area2 - overlap)
    iou = overlap / union

    xy1_max = torch.max(preds[:, :2], targets[:, :2])
    xy2_max = torch.max(preds[:, 2:], targets[:, 2:])
    wh_max = (xy2_max - xy1_max).clamp(0)
    G_area = wh_max[:, 0] * wh_max[:, 1]  # [n]

    giou = iou - (G_area - union) / G_area.clamp(1e-10)
    loss = 1. - giou
    # print(loss)
    return loss.sum()
    '''
    """
        Calculate the gious between each bbox of bboxes1 and bboxes2.
        Args:
            bboxes1(ndarray): shape (n, 4)
            bboxes2(ndarray): shape (k, 4)
        Returns:
            gious(ndarray): shape (n, k)
    """

    # bboxes1 = torch.FloatTensor(bboxes1)
    # bboxes2 = torch.FloatTensor(bboxes2)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    ious = torch.zeros((rows, cols)).cuda()
    if rows * cols == 0:
        gious = 1. - ious
        return gious.sum()
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        ious = torch.zeros((cols, rows)).cuda()
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (
            bboxes1[:, 3] - bboxes1[:, 1])
    area2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (
            bboxes2[:, 3] - bboxes2[:, 1])
    # 最大框和最小框
    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # inter_max_rb

    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # inter_min_lt

    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])  # out_max_rb

    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])  # out_min_lt

    inter = torch.clamp((inter_max_xy - inter_min_xy),
                        min=0)  # inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_area = outer[:, 0] * outer[:, 1]
    union = area1 + area2 - inter_area
    closure = outer_area

    ious = inter_area / union - (closure - union) / closure
    ious = torch.clamp(ious, min=-1.0, max=1.0)
    # print(ious)
    if exchange:
        ious = ious.T
    gious = 1. - ious
    return gious.sum()


def diou_loss(bboxes1, bboxes2):
    rows = bboxes1.shape[0]  # 第一个框的个数
    cols = bboxes2.shape[0]  # 第二个框的个数
    dious = torch.zeros((rows, cols)).cuda()  # 初始化dious变量
    if rows * cols == 0:
        dious = 1. - dious
        return dious.sum()
        # return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows)).cuda()
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2  # （x1max +x1min）/2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2  # (y1max+y1min)/2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # max((x1min,y1min ),(x2min,y2min))->返回较大的一组
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    dious = 1. - dious
    return dious.sum()


def diou_loss_anchor(bboxes1, bboxes2):
    rows = bboxes1.shape[0]  # 第一个框的个数
    cols = bboxes2.shape[0]  # 第二个框的个数
    dious = torch.zeros((rows, cols))  # 初始化dious变量
    if rows * cols == 0:
        dious = 1. - dious
        return dious.sum()
        # return dious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        dious = torch.zeros((cols, rows))
        exchange = True
    # #xmin,ymin,xmax,ymax->[:,0],[:,1],[:,2],[:,3]
    w1 = bboxes1[:, 2] - bboxes1[:, 0]
    h1 = bboxes1[:, 3] - bboxes1[:, 1]
    w2 = bboxes2[:, 2] - bboxes2[:, 0]
    h2 = bboxes2[:, 3] - bboxes2[:, 1]

    area1 = w1 * h1
    area2 = w2 * h2

    center_x1 = (bboxes1[:, 2] + bboxes1[:, 0]) / 2  # （x1max +x1min）/2
    center_y1 = (bboxes1[:, 3] + bboxes1[:, 1]) / 2  # (y1max+y1min)/2
    center_x2 = (bboxes2[:, 2] + bboxes2[:, 0]) / 2
    center_y2 = (bboxes2[:, 3] + bboxes2[:, 1]) / 2

    inter_max_xy = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # min((x1max,y1max ),(x2max,y2max)) ->返回较小一组
    inter_min_xy = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # max((x1min,y1min ),(x2min,y2min))->返回较大的一组
    out_max_xy = torch.max(bboxes1[:, 2:], bboxes2[:, 2:])
    out_min_xy = torch.min(bboxes1[:, :2], bboxes2[:, :2])

    inter = torch.clamp((inter_max_xy - inter_min_xy), min=0)
    inter_area = inter[:, 0] * inter[:, 1]
    inter_diag = (center_x2 - center_x1) ** 2 + (center_y2 - center_y1) ** 2
    outer = torch.clamp((out_max_xy - out_min_xy), min=0)
    outer_diag = (outer[:, 0] ** 2) + (outer[:, 1] ** 2)
    union = area1 + area2 - inter_area
    dious = inter_area / union - (inter_diag) / outer_diag
    dious = torch.clamp(dious, min=-1.0, max=1.0)
    if exchange:
        dious = dious.T
    dious = 1. - dious
    return dious.sum()


def focal_loss_from_logits(preds, targets, gamma=2.0, alpha=0.25):
    '''
    Args:
    preds: [n,class_num] 
    targets: [n,class_num]
    '''
    preds = preds.sigmoid()
    pt = preds * targets + (1.0 - preds) * (1.0 - targets)
    w = alpha * targets + (1.0 - alpha) * (1.0 - targets)
    loss = -w * torch.pow((1.0 - pt), gamma) * pt.log()
    return loss.sum()


class LOSS(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs):
        '''
        inputs list
        [0]preds:  ....
        [1]targets : list contains three elements [[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),1],[batch_size,sum(_h*_w),4]]
        '''
        preds, targets = inputs
        cls_logits, reg_preds = preds
        cls_targets, cnt_targets, reg_targets = targets
        mask_pos = (cnt_targets > -1).squeeze(dim=-1)  # [batch_size,sum(_h*_w)]
        cls_loss = compute_cls_loss(cls_logits, cls_targets, mask_pos).mean()  # []
        # cnt_loss=compute_cnt_loss(cnt_logits,cnt_targets,mask_pos).mean()
        # ----------
        # reg_loss = compute_reg_loss_iou(reg_preds, reg_targets, mask_pos).mean()
        reg_loss = compute_reg_loss(reg_preds, reg_targets, mask_pos).mean()
        if self.config.add_centerness:
            # total_loss=cls_loss+cnt_loss+reg_loss
            total_loss = cls_loss + reg_loss
            return cls_loss, reg_loss, total_loss
        else:
            # total_loss=cls_loss+reg_loss+cnt_loss*0.0
            total_loss = cls_loss + reg_loss
            return cls_loss, reg_loss, total_loss


class LOSS_anchor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.encoder = DataEncoder()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        mask = cls_targets > 0  # [N,#anchors]
        loc_preds_decode = self.encoder.decode_loss(loc_preds, (512, 512))
        loc_targets_decode = self.encoder.decode_loss(loc_targets, (512, 512))
        cls_loss_anchor = compute_cls_loss_anchor(cls_preds, cls_targets, mask).mean()
        reg_loss_anchor = compute_reg_loss_anchor_giou(loc_preds_decode, loc_targets_decode, mask).mean()
        total_loss_anchor = cls_loss_anchor + reg_loss_anchor
        return reg_loss_anchor, cls_loss_anchor, total_loss_anchor


if __name__ == "__main__":
    loss = compute_cnt_loss([torch.ones([2, 1, 4, 4])] * 5, torch.ones([2, 80, 1]),
                            torch.ones([2, 80], dtype=torch.bool))
    print(loss)
