'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

from .head import ClsCntRegHead, anchor_Head
from .fpn import FPN
from .backbone.resnet import resnet50
import torch.nn as nn
from .loss import GenTargets, LOSS, coords_fmap2orig, LOSS_anchor
from .loss_anchor_new import FocalLoss
import torch
import numpy as np
from .config import DefaultConfig
from dataloader.encoder import DataEncoder


class FCOS(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.backbone = resnet50(pretrained=config.pretrained, if_include_top=False)
        self.fpn = FPN(config.fpn_out_channels, use_p5=config.use_p5)
        self.head = ClsCntRegHead(config.fpn_out_channels, config.class_num,
                                  config.use_GN_head, config.cnt_on_reg, config.prior)
        # self.anchor_head = anchor_Head(config.class_num)
        self.config = config
        self.strides = self.config.strides

    def train(self, mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False

        if self.config.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.config.freeze_stage_1:
            self.backbone.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self, x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3, C4, C5 = self.backbone(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, reg_preds, loc_preds, cls_preds = self.head(all_P, x.size(0))
        # loc_preds, cls_preds = self.anchor_head(all_P, x.size(0))
        # reg_preds, coords = self._reshape_cat_out(reg_preds, self.strides)
        # reg_preds = self._coords2boxes(coords, reg_preds)
        out = {'fcos': [cls_logits, reg_preds], 'loc_preds': loc_preds, 'cls_preds': cls_preds}

        return out


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, config=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if config is None:
            self.config = DefaultConfig
        else:
            self.config = config

    def forward(self, inputs, nms_anchor):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]  
        cls_logits  list contains five [batch_size,class_num,h,w]  
        cnt_logits  list contains five [batch_size,1,h,w]  
        reg_preds   list contains five [batch_size,4,h,w] 
        '''
        cls_logits, _  = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]   [sum(_h*_w),1]
        # cnt_logits,_=self._reshape_cat_out(inputs[1],self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, coords = self._reshape_cat_out2(inputs[1], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()
        # cnt_preds=cnt_logits.sigmoid_()

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # both[batch_size,sum(_h*_w)]
        # cls_scores = (cls_scores * 1.3).clamp(max=0.998)
        boxes_anchor, labels_anchor, score_anchor = nms_anchor
        # score_anchor = score_anchor * 0.8

        # xx = labels_anchor.numpy()
        # yy = score_anchor.numpy()
        # cnt_anchor = cnt_preds.clone()
        # cnt_anchor = cnt_anchor.repeat(1, 1, 3).view(-1, 16368)     # 49104
        # score_anchor = score_anchor * (cnt_anchor.squeeze(dim=-1))

        if self.config.add_centerness:
            # cls_scores = cls_scores * (cnt_preds.squeeze(dim=-1))   # [batch_size,sum(_h*_w)]
            pass
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]
        labels_anchor = labels_anchor + 1

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]
        # boxes -- boxes_anchor         [b,5456,4] -- [b,49104,4]
        # cls_scores -- score_anchor    [b,5456] -- [b,49104]
        # cls_classes -- labels_anchor  [b,5456] -- [b,49104]

        # --
        # max_num = 150
        # topk_ind_anchor = torch.topk(score_anchor, max_num, dim=-1, largest=True, sorted=True)[1]
        # topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]
        # xx = score_anchor[:, topk_ind_anchor]
        # yy = cls_scores[:, topk_ind]
        # --

        # cls_scores = torch.cat((cls_scores, score_anchor), dim=1)           # [b,54560]
        # cls_classes = torch.cat((cls_classes, labels_anchor), dim=1)        # [b,54560]
        # boxes = torch.cat((boxes, boxes_anchor), dim=1)                     # [b,54560,4]
        # cls_scores = score_anchor             # [b,54560]
        # cls_classes = labels_anchor           # [b,54560]
        # boxes = boxes_anchor                  # [b,54560,4]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        # print(cls_scores[:, topk_ind])
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        # ---
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]

        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        # boxes_anchor, labels_anchor, score_anchor = nms_anchor

        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            # _cls_scores_b = torch.cat((_cls_scores_b, score_anchor), dim=0)
            # _cls_classes_b = torch.cat((_cls_classes_b, labels_anchor), dim=0)
            # _boxes_b = torch.cat((_boxes_b, boxes_anchor), dim=0)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])

        max_num = 100
        for i in range(len(_cls_scores_post)):
            _cls_scores_post[i] = torch.nn.functional.pad(_cls_scores_post[i],
                                                          (0, max_num - _cls_scores_post[i].shape[0]), value=-2)
        for i in range(len(_cls_classes_post)):
            _cls_classes_post[i] = torch.nn.functional.pad(_cls_classes_post[i],
                                                           (0, max_num - _cls_classes_post[i].shape[0]), value=-2)
        for i in range(len(_boxes_post)):
            _boxes_post[i] = torch.nn.functional.pad(_boxes_post[i], (0, 0, 0, max_num - _boxes_post[i].shape[0]),
                                                     value=-2)

        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            # aaa = order[1:]
            # bbb = areas[order[1:]]
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    @staticmethod
    def box_nms_diou(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inter_diag = (center_x[i] - center_x[order[1:]]) ** 2 + (center_y[i] - center_y[order[1:]]) ** 2
            out_xmin = x1[order[1:]].clamp(max=float(x1[i]))
            out_ymin = y1[order[1:]].clamp(max=float(y1[i]))
            out_xmax = x2[order[1:]].clamp(min=float(x2[i]))
            out_ymax = y2[order[1:]].clamp(min=float(y2[i]))
            outer_diag = (out_xmax - out_xmin) ** 2 + (out_ymax - out_ymin) ** 2
            iou = iou - inter_diag / outer_diag

            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    @staticmethod
    def box_softnms(boxes, scores, thr, sigma=0.5):
        """
        boxes: [?,4]
        scores: [?]
        """
        N = boxes.shape[0]
        # thr = 0.001
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        indexes = torch.arange(0, N, dtype=torch.float).cuda().view(N, 1)
        boxes = torch.cat((boxes, indexes), dim=1)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)

        for i in range(N):
            # intermediate parameters for later parameters exchange
            tscore = scores[i].clone()
            pos = i + 1

            if i != N - 1:
                maxscore, maxpos = torch.max(scores[pos:], dim=0)
                if tscore < maxscore:
                    boxes[i], boxes[maxpos.item() + i + 1] = boxes[maxpos.item() + i + 1].clone(), boxes[i].clone()
                    scores[i], scores[maxpos.item() + i + 1] = scores[maxpos.item() + i + 1].clone(), scores[i].clone()
                    areas[i], areas[maxpos + i + 1] = areas[maxpos + i + 1].clone(), areas[i].clone()

            # IoU calculate
            '''
            yy1 = np.maximum(boxes[i, 0].to("cpu").numpy(), boxes[pos:, 0].to("cpu").numpy())
            xx1 = np.maximum(boxes[i, 1].to("cpu").numpy(), boxes[pos:, 1].to("cpu").numpy())
            yy2 = np.minimum(boxes[i, 2].to("cpu").numpy(), boxes[pos:, 2].to("cpu").numpy())
            xx2 = np.minimum(boxes[i, 3].to("cpu").numpy(), boxes[pos:, 3].to("cpu").numpy())

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            '''
            yy1 = torch.max(boxes[i, 1], boxes[pos:, 1])
            xx1 = torch.max(boxes[i, 0], boxes[pos:, 0])
            yy2 = torch.min(boxes[i, 3], boxes[pos:, 3])
            xx2 = torch.min(boxes[i, 2], boxes[pos:, 2])

            w = torch.max(torch.tensor(0.0).cuda(), xx2 - xx1 + 1)
            h = torch.max(torch.tensor(0.0).cuda(), yy2 - yy1 + 1)

            inter = w * h
            ovr = torch.div(inter, (areas[i] + areas[pos:] - inter))

            # Gaussian decay
            weight = torch.exp(-(ovr * ovr) / sigma)
            scores[pos:] = weight * scores[pos:]

        # select the boxes and keep the corresponding indexes
        keep = boxes[:, 4][scores > thr].cpu().numpy().tolist()
        # keep = boxes[:, 4][scores > thr].int()

        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        # boxes_for_nms = boxes
        # ---
        # boxes_for_nms = torch.cat((boxes_for_nms, boxes_anchor), dim=0)
        # scores = torch.cat((scores, score_anchor), dim=0)
        # keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        keep = self.box_nms_diou(boxes_for_nms, scores, iou_threshold)
        # keep = self.box_softnms(boxes_for_nms, scores, 0.001, 0.5)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
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

    def _reshape_cat_out2(self, inputs, strides):
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
            pred = pred * stride
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, mode="training", config=None):
        super().__init__()
        if config is None:
            config = DefaultConfig
        self.mode = mode
        self.fcos_body = FCOS(config=config)
        self.encoder = DataEncoder()
        if mode == "training":
            self.target_layer = GenTargets(strides=config.strides, limit_range=config.limit_range)
            self.loss_layer = LOSS()
            self.criterion = LOSS_anchor()
        elif mode == "inference":
            self.detection_head = DetectHead(config.score_threshold, config.nms_iou_threshold,
                                             config.max_detection_boxes_num, config.strides, config)
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        '''
        inputs 
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''
        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes, loc_targets, cls_targets = inputs
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer([out['fcos'], batch_boxes, batch_classes])  # 目标
            losses = self.loss_layer([out['fcos'], targets])
            loss_anchor = self.criterion(out['loc_preds'], loc_targets, out['cls_preds'], cls_targets)
            loss = 0.5 * loss_anchor[-1] + losses[-1]
            return loss_anchor, losses, loss
        elif self.mode == "inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            # boxes_anchor, labels_anchor, score_anchor = self.encoder.decode(out['loc_preds'].data.squeeze(), out['cls_preds'].data.squeeze(), (512, 512))
            boxes_anchor, labels_anchor, score_anchor = self.encoder.decode(out['loc_preds'], out['cls_preds'],
                                                                            (512, 512), batch_imgs.shape[0])
            nms_anchor = [boxes_anchor, labels_anchor, score_anchor]
            scores, classes, boxes = self.detection_head(out['fcos'], nms_anchor)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes
