import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import functools

VOC_CLASSES = (
    '__background__', 'BoLiJueYuanZi', 'FuHeJueYuanZi', 'FangZhenChui', 'JunYaHuan', 'PingBiHuan')
# VOC_CLASSES = ('__background__', 'insulator_good', 'insulator_bad')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

def sort_by_score(pred_boxes, pred_labels, pred_scores):
    score_seq = [(-score).argsort() for index, score in enumerate(pred_scores)]
    pred_boxes = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, score_seq)]
    pred_labels = [sample_boxes[mask] for sample_boxes, mask in zip(pred_labels, score_seq)]
    pred_scores = [sample_boxes[mask] for sample_boxes, mask in zip(pred_scores, score_seq)]
    return pred_boxes, pred_labels, pred_scores

def iou_2d(cubes_a, cubes_b):
    """
    numpy 计算IoU
    :param cubes_a: [N,(x1,y1,x2,y2)]
    :param cubes_b: [M,(x1,y1,x2,y2)]
    :return:  IoU [N,M]
    """
    # 扩维
    cubes_a = np.expand_dims(cubes_a, axis=1)  # [N,1,4]
    cubes_b = np.expand_dims(cubes_b, axis=0)  # [1,M,4]

    # 分别计算高度和宽度的交集
    overlap = np.maximum(0.0,
                         np.minimum(cubes_a[..., 2:], cubes_b[..., 2:]) -
                         np.maximum(cubes_a[..., :2], cubes_b[..., :2]))  # [N,M,(w,h)]

    # 交集
    overlap = np.prod(overlap, axis=-1)  # [N,M]

    # 计算面积
    area_a = np.prod(cubes_a[..., 2:] - cubes_a[..., :2], axis=-1)
    area_b = np.prod(cubes_b[..., 2:] - cubes_b[..., :2], axis=-1)

    # 交并比
    iou = overlap / (area_a + area_b - overlap)
    return iou

def _compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def eval_ap_2d(gt_boxes, gt_labels, pred_boxes, pred_labels, pred_scores, iou_thread, num_cls):
    """
    :param gt_boxes: list of 2d array,shape[(a,(x1,y1,x2,y2)),(b,(x1,y1,x2,y2))...]
    :param gt_labels: list of 1d array,shape[(a),(b)...],value is sparse label index
    :param pred_boxes: list of 2d array, shape[(m,(x1,y1,x2,y2)),(n,(x1,y1,x2,y2))...]
    :param pred_labels: list of 1d array,shape[(m),(n)...],value is sparse label index
    :param pred_scores: list of 1d array,shape[(m),(n)...]
    :param iou_thread: eg. 0.5
    :param num_cls: eg. 4, total number of class including background which is equal to 0
    :return: a dict containing average precision for each cls
    """
    all_ap = {}
    for label in range(num_cls)[1:]:
        # get samples with specific label
        true_label_loc = [sample_labels == label for sample_labels in gt_labels]
        gt_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(gt_boxes, true_label_loc)]

        pred_label_loc = [sample_labels == label for sample_labels in pred_labels]
        bbox_single_cls = [sample_boxes[mask] for sample_boxes, mask in zip(pred_boxes, pred_label_loc)]
        scores_single_cls = [sample_scores[mask] for sample_scores, mask in zip(pred_scores, pred_label_loc)]

        fp = np.zeros((0,))
        tp = np.zeros((0,))
        scores = np.zeros((0,))
        total_gts = 0
        # loop for each sample
        for sample_gts, sample_pred_box, sample_scores in zip(gt_single_cls, bbox_single_cls, scores_single_cls):
            total_gts = total_gts + len(sample_gts)
            assigned_gt = []  # one gt can only be assigned to one predicted bbox
            # loop for each predicted bbox
            for index in range(len(sample_pred_box)):
                scores = np.append(scores, sample_scores[index])
                if len(sample_gts) == 0:  # if no gts found for the predicted bbox, assign the bbox to fp
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
                    continue
                pred_box = np.expand_dims(sample_pred_box[index], axis=0)
                iou = iou_2d(sample_gts, pred_box)
                gt_for_box = np.argmax(iou, axis=0)
                max_overlap = iou[gt_for_box, 0]
                if max_overlap >= iou_thread and gt_for_box not in assigned_gt:
                    fp = np.append(fp, 0)
                    tp = np.append(tp, 1)
                    assigned_gt.append(gt_for_box)
                else:
                    fp = np.append(fp, 1)
                    tp = np.append(tp, 0)
        # sort by score
        indices = np.argsort(-scores)
        fp = fp[indices]
        tp = tp[indices]
        # compute cumulative false positives and true positives
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        # compute recall and precision
        recall = tp / total_gts
        precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
        ap = _compute_ap(recall, precision)
        all_ap[label] = ap
        # print(recall, precision)
    return all_ap


# model_root = "/mnt/hdd1/benkebishe01/dianwang/insulator"
model_root = "/mnt/hdd1/benkebishe01/dianwang/five/new1.0"


def compare(x, y):
    stat_x = os.stat(model_root + "/" + x)

    stat_y = os.stat(model_root + "/" + y)

    if stat_x.st_ctime < stat_y.st_ctime:
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0


if __name__ == "__main__":
    from model.fcos import FCOSDetector
    from dataloader.dataset import Dataset
    from tensorboardX import SummaryWriter

    images_root = '/home'
    # val_path = '/mnt/hdd1/benkebishe01/VOCdevkit_insulator/2007_val.txt'
    val_path = '/home/benkebishe01/fcos_sample/dataloader/2007_test.txt'
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_loader = torch.utils.data.DataLoader(
        Dataset(images_root, val_path, img_size=512,
                transform=transform, train=False),
        batch_size=64,
        shuffle=False)
    draw = False
    if draw:
        writer = SummaryWriter(comment='voc_mAP')

    model = FCOSDetector(mode="inference")
    # model.load_state_dict(torch.load("/mnt/hdd1/benkebishe01/FCOS/80epoch/fcos_voc_3/voc_epoch80_loss0.8893.pth"))
    model = model.to(device).eval()
    print("===>success loading model")

    names = os.listdir(model_root)
    names = sorted(names, key=functools.cmp_to_key(compare))

    for name in names:
        # print(name)
        model.load_state_dict(torch.load(os.path.join(model_root + "/" + name)))

        gt_boxes = []
        gt_classes = []
        pred_boxes = []
        pred_classes = []
        pred_scores = []
        num = 0
        for img, boxes, classes, _, _ in val_loader:
            with torch.no_grad():
                # out = model(img.cuda())
                out = model(img.to(device))
                # xxx = out[2][0]
                # yyy = out[1][0]
            for i in range(boxes.shape[0]):
                pred_boxes.append(out[2][i].cpu().numpy())
                pred_classes.append(out[1][i].cpu().numpy())
                pred_scores.append(out[0][i].cpu().numpy())
                gt_boxes.append(boxes[i].numpy())
                gt_classes.append(classes[i].numpy())
                num += 1
            # print(num, end=' ')

        # print(gt_boxes[0],gt_classes[0])
        # print(pred_boxes[0],pred_classes[0],pred_scores[0])

        pred_boxes, pred_classes, pred_scores = sort_by_score(pred_boxes, pred_classes, pred_scores)
        all_AP = eval_ap_2d(gt_boxes, gt_classes, pred_boxes, pred_classes, pred_scores, 0.5, len(VOC_CLASSES))  # ???
        print("\nall classes AP=====>\n", all_AP)
        mAP = 0.
        for class_id, class_mAP in all_AP.items():
            mAP += float(class_mAP)
        mAP /= (len(VOC_CLASSES) - 1)
        print("mAP=====>%.8f\n" % mAP)

        if draw:
            writer.add_scalar("mAP/mAP", mAP)
        # with open('dianwang/insulator.txt', 'a') as f:
        #     # f.write(name + ":" + str(mAP) + '\n')
        #     f.write(str(mAP) + '\n')
