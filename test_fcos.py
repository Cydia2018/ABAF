'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''

import cv2
from model.fcos import FCOSDetector
import torch
from torchvision import transforms
import numpy as np
# from dataloader.VOC_dataset import VOCDataset
# from dataloader.COCO_dataset import COCODataset
import time
from PIL import Image

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

'''
VOC_CLASSES = (  # always index 0
    '__background__',
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
'''

VOC_CLASSES = (  # always index 0
    '__background__',
    'Glass insulator', 'Composite insulator', 'Vibration damper', 'Grading ring', 'Shielding ring',
    'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')


color_list = np.array(
    [
        0.000, 0.000, 0.000,
        1.000, 1.000, 1.000,    # white
        0.850, 0.325, 0.098,    # dark blue
        0.929, 0.694, 0.125,    # blue
        0.494, 0.184, 0.556,    # pink
        0.466, 0.674, 0.188,    # green
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000
    ]
).astype(np.float32)
color_list = color_list.reshape((-1, 3)) * 255

def preprocess_img(img, img_size=416):
    # img = np.array(img)  # h w
    img = img[..., :3]
    # pdb.set_trace()
    h, w, _ = img.shape
    dim_diff = np.abs(h - w)
    # Upper (left) and lower (right) padding
    pad1, pad2 = dim_diff // 2, dim_diff // 2
    # Determine padding
    pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
    input_img = np.pad(img, pad, 'constant', constant_values=127.5)
    # Resize and normalize
    input_img = cv2.resize(input_img, (img_size, img_size), interpolation=cv2.INTER_CUBIC)
    # Channels-first

    return input_img


def convertSyncBNtoBN(module):
    module_output = module
    if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
        module_output = torch.nn.BatchNorm2d(module.num_features,
                                             module.eps, module.momentum,
                                             module.affine,
                                             module.track_running_stats)
        if module.affine:
            module_output.weight.data = module.weight.data.clone().detach()
            module_output.bias.data = module.bias.data.clone().detach()
        module_output.running_mean = module.running_mean
        module_output.running_var = module.running_var
    for name, child in module.named_children():
        module_output.add_module(name, convertSyncBNtoBN(child))
    del module
    return module_output


if __name__ == "__main__":
    class Config():
        # backbone
        pretrained = False
        freeze_stage_1 = True
        freeze_bn = True

        # fpn
        fpn_out_channels = 256
        use_p5 = True

        # head
        class_num = 20
        use_GN_head = True
        prior = 0.01
        add_centerness = True
        cnt_on_reg = True

        # training
        strides = [8, 16, 32, 64, 128]
        limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

        # inference
        score_threshold = 0.05
        nms_iou_threshold = 0.5
        max_detection_boxes_num = 500


    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    model = FCOSDetector(mode="inference", config=Config)
    # model = torch.nn.DataParallel(model)
    # model=torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # print("INFO===>success convert BN to SyncBN")
    # model.load_state_dict(torch.load("/mnt/hdd1/benkebishe01/FCOS/fcos_val/voc2012_epoch80_loss1.1556.pth"))
    # model.load_state_dict(torch.load("/mnt/hdd1/benkebishe01/FCOS/diou/new_voc3/voc_epoch25_loss1.1657.pth"))
    model.load_state_dict(torch.load("/mnt/hdd1/benkebishe01/dianwang/five/new1.0/voc_epoch29_loss0.3873.pth"))
    # retinanet_kmean_ml/voc_epoch20_loss0.1154.pth
    # retinanet_ml_new/voc_epoch24_loss0.1074.pth
    # model.load_state_dict(torch.load("/mnt/hdd1/benkebishe01/FCOS/fcos_without_sample/voc2012_epoch74_loss0.9278.pth", map_location=torch.device('cpu')))
    # model=convertSyncBNtoBN(model)
    # print("INFO===>success convert SyncBN to BN")
    model = model.to(device).eval()
    print("===>success loading model")

    import os

    root = "./images/test_new/"
    names = os.listdir(root)
    for name in names:
        img_in = cv2.imread(root + name)
        img_pad = preprocess_img(img_in, 512)
        # img_ = Image.fromarray(img_pad.copy())
        # img_rgb = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
        img_tensor = transform(img_pad)
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            scores, classes, boxes = model(img_tensor.unsqueeze_(dim=0))

        boxes = boxes[0].cpu().numpy().tolist()
        classes = classes[0].cpu().numpy().tolist()
        scores = scores[0].cpu().numpy().tolist()

        for i, box in enumerate(boxes):
            pt1 = (int(box[0]), int(box[1]))
            pt2 = (int(box[2]), int(box[3]))
            # print(classes[i])
            cat = int(classes[i])
            c = color_list[cat].tolist()
            txt = '{}{:.4f}'.format(VOC_CLASSES[cat], scores[i])
            print(txt)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cat_size = cv2.getTextSize(txt, font, 0.4, 1)[0]
            img_pad = cv2.rectangle(img_pad, pt1, pt2, c, 1)
            img_pad = cv2.rectangle(img_pad,
                                 (int(box[0]), int(box[1]) - cat_size[1] - 0),
                                 (int(box[0]) + cat_size[0], int(box[1]) - 0), c, 1)
            img_pad = cv2.putText(img_pad, txt, (int(box[0]), int(box[1]) - 2),
                               font, 0.4, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)

        cv2.imwrite("./out_dianwang/" + name, img_pad)

        '''
        img_pad = cv2.rectangle(img_pad, pt1, pt2, (0, 255, 0), 2)
        img_pad = cv2.putText(img_pad, "%s %.3f" % (VOC_CLASSES[int(classes[i])], scores[i]),
                                  (int(box[0]), int(box[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 200, 20], 2)
        cv2.imwrite("./out_fcos/" + name, img_pad)

        bbox = np.array(bbox, dtype=np.int32)
        cat = int(cat)
        c = color_list[cat].tolist()
        txt = '{}{:.2f}'.format(VOC_CLASSES[cat], conf)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cat_size = cv2.getTextSize(txt, font, 0.5, 2)[0]
        imgs = cv2.rectangle(
            imgs, (bbox[0], bbox[1]), (bbox[2], bbox[3]), c, 2)
        if show_txt:
            imgs = cv2.rectangle(imgs,
                                 (bbox[0], bbox[1] - cat_size[1] - 2),
                                 (bbox[0] + cat_size[0], bbox[1] - 2), c, 2)
            imgs = cv2.putText(imgs, txt, (bbox[0], bbox[1] - 2),
                               font, 0.5, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
        '''




