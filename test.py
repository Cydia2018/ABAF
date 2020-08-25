#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 16:37:20 2019

@author: wei
"""
import os
import sys

sys.path.append(os.getcwd() + '/models')
sys.path.append(os.getcwd() + '/datasets')
import cv2
import time
import torch
import random
import pprint
import datetime
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from PIL import Image, ImageDraw
# from config import config
from model.fcos import FCOSDetector
from torch.utils.data import DataLoader
from dataloader.dataset import ImageFolder

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--test_path', type=str, default=r'./images/input', help='size of each image dimension')
opt = parser.parse_args()


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cuda')

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
    cnt_on_reg = False

    # training
    strides = [8, 16, 32, 64, 128]
    limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]

    # inference
    score_threshold = 0.2
    nms_iou_threshold = 0.5
    max_detection_boxes_num = 150

    CLASSES_NAME = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )



model = FCOSDetector(mode="inference", config=Config)
# model = torch.nn.DataParallel(model)
ckpt = torch.load('/mnt/hdd1/benkebishe01/FCOS/fcos_fusion/voc2012_512x512_epoch71_loss0.7739.pth')
# ckpt = torch.load('/mnt/hdd1/benkebishe01/fcos_anchor/voc2012_512x512_epoch68_loss0.7201.pth')

model.load_state_dict(ckpt)
model.to(device).eval()
print('loading weights successfully...')

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

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

root = "./images/input/"
names = os.listdir(root)
for i, name in enumerate(names):
    img_in = cv2.imread(root+name)
    img_pad = preprocess_img(img_in, 512)
    # img_ = Image.fromarray(img_pad.copy())
    img_tensor = transform(img_pad)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        _, _ , _ , boxes, labels, scores = model(img_tensor.unsqueeze_(dim=0))

    boxes = boxes.tolist()
    labels = labels.tolist()
    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        img_pad = cv2.rectangle(img_pad, pt1, pt2, (0, 255, 0), 4)
        img_pad = cv2.putText(img_pad, "%s %.3f" % (Config.CLASSES_NAME[int(labels[i]) + 1], scores[i]),
                              (int(box[0]), int(box[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 200, 20], 2)
    cv2.imwrite("./out_anchor/" + name, img_pad)

    '''
    img_rgb = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    draw = ImageDraw.Draw(img)
    for box in boxes:
        draw.rectangle(list(box), outline='red')
    img.save("./out_anchor/" + str(i) + '.jpg')
    '''

'''
dataset = ImageFolder(opt.test_path, 512, transform)
dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=False)

imgs_path = []
imgs_detection = []
prev_time = time.time()
print('\nPerforming object detection: %d samples...' % len(dataset))
for b, input_img in enumerate(dataloader):
    # import pdb
    # pdb.set_trace()
    input_img = input_img.numpy()
    img_tensor = Image.fromarray(input_img)
    img_tensor = transform(img_tensor)
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        _, _ , _ , boxes, labels = model(img_tensor)

    boxes = boxes.cpu().numpy().tolist()
    # classes = classes[0].cpu().numpy().tolist()
    # scores = scores[0].cpu().numpy().tolist()

    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        img_pad = cv2.rectangle(input_img, pt1, pt2, (0, 255, 0))
        # img_pad = cv2.putText(img_pad, "%s %.3f" % (VOCDataset.CLASSES_NAME[int(classes[i])], scores[i]),
        #                      (int(box[0]), int(box[1]) + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 200, 20], 2)
    cv2.imwrite("./out_images/" + str(b) + '.jpg', img_pad)
'''















