import xml.etree.ElementTree as ET
import pickle
import os

from os import listdir, getcwd
from os.path import join

# sets=[('2012', 'train'), ('2012', 'val'), ('2007', 'train'), ('2007', 'val'), ('2007', 'test')]
# sets=[('2007', 'trainval'), ('2007', 'test')]
sets=[('2007', 'train'), ('2007', 'val')]
# classes = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# classes = ["BoLiJueYuanZi", "FuHeJueYuanZi", "FangZhenChui", "JunYaHuan", "PingBiHuan"]
classes = ["insulator_good", "insulator_bad"]


def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def convert_annotation(year, image_id, file):
    in_file = open('/mnt/hdd1/benkebishe01/VOCdevkit_insulator/VOC%s/Annotations/%s.xml'%(year, image_id))
    # out_file = open('VOCdevkit/VOC%s/labels/%s.txt'%(year, image_id), 'w')
    out_file = file
    tree=ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text
        cls = obj.find('name').text
        if cls not in classes:
            continue
        # if cls not in classes or int(difficult) == 1:
        #     continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bb = convert((w,h), b)
        for a in b:
            out_file.write(str(a)+" ")
        out_file.write(str(cls_id)+" ")

wd = getcwd()

for year, image_set in sets:
    # if not os.path.exists('VOCdevkit/VOC%s/labels/'%(year)):
    #     os.makedirs('VOCdevkit/VOC%s/labels/'%(year))
    print(os.path.abspath('/mnt/hdd1/benkebishe01/VOCdevkit_insulator/VOC%s/ImageSets/Main'))
    image_ids = open('/mnt/hdd1/benkebishe01/VOCdevkit_insulator/VOC%s/ImageSets/Main/%s.txt'%(year, image_set)).read().strip().split()
    list_file = open('/mnt/hdd1/benkebishe01/VOCdevkit_insulator/%s_%s.txt'%(year, image_set), 'w')
    for image_id in image_ids:
        list_file.write('/mnt/hdd1/benkebishe01/VOCdevkit_insulator/VOC%s/JPEGImages/%s.jpg '%(year, image_id))
        convert_annotation(year, image_id, list_file)
        list_file.write('\n')
    list_file.close()

