import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import functools
from model.fcos import FCOSDetector
import torch
# gpus = [0, 1]
# torch.device('cuda:2')

import torchvision.transforms as transforms
# from dataloader.VOC_dataset import VOCDataset
from dataloader.dataset import Dataset
import math, time
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter


model = FCOSDetector(mode="training")
model = model.cuda().eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 2012_train 2007_val
cfg = {'images_root': '/home', 'train_path': '/mnt/hdd1/benkebishe01/data/train.txt', 'test_path': '/mnt/hdd1/benkebishe01/data/val.txt',
      'img_size': 512}

test_loader = torch.utils.data.DataLoader(
        Dataset(cfg['images_root'], cfg['test_path'],img_size=cfg['img_size'],
                    transform=transform, train=False),
                    batch_size=4,
                    shuffle=False)

draw = False

if draw:
    writer = SummaryWriter(comment='test_voc_one')

# model_root = "/mnt/hdd1/benkebishe01/FCOS/fcos_giou_one_new"
model_root = "/mnt/hdd1/benkebishe01/FCOS/fcos_voc_one"


def compare(x, y):
    stat_x = os.stat(model_root + "/" + x)

    stat_y = os.stat(model_root + "/" + y)

    if stat_x.st_ctime < stat_y.st_ctime:
        return -1
    elif stat_x.st_ctime > stat_y.st_ctime:
        return 1
    else:
        return 0

names = os.listdir(model_root)
names = sorted(names, key=functools.cmp_to_key(compare))
for name in names:
    print(name)
    model.load_state_dict(torch.load(os.path.join(model_root+"/"+name)))
    loss_ = []
    for epoch_step, (batch_imgs, batch_boxes, batch_classes, loc_target, cls_target) in enumerate(test_loader):
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()
        loc_target = loc_target.cuda()
        cls_target = cls_target.cuda()

        with torch.no_grad():
            loss_anchor, loss_fcos, loss = model([batch_imgs, batch_boxes, batch_classes, loc_target, cls_target])

        print("steps:%d cls_loss:%.4f cnt_loss:%.4f reg_loss:%.4f loc_anchor:%.4f cls_anchor:%.4f" % \
            (epoch_step + 1, loss_fcos[0], loss_fcos[1], loss_fcos[2], loss_anchor[0], loss_anchor[1]))

        loss_.append(loss)
    loss_avg = torch.mean(torch.stack(loss_))

    if draw:
        writer.add_scalar("loss/loss", loss_avg)
    with open('test_loss_one.txt', 'a') as f:
        f.write(str(loss_avg.item())+'\n')
