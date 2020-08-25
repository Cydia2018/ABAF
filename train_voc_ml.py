'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-09-26
@Email: xxxmy@foxmail.com
'''
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

from model.fcos import FCOSDetector
import torch

import torchvision.transforms as transforms
# from dataloader.VOC_dataset import VOCDataset
from dataloader.dataset import Dataset
import math, time
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

model = FCOSDetector(mode="training")
# model = torch.nn.DataParallel(model.cuda(), device_ids=range(torch.cuda.device_count()))
model = model.cuda()
# model=FCOSDetector(mode="training")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [28, 48], gamma=0.1)

BATCH_SIZE = 16
EPOCHS = 60
WARMPUP_STEPS_RATIO = 0.12

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

# 2012_train 2007_val
cfg = {'images_root': '/home', 'train_path': '/mnt/hdd1/benkebishe01/data/train.txt',
       'test_path': '/mnt/hdd1/benkebishe01/data/val.txt',
       'img_size': 512}

train_dataset = Dataset(cfg['images_root'], cfg['train_path'], img_size=cfg['img_size'], transform=transform,
                        train=True)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    Dataset(cfg['images_root'], cfg['test_path'], img_size=cfg['img_size'],
            transform=transform, train=False),
    batch_size=4,
    shuffle=False, )

steps_per_epoch = len(train_dataset) // BATCH_SIZE
TOTAL_STEPS = steps_per_epoch * EPOCHS
WARMPUP_STEPS = TOTAL_STEPS * WARMPUP_STEPS_RATIO

# global GLOBAL_STEPS
GLOBAL_STEPS = 1
LR_INIT = 5e-5
LR_END = 1e-6
draw = True

if draw:
    writer = SummaryWriter(comment='9_nocnt_nonorm')


def lr_func():
    if GLOBAL_STEPS < WARMPUP_STEPS:
        lr = GLOBAL_STEPS / WARMPUP_STEPS * LR_INIT
    else:
        lr = LR_END + 0.5 * (LR_INIT - LR_END) * (
            (1 + math.cos((GLOBAL_STEPS - WARMPUP_STEPS) / (TOTAL_STEPS - WARMPUP_STEPS) * math.pi))
        )
    return float(lr)


model.train()

for epoch in range(EPOCHS):
    loss_ = []
    for epoch_step, (batch_imgs, batch_boxes, batch_classes, loc_target, cls_target) in enumerate(train_loader):
        # batch_imgs, batch_boxes, batch_classes, loc_target, cls_target = data
        batch_imgs = batch_imgs.cuda()
        batch_boxes = batch_boxes.cuda()
        batch_classes = batch_classes.cuda()
        loc_target = loc_target.cuda()
        cls_target = cls_target.cuda()

        # lr = lr_func()
        for param in optimizer.param_groups:
            lr = param['lr']

        # start_time = time.time()

        optimizer.zero_grad()
        loss_anchor, loss_fcos, loss = model([batch_imgs, batch_boxes, batch_classes, loc_target, cls_target])
        # loss = loss.mean()
        loss.backward()
        optimizer.step()

        # end_time = time.time()
        # cost_time = int((end_time-start_time)*1000)

        print(
            "global_steps:%d epoch:%d steps:%d/%d cls_loss:%.4f reg_loss:%.4f loc_anchor:%.4f cls_anchor:%.4f" % \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, loss_fcos[0], loss_fcos[1], loss_anchor[0], loss_anchor[1]))
        '''
        print(
            "global_steps:%d epoch:%d steps:%d/%d loss_fcos:%.4f loss_anchor:%.4f cnt_loss:%.4f"% \
            (GLOBAL_STEPS, epoch + 1, epoch_step + 1, steps_per_epoch, loss_fcos[0]+loss_fcos[2], loss_anchor[0]+loss_anchor[1], loss_fcos[1]))
        '''
        if draw:
            writer.add_scalar("loss/loss", loss, global_step=GLOBAL_STEPS)
            writer.add_scalar("loss/cls_loss", loss_fcos[0], global_step=GLOBAL_STEPS)
            # writer.add_scalar("loss/cnt_loss", loss_fcos[1], global_step=GLOBAL_STEPS)
            writer.add_scalar("loss/reg_loss", loss_fcos[1], global_step=GLOBAL_STEPS)
            writer.add_scalar("loss/loc_anchor", loss_anchor[0], global_step=GLOBAL_STEPS)
            writer.add_scalar("loss/cls_anchor", loss_anchor[1], global_step=GLOBAL_STEPS)
            writer.add_scalar("lr", lr, global_step=GLOBAL_STEPS)

        GLOBAL_STEPS += 1
        loss_.append(loss)

    loss_avg = torch.mean(torch.stack(loss_))
    torch.save(model.state_dict(), "/mnt/hdd1/benkebishe01/FCOS/diou/9_nocnt_nonorm/voc_epoch%d_loss%.4f.pth" % (epoch + 1, loss_avg.item()))
    scheduler.step()
