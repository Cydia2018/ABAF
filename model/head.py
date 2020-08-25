'''
@Author: xxxmy
@Github: github.com/VectXmy
@Date: 2019-10-06
@Email: xxxmy@foxmail.com
'''

import torch.nn as nn
import torch
import math

class ScaleExp(nn.Module):
    def __init__(self,init_value=1.0):
        super(ScaleExp,self).__init__()
        self.scale=nn.Parameter(torch.tensor([init_value],dtype=torch.float32))
    def forward(self,x):
        return torch.exp(x*self.scale)

class ClsCntRegHead(nn.Module):
    def __init__(self,in_channel,class_num,GN=True,cnt_on_reg=True,prior=0.01):
        '''
        Args  
        in_channel  
        class_num  
        GN  
        prior  
        '''
        super(ClsCntRegHead,self).__init__()
        self.prior=prior
        self.class_num=class_num
        self.num_anchors = 9
        self.cnt_on_reg=cnt_on_reg
        
        cls_branch=[]
        reg_branch=[]

        for i in range(4):
            cls_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                cls_branch.append(nn.GroupNorm(32,in_channel))
            cls_branch.append(nn.ReLU(True))

            reg_branch.append(nn.Conv2d(in_channel,in_channel,kernel_size=3,padding=1,bias=True))
            if GN:
                reg_branch.append(nn.GroupNorm(32,in_channel))
            reg_branch.append(nn.ReLU(True))

        self.cls_conv=nn.Sequential(*cls_branch)
        self.reg_conv=nn.Sequential(*reg_branch)

        self.cls_logits=nn.Conv2d(in_channel,class_num,kernel_size=3,padding=1)
        self.cnt_logits=nn.Conv2d(in_channel,1,kernel_size=3,padding=1)
        self.reg_pred=nn.Conv2d(in_channel,4,kernel_size=3,padding=1)

        self.loc_head = nn.Conv2d(in_channel, self.num_anchors * 4, kernel_size=3, padding=1)
        self.cls_head = nn.Conv2d(in_channel, self.num_anchors * self.class_num, kernel_size=3, padding=1)
        
        self.apply(self.init_conv_RandomNormal)
        
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - prior) / prior))
        pi = 0.01
        nn.init.constant_(self.cls_head.bias, -math.log((1 - pi) / pi))

        self.scale_exp = nn.ModuleList([ScaleExp(1.0) for _ in range(5)])
    
    def init_conv_RandomNormal(self,module,std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, inputs, b):
        '''inputs:[P3~P7]'''
        cls_logits=[]
        cnt_logits=[]
        reg_preds=[]
        loc_preds = []
        cls_preds = []

        for index, P in enumerate(inputs):
            cls_conv_out = self.cls_conv(P)
            reg_conv_out = self.reg_conv(P)

            loc_pred = self.loc_head(reg_conv_out)
            cls_pred = self.cls_head(cls_conv_out)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, self.class_num)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

            cls_logits.append(self.cls_logits(cls_conv_out))
            # if not self.cnt_on_reg:
            #     cnt_logits.append(self.cnt_logits(cls_conv_out))
            # else:
            #     cnt_logits.append(self.cnt_logits(reg_conv_out))
            reg_preds.append(self.scale_exp[index](self.reg_pred(reg_conv_out)))
        # return cls_logits, cnt_logits, reg_preds, torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)
        return cls_logits, reg_preds, torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)


class anchor_Head(nn.Module):
    def __init__(self, class_num):
        '''
        Args
        in_channel
        class_num
        GN
        prior
        '''
        super(anchor_Head, self).__init__()
        # self.prior = prior
        self.class_num = class_num
        self.num_anchors = 9
        self.loc_head = self._make_head(self.num_anchors * 4)
        self.cls_head = self._make_head(self.num_anchors * self.class_num)

        self.apply(self.init_conv_RandomNormal)

        pi = 0.01
        nn.init.constant_(self.cls_head[-1].bias, -math.log((1 - pi) / pi))

    def init_conv_RandomNormal(self, module, std=0.01):
        if isinstance(module, nn.Conv2d):
            nn.init.normal_(module.weight, std=std)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def forward(self, inputs, b):
        loc_preds = []
        cls_preds = []
        for fm in inputs:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, 4)  # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0, 2, 3, 1).contiguous().view(b, -1, self.class_num)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)
        return torch.cat(loc_preds, 1), torch.cat(cls_preds, 1)

