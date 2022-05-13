import torch
import torch.nn as nn
from functools import partial
import random

class ConvBlock(nn.Module):

    def __init__(self, inplanes,med_planes, stride=1, res_conv=False, act_layer=nn.ReLU, groups=1,
                drop_block=None, drop_path=None):
        super(ConvBlock, self).__init__()
        """
        inplanes:cnn的通道
        med_planes:transformer的通道
        inplanes:映射回来后的通道
        """
        
        self.bn = nn.BatchNorm2d(med_planes)
        self.act = act_layer(inplace=True)

        self.conv1 = nn.Conv2d(inplanes, med_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(med_planes)
        self.act1 = act_layer(inplace=True)

        self.conv2 = nn.Conv2d(med_planes, med_planes, kernel_size=3, stride=stride, groups=groups, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(med_planes)
        self.act2 = act_layer(inplace=True)

        self.conv3 = nn.Conv2d(med_planes, inplanes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(inplanes)
        self.act3 = act_layer(inplace=True)

        
    def zero_init_last_bn(self):
        nn.init.zeros_(self.bn3.weight)

    def forward(self, x, x_t, return_x_2=False):
        p = random.random()
        if p < 0.05 and self.training:
            x_t = torch.zeros_like(x_t)
        # if p >= 0 and not self.training:
        #     x_t = torch.zeros_like(x_t)

        x_t = self.bn(x_t)
        x_t = self.act(x_t)
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)

        x = self.conv2(x + x_t)
        x = self.bn2(x)
        x2 = self.act2(x)

        x = self.conv3(x2)
        x = self.bn3(x)
        x += residual
        x = self.act3(x)

        if return_x_2:
            return x, x2
        else:
            return x