import math
import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """
    convolution + batch_norm + ReLU
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding = 0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)

class DownSample(nn.Module):
    """
    down sampling by max_pool
    """
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UpSample, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2, msg):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1])

        x = self.conv(x)
        return x
    
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class Unet_wm_embedder(nn.Module):
    def __init__(self, Unet_configs, n_channels, n_classes, bilinear=True):
        super(Unet_wm_embedder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 512)
        self.down_4 = DownSample(512, 1024)

        self.up_1 = UpSample(1024, 512, bilinear)
        self.up_2 = UpSample(1024, 512, bilinear)
        self.up_3 = UpSample(1024, 512, bilinear)
        self.up_4 = UpSample(1024, 512, bilinear)

        self.outc = OutConv(64, n_classes)
        self.msg_diffusion_layer = 
        

    def forward(self, spect, msg):
        # input spect shape:[batch_size(1), 1, 513, x]
        # input_msg shape:[batch_size(1), 1, msg_length]
        msg = self.msg_diffusion_layer(msg)
        x1 = self.inc(spect)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        logits = self.outc(x)

        return logits


class Unet_wm_extractor(nn.Module):
    def __init__(self, Unet_wm_extractor_configs, input_channels):
        super(Unet_wm_extractor, self).__init__()
    
    def forward(self, spect):
        wm = spect

        return wm
        