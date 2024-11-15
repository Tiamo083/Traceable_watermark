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
        
        x = torch.cat([x2, x1, msg], dim = 1)
        x = self.conv(x)
        return x


class MsgUpSample(nn.Module):
    def __init__(self, out_channels):
        super(MsgUpSample, self).__init__()
        self.MsgUpNet = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, msg):
        return self.MsgUpNet(msg)


class MsgDownSample(nn.Module):
    def __init__(self, in_channels):
        super(MsgDownSample, self).__init__()
        self.MsgDownNet = nn.Sequential(
            nn.ConvTranspose2d(in_channels, 1, stride = 1, kernel_size=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1, 1, stride=1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, msg):
        return self.MsgDownNet(msg)


class DDIMModel(nn.Module):
    def __init__(self, dim, num_layers, hidden_dim, beta_min, beta_max, beta_steps):
        super(DDIMModel, self).__init__()
        self.dim = dim  # 特征维度（不包括batch和channel维度）
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.beta_steps = beta_steps
        self.betas = self._create_betas()

        self.alphas_cumprod = (1 - self.betas).cumprod(dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU()
            ) for _ in range(num_layers - 2)],
            nn.Linear(hidden_dim, dim)
        )
    
    def _create_betas(self):
        betas = torch.linspace(self.beta_min, self.beta_max, self.beta_steps).float()
        return betas
    
    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散过程（仅用于训练时生成噪声数据）
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        x_noise = self.sqrt_alphas_cumprod[t] * x_start + self.sqrt_one_minus_alphas_cumprod[t] * noise
        return x_noise, noise
    
    def p_sample(self, x_t, t, noise=None, num_steps=None):
        """
        逆向去噪过程（用于生成样本）
        """
        if num_steps is None:
            num_steps = t + 1
        
        noise_preds = []
        x_curr = x_t

        for i in range(num_steps - 1, -1, -1):
            t_prev = max(0, i - 1)
            x_curr_noiseless = (x_curr - self.sqrt_one_minus_alphas_cumprod[t_prev + 1] * noise) / self.sqrt_alphas_cumprod[t_prev + 1]
            noise_pred = self.net(x_curr_noiseless)
            
            if noise is None:
                noise = torch.randn_like(x_curr)
            
            # DDIM specific reparameterization
            x_prev = (x_curr_noiseless * self.sqrt_alphas_cumprod[t_prev] + 
                      (1 - self.alphas_cumprod[t_prev]) * (noise_pred + noise) / self.sqrt_one_minus_alphas_cumprod[t_prev])
            
            noise_preds.append(noise_pred)
            x_curr = x_prev
        
        return x_curr, noise_preds[::-1]
    
    def forward(self, x_t, t, noise=None):
        """
        在训练时，预测噪声
        """
        x_curr_noiseless = (x_t - self.sqrt_one_minus_alphas_cumprod[t] * noise) / self.sqrt_alphas_cumprod[t]
        noise_pred = self.net(x_curr_noiseless)
        return noise_pred


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)


class Unet_wm_embedder(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
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
        self.up_2 = UpSample(512, 256, bilinear)
        self.up_3 = UpSample(256, 128, bilinear)
        self.up_4 = UpSample(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

        # self.msg_diffusion_layer = MsgDiffusion()
        self.msg_up_1 = MsgUpSample(512)
        self.msg_up_2 = MsgUpSample(256)
        self.msg_up_3 = MsgUpSample(128)
        self.msg_up_4 = MsgUpSample(64)
        

    def forward(self, spect, msg):
        # input spect shape:[batch_size(1), 1, 513, x]
        # input_msg shape:[batch_size(1), 1, msg_length]
        # msg shape: [batch_size(1), 1, 513]

        x1 = self.inc(spect)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        msg1 = self.msg_up_1(msg)
        msg2 = self.msg_up_2(msg)
        msg3 = self.msg_up_3(msg)
        msg4 = self.msg_up_4(msg)

        x = self.up_1(x5, x4, msg4)
        x = self.up_2(x, x3, msg3)
        x = self.up_3(x, x2, msg2)
        x = self.up_4(x, x1, msg1)
        encoded_spect = self.outc(x)

        return encoded_spect


class Unet_wm_extractor(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(Unet_wm_extractor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 512)
        self.down_4 = DownSample(512, 1024)

        self.up_1 = UpSample(1024, 512, bilinear)
        self.up_2 = UpSample(512, 256, bilinear)
        self.up_3 = UpSample(256, 128, bilinear)
        self.up_4 = UpSample(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

        self.msg_down = MsgDownSample(n_classes)
        # self.reversed_diffusion = ReversedDiffusion()

    def forward(self, spect):
        x1 = self.inc(spect)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        x = self.up_1(x5, x4)
        x = self.up_2(x, x3)
        x = self.up_3(x, x2)
        x = self.up_4(x, x1)
        decoded_spect = self.outc(x)

        msg = self.msg_down(decoded_spect)
        # msg = self.reversed_diffusion(msg)

        return msg
        