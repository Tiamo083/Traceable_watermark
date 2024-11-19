import math
from turtle import forward
from networkx import reverse
import torch
import torch.nn as nn
import numpy as np
import pdb
from torch.nn import functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2, msg):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        msg = msg.unsqueeze(3).repeat(1, 1, 1, x1.shape[3])
        x = torch.cat([x2, x1, msg], dim = 1)
        x = self.conv(x)
        return x

class MsgUpSample(nn.Module):
    def __init__(self, msg_length, out_features):
        super(MsgUpSample, self).__init__()
        self.msg_length = msg_length
        self.out_features = out_features

        self.MsgUpLinear = torch.nn.Sequential(
                torch.nn.Linear(in_features=self.msg_length, out_features=max(44, self.out_features // 2)),
                nn.BatchNorm1d(1),
                nn.ReLU(inplace=True),
                torch.nn.Linear(in_features=max(44, self.out_features // 2), out_features=self.out_features),
                nn.BatchNorm1d(1),
                nn.ReLU(inplace=True)
        )
    
    def forward(self, msg):
        # msg shape [1, 1, out_features]
        # msg_diffusion_modules
        msg = self.MsgUpLinear(msg)
        return msg
        

class MsgDownSample(nn.Module):
    def __init__(self, msg_length, in_features):
        super(MsgDownSample, self).__init__()
        self.MsgDownNet = nn.Sequential(
            torch.nn.Linear(in_features=in_features, out_features=in_features // 2),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True),
            torch.nn.Linear(in_features=in_features // 2, out_features=msg_length),
            nn.BatchNorm1d(1),
            nn.ReLU(inplace=True)
        )

    def forward(self, msg):
        msg = self.MsgDownNet(msg)
        return msg

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)

class UNetModelMsg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetModelMsg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(self.in_channels, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        
        self.up_1 = UpSample(256, 128)
        self.up_2 = UpSample(128, 64)

        self.outc = OutConv(64, self.out_channels)
    
    def forward(self, msg):
        msg1 = self.inc(msg)
        msg2 = self.down_1(msg1)
        msg3 = self.down_2(msg2)

        msg = self.up_1(msg3, msg2)
        msg = self.up_2(msg, msg1)
        feature_mixed_msg = self.outc(msg)
        return feature_mixed_msg

class DiffusionModel(nn.Module):
    def __init__(self, input_dim, num_time_steps, beta_min=1e-4, beta_max=2e-2):
        super(DiffusionModel, self).__init__()
        self.input_dim = input_dim
        self.num_time_steps = num_time_steps
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.UNet = UNetModelMsg(1, 1)
        self.betas = torch.linspace(beta_min, beta_max, num_time_steps, dtype=torch.float32).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_cumprod = torch.cumprod(self.betas, dim=0)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.one_minus_alphas_cumprod, dim=0)
    
    def forward(self, x_t, t):
        x_t_t = torch.cat([x_t, t.unsqueeze(-1)], dim=-1)
        
        # Encode
        mean = self.UNet(x_t_t)
        
        # For simplicity, we assume a fixed variance (beta_t) for the noise prediction
        variance = self.betas[t]
        
        return mean, variance

    def q_sample(self, x_0, t):
        """
        Sample from the forward diffusion process q(x_t|x_0)
        """
        noise = torch.randn_like(x_0)
        alpha_cumprod_t = self.alphas_cumprod[t]
        sqrt_alphas_cumprod_t = torch.sqrt(alpha_cumprod_t)
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - alpha_cumprod_t)
        
        return sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise

    def p_sample(self, x_t, t, noise=None):
        """
        Sample from the reverse diffusion process p(x_{t-1}|x_t)
        """
        if noise is None:
            noise = torch.randn_like(x_t)
        
        mean, variance = self.forward(x_t, t)
        beta_t = self.betas[t]
        
        # Compute the necessary coefficients
        sqrt_alphas_cumprod_tm1 = torch.sqrt(self.alphas_cumprod[t-1])
        sqrt_one_minus_alphas_cumprod_t = torch.sqrt(1.0 - self.alphas_cumprod[t])
        sqrt_one_minus_alphas_t = torch.sqrt(1.0 - self.alphas[t])
        
        # Reverse process mean computation
        x_tm1 = (x_t - sqrt_one_minus_alphas_cumprod_t * mean) / sqrt_alphas_cumprod_tm1
        
        # Adding noise for sampling
        x_tm1 = x_tm1 + sqrt_one_minus_alphas_t * noise
        
        return x_tm1

    def p_sample_loop(self, shape, num_time_steps, noise=None):
        """
        Generate a sample from the learned distribution
        """
        
        if noise is None:
            noise = torch.randn(shape, device=device)
        
        batch_size = shape[0]
        x_T = torch.randn(shape, device=device)  # Start from pure noise at t=T
        
        for t in reversed(range(num_time_steps)):
            x_T = self.p_sample(x_T, t, noise[:, t, :])
        
        return x_T
    
    def inference(self, msg, num_time_steps):
        msg_t = torch.randn(msg.shape).to(device)
        with torch.no_grad():
            for time in range(num_time_steps - 1, -1, -1):
                beta_t = self.betas[time]
                alpha_cumprod = torch.cumprod(1 - self.betas[:time + 1])

                if time == num_time_steps - 1:
                    msg_t_prev = msg_t
                else:
                    noise_pred = self.forward(msg_t_prev, time.to(device))
                    msg_t_prev = (msg_t_prev - torch.sqrt(1 - alpha_cumprod[time]) * noise_pred) / torch.sqrt(alpha_cumprod[time - 1])
                
                msg_t = msg_t_prev
        
        return msg_t_prev

    def loss_cal(self, msg):
        batch_size, input_dim = msg.shape[1], msg.shape[2]
        total_loss = 0.0
        for time in range(self.num_time_steps):
            t_tensor = torch.full((batch_size,), time, dtype=torch.long, device = device)
            msg_t = self.q_sample(msg, t_tensor)

            mean, variance = self.forward(msg_t, t_tensor)

            noise = (msg_t - self.sqrt_alphas_cumprod[time] * msg) / self.sqrt_one_minus_alphas_cumprod[time]
        
            # Compute the loss (mean squared error for simplicity)
            loss = ((mean - noise) ** 2).mean() + (variance - self.betas[time]).abs().mean()  # L2 loss for mean, L1 loss for variance
        
            total_loss += loss

        return total_loss / self.num_time_steps
    

class Unet_wm_embedder(nn.Module):
    def __init__(self, n_channels, n_classes, msg_length, num_time_steps, beta_min=1e-4, beta_max=2e-2):
        super(Unet_wm_embedder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_time_steps = num_time_steps

        self.inc = DoubleConv(n_channels, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 512)
        self.down_4 = DownSample(512, 1024)

        self.up_1 = UpSample(1024, 512)
        self.up_2 = UpSample(512, 256)
        self.up_3 = UpSample(256, 128)
        self.up_4 = UpSample(128, 64)

        self.outc = OutConv(64, n_classes)

        self.msg_diffusion = DiffusionModel(1, self.num_time_steps, self.beta_min, self.beta_max)
        self.msg_up_1 = MsgUpSample(msg_length, 56)
        self.msg_up_2 = MsgUpSample(msg_length, 121)
        self.msg_up_3 = MsgUpSample(msg_length, 250)
        self.msg_up_4 = MsgUpSample(msg_length, 509)
        

    def forward(self, spect, msg):
        # input spect shape:[batch_size(1), 1, 513, x]
        # input_msg shape:[batch_size(1), 1, msg_length]
        # msg shape: [batch_size(1), 1, 513]
        import pdb
        pdb.set_trace()
        x1 = self.inc(spect)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        x5 = self.down_4(x4)

        msg = self.msg_diffusion.q_sample(msg, self.num_time_steps)
        msg1 = self.msg_up_1(msg)
        msg2 = self.msg_up_2(msg)
        msg3 = self.msg_up_3(msg)
        msg4 = self.msg_up_4(msg)

        x = self.up_1(x5, x4, msg1)
        x = self.up_2(x, x3, msg2)
        x = self.up_3(x, x2, msg3)
        x = self.up_4(x, x1, msg4)
        encoded_spect = self.outc(x)

        return encoded_spect

class Unet_wm_extractor(nn.Module):
    def __init__(self, msg_length, n_channels, n_classes, reverse_diffusion_model):
        super(Unet_wm_extractor, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 512)
        self.down_4 = DownSample(512, 1024)

        self.up_1 = UpSample(1024, 512)
        self.up_2 = UpSample(512, 256)
        self.up_3 = UpSample(256, 128)
        self.up_4 = UpSample(128, 64)

        self.outc = OutConv(64, n_classes)

        self.msg_down = MsgDownSample(msg_length, 513)
        self.reverse_diffusion = reverse_diffusion_model

    def forward(self, spect, num_time_steps):
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
        msg = torch.mean(msg, dim = -1, keepdim=True).squeeze(-1)
        msg = self.reverse_diffusion.inference(msg, num_time_steps)

        return msg
        