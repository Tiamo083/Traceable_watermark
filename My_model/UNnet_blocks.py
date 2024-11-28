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
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
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
    
    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim = 1)
        x = self.conv(x)
        return x
    
class UNetUpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetUpSample, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        
        self.conv = DoubleConv(in_channels + 1, out_channels)
    
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
                nn.LeakyReLU(),
                torch.nn.Linear(in_features=max(44, self.out_features // 2), out_features=self.out_features),
                nn.BatchNorm1d(1),
                nn.LeakyReLU()
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
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            torch.nn.Linear(in_features=in_features // 2, out_features=msg_length),
            nn.BatchNorm2d(1),
            nn.LeakyReLU()
        )

    def forward(self, msg):
        msg = msg.transpose(2, 3)
        msg = self.MsgDownNet(msg)
        msg = msg.transpose(2, 3)
        return msg

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels, type="None"):
        super(OutConv, self).__init__()
        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if type == "MSG_Decoder":
            self.after_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
            )
        else:
            self.after_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, kernel_size=2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU()
                )
    
    def forward(self, x1, x2):
        x1 = self.pre_conv(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = self.after_conv(x1)
        return x

class MsgEmbedderOutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MsgEmbedderOutConv, self).__init__()
        self.pre_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.after_conv = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.InstanceNorm2d(out_channels),
            nn.LeakyReLU()
        )
    
    def forward(self, x1, x2):
        x1 = self.pre_conv(x1)

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = self.after_conv(x1)
        return x

class UNetModelMsg(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetModelMsg, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(self.in_channels, 64)
        self.down_1 = DownSample(64, 128)

        self.up_1 = UpSample(128, 64)

        self.outc = OutConv(64, self.out_channels)
    
    def forward(self, msg):
        msg = msg.unsqueeze(-1).repeat(1, 1, 1, msg.shape[-1])
        msg1 = self.inc(msg)
        msg2 = self.down_1(msg1)
        up_msg = self.up_1(msg2, msg1)
        feature_mixed_msg = self.outc(up_msg, msg)
        feature_mixed_msg = torch.mean(feature_mixed_msg, keepdim = True, dim = -1).squeeze(-1)
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
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(self.one_minus_alphas_cumprod)
    
    def forward(self, x_t, t):
        if torch.is_tensor(t):
            t = t.unsqueeze(-1).unsqueeze(-1).repeat(x_t.shape[0], x_t.shape[1], 1).to(device)
        else:
            t = torch.full(size=(x_t.shape[0], x_t.shape[1], 1), fill_value=t).to(device)
        x_t_t = torch.cat([x_t, t], dim=-1)
        
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

    def q_sample_loop(self, msg, num_time_nums):
        x_T = msg
        for t in range(num_time_nums):
            x_T = self.q_sample(x_T, t)
        
        return x_T

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
    
    @torch.no_grad()
    def inference(self, msg, num_time_steps):
        msg_t = torch.randn(msg.shape).to(device)
        with torch.no_grad():
            for time in range(num_time_steps - 1, -1, -1):
                beta_t = self.betas[time]
                alpha_cumprod = torch.cumprod(1 - self.betas[:time + 1], dim = 0)
                if time == num_time_steps - 1:
                    msg_t_prev = msg_t
                else:
                    noise_pred = self.forward(msg_t_prev, time)
                    msg_t_prev = (msg_t_prev - torch.sqrt(1 - alpha_cumprod[time]) * noise_pred[0]) / torch.sqrt(alpha_cumprod[time - 1])
                
                msg_t = msg_t_prev
        
        return msg_t_prev

    def loss_cal(self, msg):
        batch_size, input_dim = msg.shape[1], msg.shape[2]
        total_loss = 0.0
        for time in range(self.num_time_steps):
            t_tensor = torch.full((batch_size,), time, dtype=torch.long, device = device)
            msg_t = self.q_sample_loop(msg, time)

            mean, variance = self.forward(msg_t, t_tensor)

            noise = msg_t - msg
            # Compute the loss (mean squared error for simplicity)
            loss = ((mean - noise) ** 2).mean() + (variance - self.betas[time]).abs().mean()  # L2 loss for mean, L1 loss for variance

            total_loss = total_loss + loss

        return total_loss / self.num_time_steps
    
class InvertibleLinear(nn.Module):
    def __init__(self, input_dim):
        super(InvertibleLinear, self).__init__()
        # 初始化权重矩阵，并确保它是可逆的（例如，使用LU分解）
        # 为了简单起见，这里我们直接使用随机正交矩阵
        self.weight = nn.Parameter(torch.randn(input_dim, input_dim))
        # 强制权重矩阵为正交矩阵（即，其转置等于其逆）
        with torch.no_grad():
            # 使用QR分解来确保正交性
            q, r = torch.qr(self.weight)
            self.weight.copy_(q)  # 将正交矩阵复制回权重
        self.bias = nn.Parameter(torch.randn(input_dim))
        self.log_det_jacobian = nn.Parameter(torch.zeros(1))  # 用于记录雅可比行列式的对数
 
    def forward(self, x):
        # 前向传播：y = Wx + b
        y = torch.matmul(x, self.weight.t()) + self.bias  # 注意这里使用转置权重
        # 更新雅可比行列式的对数（对于线性变换，它是权重的行列式的对数）
        self.log_det_jacobian.data += torch.log(torch.abs(torch.det(self.weight.t())))
        return y, self.log_det_jacobian
 
    def inverse(self, y):
        # 逆变换：x = W'y - W'Wb + b (其中W'是W的逆，对于正交矩阵W'，W' = W^T)
        x = torch.matmul(y - self.bias, self.weight)  # 注意这里没有使用转置，因为W是正交的
        # 注意：逆变换不需要再次计算雅可比行列式的对数，因为它在前向传播时已经计算过了
        # 如果你需要在逆变换中访问它，你可以从外部传递它
        return x
 
class FlowBasedModel(nn.Module):
    def __init__(self, input_dim, num_flows):
        super(FlowBasedModel, self).__init__()
        self.flows = nn.ModuleList([InvertibleLinear(input_dim) for _ in range(num_flows)])
 
    def forward(self, x):
        log_det_jacobians = []
        for flow in self.flows:
            x, log_det_jacobian = flow(x)
            log_det_jacobians.append(log_det_jacobian)
        # 返回变换后的数据和所有雅可比行列式的对数之和
        return x, sum(log_det_jacobians)
 
    def inverse(self, y):
        x = y
        for flow in reversed(self.flows):
            x = flow.inverse(x)
        # 注意：逆变换通常不需要返回雅可比行列式的对数，因为它在训练生成模型时不是必需的
        return x


class MsgConv(nn.Module):
    def __init__(self, input_channels, out_channels, input_features, out_features):
        super(MsgConv, self).__init__()
        self.input_features = input_features
        self.out_features = out_features
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.MsgConvLayer = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=out_features),
            nn.Conv1d(in_channels=input_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU()
        )

    def forward(self, msg):
        return self.MsgConvLayer(msg)


class Unet_wm_embedder(nn.Module):
    def __init__(self, n_channels, n_classes, msg_length, num_time_steps, beta_min=1e-4, beta_max=2e-2):
        super(Unet_wm_embedder, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.num_time_steps = num_time_steps
        self.msg_length = msg_length

        self.inc = DoubleConv(n_channels, 64)
        self.down_1 = DownSample(64, 128)
        self.down_2 = DownSample(128, 256)
        self.down_3 = DownSample(256, 512)
        # self.down_4 = DownSample(512, 1024)

        # self.up_1 = UNetUpSample(1024, 512)
        self.up_2 = UNetUpSample(512, 256)
        self.up_3 = UNetUpSample(256, 128)
        self.up_4 = UNetUpSample(128, 64)

        self.outc = MsgEmbedderOutConv(64, n_classes)

        self.msg_diffusion = DiffusionModel(1, self.num_time_steps, self.beta_min, self.beta_max)
        self.msg_up_1 = MsgUpSample(msg_length, 56)
        self.msg_up_2 = MsgUpSample(msg_length, 121)
        self.msg_up_3 = MsgUpSample(msg_length, 250)
        self.msg_up_4 = MsgUpSample(msg_length, 509)
        
        self.msg_flow = FlowBasedModel(input_dim=msg_length, num_flows=5)
        self.msg_conv = MsgConv(input_channels=1, out_channels=1, input_features=msg_length, out_features=msg_length + 4)

    def forward(self, spect, msg):
        # input spect shape:[batch_size(1), 1, 513, x]
        # input_msg shape:[batch_size(1), 1, msg_length]
        x1 = self.inc(spect)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        # x5 = self.down_4(x4)

        # set msg to [batch_size, 1, msg_length]
        # diff_loss = self.msg_diffusion.loss_cal(msg)
        # msg = self.msg_diffusion.q_sample_loop(msg, self.num_time_steps)

        # msg, log_det_jacobian_sum = self.msg_flow(msg)
        msg = self.msg_conv(msg)
        msg1 = self.msg_up_1(msg)
        msg2 = self.msg_up_2(msg)
        msg3 = self.msg_up_3(msg)
        msg4 = self.msg_up_4(msg)

        # up_x = self.up_1(x5, x4, msg1)
        # up_x = self.up_2(up_x, x3, msg2)
        # up_x = self.up_3(up_x, x2, msg3)
        # up_x = self.up_4(up_x, x1, msg4)
        up_x = self.up_2(x4, x3, msg2)
        up_x = self.up_3(up_x, x2, msg3)
        up_x = self.up_4(up_x, x1, msg4)
        encoded_spect = self.outc(up_x, spect)

        return encoded_spect, self.msg_diffusion

class Unet_wm_extractor(nn.Module):
    def __init__(self, msg_length, n_channels, n_classes):
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

        self.outc = OutConv(64, n_classes, "MSG_Decoder")

        self.msg_down = MsgDownSample(msg_length, 513)
        self.msg_conv = MsgConv(input_channels=1, out_channels=1, input_features=msg_length, out_features=msg_length + 4)

    def forward(self, spect, num_time_steps, reverse_diffusion_model):
        x1 = self.inc(spect)
        x2 = self.down_1(x1)
        x3 = self.down_2(x2)
        x4 = self.down_3(x3)
        # x5 = self.down_4(x4)

        # x = self.up_1(x5, x4)
        # x = self.up_2(x, x3)
        # x = self.up_3(x, x2)
        # x = self.up_4(x, x1)

        up_x = self.up_2(x4, x3)
        up_x = self.up_3(up_x, x2)
        up_x = self.up_4(up_x, x1)
        decoded_spect = self.outc(up_x, spect)

        msg = self.msg_down(decoded_spect)
        msg = torch.mean(msg, dim = -1, keepdim=True).squeeze(-1)
        # msg = reverse_diffusion_model.inference(msg, num_time_steps)
        msg = self.msg_conv(msg)

        return msg
        
class UNet_audio_discriminator(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_audio_discriminator, self).__init__()
        