from base64 import encode
import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Tanh
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter,  ReluBlock
from distortions.mel_transform import STFT
from distortions.dl import distortion
import pdb


class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["hidden_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]

        #MLP for the input wm
        self.msg_linear_in = FCBlock(msg_length, win_dim, activation=LeakyReLU(inplace=True))

        #stft transform
        self.stft = STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"])

        self.ENc = Conv2Encoder(input_channel=1, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_CE)

        self.EM = WatermarkEmbedder(input_channel=self.EM_input_dim, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_EM)

    def forward(self, x, msg):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  

        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked



class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion()
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(input_channel=1, hidden_dim=model_config["conv2"]["hidden_dim"], block=self.block)
        self.stft = STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"])
        self.msg_linear_out = FCBlock(win_dim, msg_length)

    def forward(self, y):
        # print(y.shape)
        if self.robust:
            y = self.dl(y, self.robust)
        # print(f"dl:{y.shape}")
        spect, phase = self.stft.transform(y)
        # pdb.set_trace()
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                ReluBlock(1,16,3,1,1),
                ReluBlock(16,32,3,1,1),
                ReluBlock(32,64,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Linear(64,1)
        self.stft = STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"])

    def forward(self, x):
        spect, phase = self.stft.transform(x)
        x = spect.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x


def get_param_num(model):
    num_param = sum(param.numel() for param in model.parameters())
    return num_param
