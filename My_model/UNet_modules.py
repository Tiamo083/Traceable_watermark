from base64 import encode
from multiprocessing import process
from re import L
import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Tanh

from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter, ReluBlock, LSTM_Model
from .Myblocks import Msg_Process, Spect_Encoder, Watermark_Embedder, Watermark_Extracter, ReluBlock, Msg_after_Process
from .UNnet_blocks import Unet_wm_embedder, Unet_wm_extractor
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
from distortions.dl import distortion
import pdb
import hifigan
import json
import torchaudio
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_vocoder(device):
    with open("hifigan/config.json", "r") as f:
        config = json.load(f)
    config = hifigan.AttrDict(config)
    vocoder = hifigan.Generator(config)
    ckpt = torch.load("./hifigan/model/VCTK_V1/generator_v1")
    vocoder.load_state_dict(ckpt["generator"])
    vocoder.eval()
    vocoder.remove_weight_norm()
    vocoder.to(device)
    freeze_model_and_submodules(vocoder)
    return vocoder

def freeze_model_and_submodules(model):
    for param in model.parameters():
        param.requires_grad = False

    for module in model.children():
        if isinstance(module, nn.Module):
            freeze_model_and_submodules(module)

class Embedder(nn.Module):
    def __init__(self, process_config, msg_length, win_dim, num_time_steps):
        super(Embedder, self).__init__()

        self.name = "Unet_watermark_embedder"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        
        self.Unet = Unet_wm_embedder(1, 1, msg_length, num_time_steps)

        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

        

    def forward(self, x, msg, global_step):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        watermarked_spect, diffusion_model = self.Unet(spect.unsqueeze(1), msg)
        self.stft.num_samples = num_samples
        y = self.stft.inverse(watermarked_spect.squeeze(1), phase.squeeze(1))
        return y, watermarked_spect, diffusion_model

class Extractor(nn.Module):
    def __init__(self, process_config, model_config, msg_length, num_time_steps, win_dim):
        super(Extractor, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion(process_config)
        
        self.num_time_steps = num_time_steps
        self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocoder = get_vocoder(device)
        self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]

        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        self.De_Unet = Unet_wm_extractor(msg_length, 1, 1)

    def forward(self, y, source_audio, global_step, attack_type, reverse_diffusion):
        y_identity = y.clone()

        # if self.robust:
        #     y_d = self.dl(y, source_audio,attack_choice = attack_type, ratio = 10, src_path = 'Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav')
        # else:
        #     y_d = y
        
        with torch.no_grad():
            y_detach = y.detach()
            if self.robust:
                y_detach_dl = self.dl(y_detach, source_audio, attack_choice = attack_type, ratio = 10, src_path = 'Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav')
                n_y_detach_dl = y_detach_dl.shape[2]
                n_y = y.shape[2]

                if n_y_detach_dl < n_y:
                    padding = torch.full((1, 1, n_y - n_y_detach_dl), fill_value = 0.000001, dtype=y_detach_dl.dtype).to(device)
                    y_detach_dl_result = torch.cat((y_detach_dl, padding), dim = 2)
                else:
                    y_detach_dl_result = y_detach_dl[:, :, :n_y]
                
                y_gap = y_detach_dl_result - y_detach
            else:
                y_gap = torch.zeros_like(y, device=self.device, requires_grad=True).to(device)
        
        y_d = y + y_gap

        spect, phase = self.stft.transform(y_d)
        msg = self.De_Unet(spect.unsqueeze(1), self.num_time_steps, reverse_diffusion)

        
        spect_identity, phase_identity = self.stft.transform(y_identity)
        msg_identity = self.De_Unet(spect_identity.unsqueeze(1), self.num_time_steps, reverse_diffusion)

        return msg, msg_identity
    
class UNet_discriminator(nn.Module):
    def __init__(self, process_config):
        super(UNet_discriminator, self).__init__()
        self.conv = nn.Sequential(
                ReluBlock(1,16,3,1,1),
                ReluBlock(16,32,3,1,1),
                ReluBlock(32,64,3,1,1),
                ReluBlock(64,128,3,1,1),
                nn.AdaptiveAvgPool2d(output_size=(1, 1))
                )
        self.linear = nn.Sequential(
            nn.Linear(128,64),
            nn.Linear(64,32),
            nn.Linear(32,16),
            nn.Linear(16,1)
            )
        self.dis_unet = nn.Sequential()
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x):
        spect, phase = self.stft.transform(x)
        x = spect.unsqueeze(1)
        x = self.conv(x)
        x = x.squeeze(2).squeeze(2)
        x = self.linear(x)
        return x