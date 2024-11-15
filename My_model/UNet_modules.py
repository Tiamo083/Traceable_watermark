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
    def __init__(self, process_config, msg_length, win_dim):
        super(Embedder, self).__init__()

        self.name = "Unet_watermark_embedder"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        
        self.Unet = Unet_wm_embedder(1, 1, True)

        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

        

    def forward(self, x, msg, global_step):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        watermarked_spect = self.Unet(spect.unsqueeze(1), msg)
        self.stft.num_samples = num_samples
        y = self.stft.inverse(watermarked_spect.squeeze(1), phase.squeeze(1))
        return y, watermarked_spect

class Extractor(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim):
        super(Extractor, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion(y_detach,attack_choice = attack_type, ratio = 10, src_path = 'Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav')
        
        self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocoder = get_vocoder(device)
        self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]

        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        self.De_Unet = Unet_wm_extractor(1, 1, True)

    def forward(self, y, global_step, attack_type):
        y_identity = y.clone()

        if self.robust:
            y_d = self.dl(y, attack_choice = attack_type, ratio = 10, src_path = 'Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav')
        else:
            y_d = y
        
        spect, phase = self.stft.transform(y_d)
        extracted_wm = self.De_Unet(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm, dim = 2, keepdim=True).transpose(1, 2)
        msg = self.msg_Decoder(msg)

        
        spect_identity, phase_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.De_Unet(spect_identity.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        # 没有传输损失的msg
        msg_identity = self.msg_Decoder(msg_identity)

        return msg, msg_identity