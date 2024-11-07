from base64 import encode
from multiprocessing import process
from re import L
import torch
import torch.nn as nn
from torch.nn import LeakyReLU, Tanh
from .blocks import FCBlock, PositionalEncoding, Mish, Conv1DBlock, Conv2Encoder, WatermarkEmbedder, WatermarkExtracter, ReluBlock, LSTM_Model
from .Myblocks import Msg_Process, Spect_Encoder, Watermark_Embedder, Watermark_Extracter, ReluBlock, Msg_after_Process
from distortions.frequency import TacotronSTFT, fixed_STFT, tacotron_mel
from distortions.dl import distortion
import pdb
import hifigan
import json
import torchaudio

def save_spectrum(spect, phase, flag='linear'):
    import numpy as np
    import os
    import librosa
    import librosa.display
    root = "draw_figure"
    import matplotlib.pyplot as plt
    spect = spect/torch.max(torch.abs(spect))
    spec = librosa.amplitude_to_db(spect.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)
    phase = phase/torch.max(torch.abs(phase))
    spec = librosa.amplitude_to_db(phase.squeeze(0).cpu().numpy(), ref=np.max, amin=1e-5)
    img=librosa.display.specshow(spec, sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.clim(-40, 40)
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_phase_spectrogram.png'), bbox_inches='tight', pad_inches=0.0)

def save_feature_map(feature_maps):
    import os
    import matplotlib.pyplot as plt
    import librosa
    import numpy as np
    import librosa.display
    feature_maps = feature_maps.cpu().numpy()
    root = "draw_figure"
    output_folder = os.path.join(root,"feature_map_or")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    n_channels = feature_maps.shape[0]
    for channel_idx in range(n_channels):
        fig, ax = plt.subplots()
        ax.imshow(feature_maps[channel_idx, :, :], cmap='gray')
        ax.axis('off')
        output_file = os.path.join(output_folder, f'feature_map_channel_{channel_idx + 1}.png')
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.0)
        plt.close(fig)

def save_waveform(a_tensor, flag='original'):
    import os
    import librosa
    import librosa.display
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile
    root = "draw_figure"
    y = a_tensor.cpu().numpy()
    soundfile.write(os.path.join(root, flag + "_waveform.wav"), y, samplerate=22050)
    D = librosa.stft(y)
    spectrogram = np.abs(D)
    img=librosa.display.specshow(librosa.amplitude_to_db(spectrogram, ref=np.max), sr=22050, x_axis='time', y_axis='log', y_coords=None);
    plt.axis('off')
    plt.savefig(os.path.join(root, flag + '_amplitude_spectrogram_from_waveform.png'), bbox_inches='tight', pad_inches=0.0)


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

class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()
        
        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["LSTM_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]
        self.LSTM_dim = model_config["conv2"]["LSTM_dim"]

        self.vocoder_step = model_config["structure"]["vocoder_step"]


        # spect encoder
        self.EN_spect = Spect_Encoder(input_channel=1, latent_dim=self.LSTM_dim, block=self.block, n_layers=self.layers_EM)

        # watermark linear encoder
        self.msg_linear_en = Msg_Process(msg_length, win_dim, activation = LeakyReLU(inplace=True))

        # watermarked_spect decoder
        self.DE_spect = Watermark_Embedder(self.EM_input_dim, self.LSTM_dim, self.layers_EM)

        # stft transform
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

    def forward(self, x, msg, global_step):
        num_samples = x.shape[2]
        # (1, 513, x)
        spect, phase = self.stft.transform(x)

        # (1, hidden_dim, 513, x)
        spect_encoded = self.EN_spect(spect.unsqueeze(1))

        # (1, 1, 513, x)
        watermark_encoded = self.msg_linear_en(msg).transpose(1, 2).unsqueeze(1).repeat(1, 1, 1, spect_encoded.shape[3])

        # (1, hidden_dim + 2, 513, x)
        concatenated_feature = torch.cat((spect_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)

        # (1, 1, 513, x)
        watermarked_spect = self.DE_spect(concatenated_feature)

        self.stft.num_samples = num_samples
        y = self.stft.inverse(watermarked_spect.squeeze(1), phase.squeeze(1))
        return y, watermarked_spect



"""class Encoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=6, transformer_drop=0.1, attention_heads=8):
        super(Encoder, self).__init__()

        self.name = "conv2"
        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.add_carrier_noise = False
        self.block = model_config["conv2"]["block"]
        self.layers_CE = model_config["conv2"]["layers_CE"]
        self.EM_input_dim = model_config["conv2"]["LSTM_dim"] + 2
        self.layers_EM = model_config["conv2"]["layers_EM"]
        self.LSTM_dim = model_config["conv2"]["LSTM_dim"]

        self.vocoder_step = model_config["structure"]["vocoder_step"]
        #MLP for the input wm
        self.msg_linear_in = FCBlock(msg_length, win_dim, activation=LeakyReLU(inplace=True))

        #stft transform
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        
        # append n_layers multi-conv2d
        self.ENc = Conv2Encoder(input_channel=1, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_CE)
        self.LSTM_ENc = LSTM_Model(input_channel=model_config["conv2"]["hidden_dim"], hidden_dim = model_config["conv2"]["LSTM_dim"], block=self.block, n_layers=self.layers_EM)
        # append (n_layers - 1) multi-conv2d
        self.EM = WatermarkEmbedder(input_channel=self.EM_input_dim, hidden_dim = model_config["conv2"]["hidden_dim"], block=self.block, n_layers=self.layers_EM)
        

    def forward(self, x, msg, global_step):
        # x.shape:[1, 1, length]
        # msg.shape:[1, 1, 10]
        num_samples = x.shape[2]

        # spect/phase shape: [1, 513, x]
        spect, phase = self.stft.transform(x)

        # carrier_encoded shape:[1, 64, 513, x]
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        # lstm_encoded shape:[1, 128, 513, x]
        lstm_encoded = self.LSTM_ENc(carrier_encoded.transpose(1, 3).squeeze()).transpose(0, 2).unsqueeze(0)
        # 将watermark通过Encoder，并扩展至和carrier_encoded一样的shape
        # watermark_encoded.shape:[1, 1, 513, x]
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        # concatenated_feature.shape:[1, 66, 513, x]
        concatenated_feature = torch.cat((lstm_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)
        # carrier_watermarked.shape:[1, 1, 513, x]
        carrier_watermarked = self.EM(concatenated_feature)

        
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_watermarked.squeeze(1), phase.squeeze(1))
        return y, carrier_watermarked
    
    def test_forward(self, x, msg):
        num_samples = x.shape[2]
        spect, phase = self.stft.transform(x)
        
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  
        
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        return y, carrier_wateramrked
    
    def save_forward(self, x, msg):
        num_samples = x.shape[2]
        save_waveform(x.squeeze())
        spect, phase = self.stft.transform(x)
        # save spectrum
        save_spectrum(spect, phase, 'linear')
        
        carrier_encoded = self.ENc(spect.unsqueeze(1)) 
        # save feature_map
        # save_feature_map(carrier_encoded[0])
        watermark_encoded = self.msg_linear_in(msg).transpose(1,2).unsqueeze(1).repeat(1,1,1,carrier_encoded.shape[3])
        concatenated_feature = torch.cat((carrier_encoded, spect.unsqueeze(1), watermark_encoded), dim=1)  
        carrier_wateramrked = self.EM(concatenated_feature)  
        save_spectrum(carrier_wateramrked.squeeze(1), phase, 'wmed_linear')
        
        self.stft.num_samples = num_samples
        y = self.stft.inverse(carrier_wateramrked.squeeze(1), phase.squeeze(1))
        save_waveform(y.squeeze().squeeze(), 'wmed')
        return y, carrier_wateramrked
    """


class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion(process_config)
        
        self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocoder = get_vocoder(device)
        self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = Watermark_Extracter(input_channel = 1, latent_dim = model_config["conv2"]["LSTM_dim"], n_layers = nlayers_decoder)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        self.msg_Decoder = Msg_after_Process(win_dim, msg_length)
    
    def forward(self, y, global_step, attack_type):
        y_identity = y.clone()
        # if global_step > self.vocoder_step:
        #     y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
        #     y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        # else:
        #     y_d = y
        
        if self.robust:
            y_d = self.dl(y, attack_choice = attack_type, ratio = 10, src_path = 'Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav')
        else:
            y_d = y
        
        # 做傅里叶变换
        spect, phase = self.stft.transform(y_d)

        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_Decoder(msg)

        spect_identity, phase_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        # 没有传输损失的msg
        msg_identity = self.msg_Decoder(msg_identity)
        return msg, msg_identity

"""class Decoder(nn.Module):
    def __init__(self, process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=6, transformer_drop=0.1, attention_heads=8):
        super(Decoder, self).__init__()
        self.robust = model_config["robust"]
        if self.robust:
            self.dl = distortion()

        self.mel_transform = TacotronSTFT(filter_length=process_config["mel"]["n_fft"], hop_length=process_config["mel"]["hop_length"], win_length=process_config["mel"]["win_length"])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vocoder = get_vocoder(device)
        self.vocoder_step = model_config["structure"]["vocoder_step"]

        win_dim = int((process_config["mel"]["n_fft"] / 2) + 1)
        self.block = model_config["conv2"]["block"]
        self.EX = WatermarkExtracter(input_channel=1, hidden_dim=model_config["conv2"]["hidden_dim"], lstm_dim=model_config["conv2"]["LSTM_dim"], block=self.block)
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])
        self.msg_linear_out = FCBlock(win_dim, msg_length)

    def forward(self, y, global_step, attack_type):
        y_identity = y.clone()
        if global_step > self.vocoder_step:
            y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
            # y = self.vocoder(y_mel)
            # 用griffin_lim vocoder进行转换
            y_d = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        else:
            y_d = y
        if self.robust:
            # 传输损失
            # y_d是audio
            # y_d_d = self.dl(y_d, self.robust)
            y_d_d = self.dl(y_d, attack_choice = attack_type, ratio = 10, tgt_path = '/amax/home/Tiamo/Traceable_watermark/Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav')
        else:
            y_d_d = y_d
        
        # 做傅里叶变换
        spect, phase = self.stft.transform(y_d_d)

        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm, dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)

        spect_identity, phase_identity = self.stft.transform(y_identity)
        extracted_wm_identity = self.EX(spect_identity.unsqueeze(1)).squeeze(1)
        msg_identity = torch.mean(extracted_wm_identity,dim=2, keepdim=True).transpose(1,2)
        # 没有传输损失的msg
        msg_identity = self.msg_linear_out(msg_identity)
        return msg, msg_identity
    
    def test_forward(self, y):
        spect, phase = self.stft.transform(y)
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg
    
    def save_forward(self, y):
        # save mel_spectrum
        y_mel = self.mel_transform.mel_spectrogram(y.squeeze(1))
        save_spectrum(y_mel, y_mel, 'mel')
        y, reconstruct_spec = self.mel_transform.griffin_lim(magnitudes=y_mel)
        y = y.unsqueeze(1)
        save_waveform(y.squeeze().squeeze(), 'distored')
        save_spectrum(reconstruct_spec, reconstruct_spec, 'recon')
        # y = (self.mel_transform.griffin_lim(magnitudes=y_mel)).unsqueeze(1)
        spect, phase = self.stft.transform(y)
        save_spectrum(spect, spect, 'distored')
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg
    
    def mel_test_forward(self, spect):
        extracted_wm = self.EX(spect.unsqueeze(1)).squeeze(1)
        msg = torch.mean(extracted_wm,dim=2, keepdim=True).transpose(1,2)
        msg = self.msg_linear_out(msg)
        return msg"""
        


class Discriminator(nn.Module):
    def __init__(self, process_config):
        super(Discriminator, self).__init__()
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
        self.stft = fixed_STFT(process_config["mel"]["n_fft"], process_config["mel"]["hop_length"], process_config["mel"]["win_length"])

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