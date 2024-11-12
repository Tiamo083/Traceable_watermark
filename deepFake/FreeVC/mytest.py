import os
import argparse
import torch
import librosa
import time
import torchaudio as ta

from scipy.io.wavfile import write
from tqdm import tqdm

import sys
sys.path.append('deepFake/FreeVC')

import freevcutils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
from wavlm import WavLM, WavLMConfig
from speaker_encoder.voice_encoder import SpeakerEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hpfile = "deepFake/FreeVC/configs/freevc.json"
ptfile = "deepFake/FreeVC/checkpoints/freevc.pth"

hps = freevcutils.get_hparams_from_file(hpfile)

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model).to(device)
_ = net_g.eval()
_ = freevcutils.load_checkpoint(ptfile, net_g, None, True)

cmodel = freevcutils.get_cmodel(0)

if hps.model.use_spk:
    smodel = SpeakerEncoder('deepFake/FreeVC/speaker_encoder/ckpt/pretrained_bak_5805000.pt')

# src 提供音频的内容，tgt 提供音频的音色

tgt = "/amax/home/zhaoxd/temp/autovc/8534_216567_000015_000010.wav"
wav_tgt, sr = ta.load(tgt)
resampler = ta.transforms.Resample(orig_freq=sr, new_freq=hps.data.sampling_rate)
wav_tgt = resampler(wav_tgt)

wav_tgt = wav_tgt.squeeze(0).numpy()
wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
g_tgt = smodel.embed_utterance(wav_tgt)
g_tgt = torch.from_numpy(g_tgt).unsqueeze(0).to(device)

src = "/amax/home/zhaoxd/temp/autovc/251-136532-0001.flac"
wav_src, _ = librosa.load(src, sr=hps.data.sampling_rate)
wav_src = torch.from_numpy(wav_src).unsqueeze(0).to(device)
c = freevcutils.get_content(cmodel, wav_src)

audio = net_g.infer(c, g=g_tgt)
print(audio.shape)
audio = audio.squeeze(0).cpu()
ta.save("deepFake/FreeVC/outputs/freevc/test.wav", audio, hps.data.sampling_rate)
