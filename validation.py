from multiprocessing import process
import os
from sunau import AUDIO_UNKNOWN_SIZE
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np

# import wandb
import librosa
import torchaudio
from librosa.core import load
from librosa.filters import mel as librosa_mel_fn
from torch.optim import Adam
from rich.progress import track
from torch.utils.data import DataLoader
from scipy.io.wavfile import write
from model.loss import Loss_identity
from utils.tools import save, log, save_op
from utils.optimizer import ScheduledOptimMain, ScheduledOptimDisc, my_step
from itertools import chain
from torch.nn.functional import mse_loss
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from scipy.stats import multivariate_normal
import random
import pdb
import shutil
import json
import wandb
import socket


from My_model.modules import Encoder, Decoder, Discriminator
from dataset.data import wav_dataset as used_dataset

# set seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging_mark = "#" * 20
logging.basicConfig(level=logging.INFO, format = '%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def val(args, configs):
    logging.info("validation")
    process_config, model_config, train_config = configs

    # ------------- get validation dataset
    val_audios = used_dataset(process_config=process_config, train_config=train_config, flag='val')
    batch_size = train_config["optimize"]["batch_size"]
    val_audios_loader = DataLoader(val_audios, batch_size = batch_size, shuffle = False)

    # -------------- build model
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    # msg_length:16bit, the last 6 bits are used to clarify robust watermark and fragile watermark
    msg_length = train_config["watermark"]["length"]


    encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
    decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)

    # adv
    if train_config["adv"]:
        discriminator = Discriminator(process_config).to(device)
        # d_op = ScheduledOptimDisc(discriminator,train_config)
        d_op = Adam(
            params=chain(discriminator.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )
        lr_sched_d = StepLR(d_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])
    
    en_de_optim = Adam(
        params = chain(encoder.parameters(), decoder.parameters()),
        betas = train_config["optimize"]["betas"],
        eps = train_config["optimize"]["eps"],
        weight_decay=train_config["optimize"]["weight_decay"],
        lr = train_config["optimize"]["lr"]
    )
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(val_log_path, exist_ok=True)

    op_path = "/amax/home/Tiamo/Traceable_watermark/results/ckpt/pth/none-conv227_ep_20_2024-02-21_06_01_35.pth"
    checkpoint = torch.load(op_path)
    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])
    discriminator.load_state_dict(checkpoint['discriminator'])
    en_de_optim.load_state_dict(checkpoint['en_de_op'])

    