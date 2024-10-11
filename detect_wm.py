import os
import torch
import yaml
import logging
import argparse
import warnings
import numpy as np

# import wandb
import librosa
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
import random
import pdb
import shutil
import json

# set seeds
seed = 2022
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


logging_mark = "#"*20
# warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    logging.info('main function')
    process_config, model_config, train_config = configs

    if model_config["structure"]["transformer"]:
        if model_config["structure"]["mel"]:
            from model.mel_modules import Encoder, Decoder
            from dataset.data import mel_dataset as my_dataset
        else:
            from model.modules import Encoder, Decoder
            from dataset.data import twod_dataset as my_dataset
    elif model_config["structure"]["conv2"]:
        logging.info("use conv2 model")
        from model.conv2_modules import Encoder, Decoder, Discriminator
        from dataset.data import mel_dataset as my_dataset
    elif model_config["structure"]["conv2mel"]:
        if not model_config["structure"]["ab"]:
            logging.info("use conv2mel model")
            # 使用这个model的Encoder,Decoder,Discriminator
            from model.conv2_mel_modules import Encoder, Decoder, Discriminator
            from dataset.data import wav_dataset as my_dataset
        else:
            logging.info("use ablation conv2mel model")
            from model.conv2_mel_modules_ab import Encoder, Decoder, Discriminator
            from dataset.data import wav_dataset as my_dataset
    else:
        from model.conv_modules import Encoder, Decoder
        from dataset.data import oned_dataset as my_dataset
    
    audios = my_dataset(process_config = process_config, train_config = train_config, flag = 'train')
    
    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(audios)
    audios_loader = DataLoader(audios, batch_size = batch_size, shuffle = True)

    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    msg_length = train_config["watermark"]["length"]
    if model_config["structure"]["mel"] or model_config["structure"]["conv2"]:
        # 用这里的参数
        encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    else:
        encoder = Encoder(model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
        decoder = Decoder(model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    
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
    # shared parameters
    # ---------------- Loss
    loss = Loss_identity(train_config=train_config)
    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # ---------------- valid
    logging.info(logging_mark + "\t" + "Begin Valid" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    valid_len = len(audios_loader)
    # --------------- get pth
    checkpoint = torch.load("/amax/home/Tiamo/Traceable_watermark/results/ckpt/pth/none-conv24_ep_20_2023-12-19_05_02_45.pth")
    encoder.load_state_dict(checkpoint["encoder"])
    decoder.load_state_dict(checkpoint["decoder"])
    discriminator.load_state_dict(checkpoint["discriminator"])

    for ep in range(1, epoch_num + 1):
        logging.info('Epoch {}/{}'.format(ep, epoch_num))
        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            wm_avg_acc = [0, 0]
            unwm_avg_acc = [0, 0]
            avg_snr = 0
            wm_avg_wav_loss = 0
            unwm_avg_wav_loss = 0
            wm_avg_msg_loss = 0
            unwm_avg_msg_loss = 0
            avg_d_loss_on_encoded = 0
            avg_d_loss_on_cover = 0
            count = 0
            for i, sample in enumerate(audios_loader):
                count += 1
                global_step += 1
                # ---------------------- build watermark
                msg = np.random.choice([0, 1], [batch_size, 1, msg_length])
                msg = torch.from_numpy(msg).float()*2 - 1
                wav_matrix = sample["matrix"].to(device)
                msg = msg.to(device)
                encoded, carrier_wateramrked = encoder(wav_matrix, msg, global_step)
                wm_decoded = decoder(encoded, global_step, train_config["attack_type"])
                unwm_decoded = decoder(wav_matrix, global_step, train_config["attack_type"])
                wm_losses = loss.en_de_loss(wav_matrix, encoded, msg, wm_decoded)
                unwm_losses = loss.en_de_loss(wav_matrix, encoded, msg, unwm_decoded)

                # adv
                if train_config["adv"]:
                    lambda_a = lambda_m = train_config["optimize"]["lambda_a"]
                    g_target_label_encoded = torch.full((batch_size, 1), 1, device=device).float()
                    d_on_encoded_for_enc = discriminator(encoded)
                    g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                if train_config["adv"]:
                    d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                    d_on_cover = discriminator(wav_matrix)
                    d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)

                    d_target_label_encoded = torch.full((batch_size, 1), 0, device=device).float()
                    d_on_encoded = discriminator(encoded.detach())
                    d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)

                wm_decoder_acc = [((wm_decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((wm_decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                unwm_decoder_acc = [((unwm_decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((unwm_decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                # norm2=mse_loss(wav_matrix.detach(),zero_tensor)
                wm_avg_acc[0] += wm_decoder_acc[0]
                wm_avg_acc[1] += wm_decoder_acc[1]
                unwm_avg_acc[0] += unwm_decoder_acc[0]
                unwm_avg_acc[1] += unwm_decoder_acc[1]
                wm_avg_wav_loss += wm_losses[0]
                wm_avg_msg_loss += wm_losses[1]
                unwm_avg_wav_loss += unwm_losses[0]
                unwm_avg_msg_loss += unwm_losses[1]
                avg_d_loss_on_cover += d_loss_on_cover
                avg_d_loss_on_encoded += d_loss_on_encoded
                avg_snr += snr
            
            wm_avg_acc[0] /= count
            wm_avg_acc[1] /= count
            unwm_avg_acc[0] /= count
            unwm_avg_acc[1] /= count
            wm_avg_wav_loss /= count
            wm_avg_msg_loss /= count
            unwm_avg_wav_loss /= count
            unwm_avg_msg_loss /= count
            avg_d_loss_on_encoded /= count
            avg_d_loss_on_cover /= count
            avg_snr /= count
            logging.info('#e' * 60)
            logging.info("epoch:{} - wm_wav_loss:{:.8f} - unwm_wav_loss:{:.8f} - wm_msg_loss:{:.8f} - unwm_msg_loss:{:.8f} - wm_acc:[{:.8f},{:.8f}] - unwm_acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    ep, wm_avg_wav_loss, unwm_avg_wav_loss, wm_avg_msg_loss, unwm_avg_msg_loss, wm_avg_acc[0], wm_avg_acc[1], unwm_avg_acc[0], unwm_avg_acc[1], avg_snr, avg_d_loss_on_encoded, avg_d_loss_on_cover))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument(
        "-p",
        "--process_config",
        type=str,
        required=True,
        help="path to process.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    args = parser.parse_args()

    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(args, configs)