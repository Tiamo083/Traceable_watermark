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
import gc


from My_model.modules import Encoder, Decoder, Discriminator
from dataset.data import wav_dataset as used_dataset


# 创建水印，选择一个分布来生成水印
def generate_watermark(batch_size, length, micro, sigma):
    sigmoid = torch.nn.Sigmoid()
    eye_matrix = np.eye(length)
    mask_convariance_maxtix = eye_matrix * (sigma ** 2)
    center = np.ones(length) * micro

    w_bit = multivariate_normal.rvs(mean = center, cov = mask_convariance_maxtix, size = [batch_size, 1])
    w_bit = torch.from_numpy(w_bit).float()
    return w_bit

# seet seeds
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging_mark = "#" * 20
logging.basicConfig(level=logging.INFO, format = '%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args, configs):
    print('main function')
    process_config, model_config, train_config = configs
    pre_step = 0

    # -------------- get train dataset 
    train_audios = used_dataset(process_config=process_config, train_config=train_config, flag='val')
    val_audios = used_dataset(process_config=process_config, train_config=train_config, flag='test')
    
    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(train_audios)
    train_audio_loader = DataLoader(train_audios, batch_size=batch_size, shuffle=True)
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle = False)

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
    # shared parameters
    if model_config["structure"]["share"]:
        if model_config["structure"]["transformer"]:
            decoder.msg_decoder = encoder.encoder
        else:
            decoder.wav_encoder = encoder.wav_encoder
        
    # -------------- optimizer
    en_de_optim = Adam(
        params = chain(encoder.parameters(), decoder.parameters()),
        betas = train_config["optimize"]["betas"],
        eps = train_config["optimize"]["eps"],
        weight_decay=train_config["optimize"]["weight_decay"],
        lr = train_config["optimize"]["lr"]
    )
    lr_sched = StepLR(en_de_optim, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])

    # -------------- Loss 
    # use MSELoss to evaluate msg_loss and embedding_loss
    loss = Loss_identity(train_config=train_config)
    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)

    # ---------------- train
    print(logging_mark + "\t" + "Begin Trainging" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    train_len = len(train_audio_loader)

    for ep in range(1, epoch_num+1):
        encoder.train()
        decoder.train()
        discriminator.train()
        step = 0
        print('Epoch {}/{}'.format(ep, epoch_num))
        # for sample in track(audios_loader):
        # @TODO save this as track
        for i, sample in enumerate(train_audio_loader):
            global_step += 1
            step += 1
            # ---------------- build watermark
            # fragile_msg = np.random.choice([0, 1], [batch_size, 1, msg_length])
            # fragile_msg = torch.from_numpy(fragile_msg).float()*2 - 1
            # wav_matrix = sample["matrix"].to(device)
            # fragile_msg = fragile_msg.to(device)
            # fragile_encoded, _ = encoder(wav_matrix, fragile_msg, global_step)
            # fragile_decoded = decoder(fragile_encoded, global_step, 0)
            # fragile_losses = loss.en_de_loss(wav_matrix, fragile_encoded, fragile_msg, fragile_decoded)

            fragile_msg = np.random.choice([0, 1], [batch_size, 1, int(msg_length // 2)])
            fragile_msg = torch.from_numpy(fragile_msg).float() * 2 - 1
            robust_msg = np.random.choice([0,1], [batch_size, 1, int(msg_length // 2)])
            robust_msg = torch.from_numpy(robust_msg).float()*2 - 1
            # 把msg从0, 1变成-1，1            
            msg = torch.cat((robust_msg, fragile_msg), dim = 2)
            
            wav_matrix = sample["matrix"].to(device)
            msg = msg.to(device)
            encoded, carrier_wateramrked = encoder(wav_matrix, msg, global_step)
            decoded = decoder(encoded, global_step, train_config["attack_type"])

            losses = loss.half_en_de_loss(wav_matrix, encoded, msg, decoded[0], decoded[1])
            # pdb.set_trace()

            if global_step < pre_step:
                sum_loss = lambda_m*losses[1] + lambda_m*losses[2] # + lambda_m * losses[3]
            else:
                sum_loss = lambda_e*losses[0] + lambda_m*losses[1] + lambda_m*losses[2] # + lambda_m * losses[3]

            if train_config["adv"]:
                # 提升discriminator的能力
                d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)
                d_loss_on_cover.backward()

                d_target_label_encoded = torch.full((batch_size, 1), 0, device=device).float()
                d_on_encoded = discriminator(encoded.detach())

                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)

                d_loss_on_encoded.backward()
                my_step(d_op, lr_sched_d, global_step, train_len)

            # adv
            if train_config["adv"]:
                # 判断encoded后的音频是真还是假，我们想要它被判断成真
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"] # modify weights of m and a for better convergence
                g_target_label_encoded = torch.full((batch_size, 1), 1, device=device).float()

                d_on_encoded_for_enc = discriminator(encoded)
                # target label for encoded images should be 'cover', because we want to fool the discriminator

                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)

                sum_loss += lambda_a*g_loss_adv

            sum_loss.backward()
            
            my_step(en_de_optim, lr_sched, global_step, train_len)
            print("curr_step = ", step, "show_circle = ", show_circle, "train_set = ", len(train_audio_loader))
            
            if step % show_circle == 0:
                # save generated audios
                save_audio_path = os.path.join("results/wm_speech", "attack:{}_epoch:{}.wav".format(train_config["attack_type"], ep))
                torchaudio.save(save_audio_path, src = encoded.detach().squeeze(1).to("cpu"), sample_rate = sample["trans_sr"])

                robust_msg, fragile_msg = torch.chunk(input = msg, chunks = 2, dim = 2)
                att_rec_r_msg, att_rec_f_msg = torch.chunk(input = decoded[0], chunks = 2, dim = 2)
                rec_r_msg, rec_f_msg = torch.chunk(input = decoded[1], chunks = 2, dim = 2)


                # decoder_acc = (decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()
                attack_decoder_acc = [((att_rec_r_msg >= 0).eq(robust_msg >= 0).sum().float() / robust_msg.numel()).item(), ((att_rec_f_msg >= 0).eq(fragile_msg >= 0).sum().float() / fragile_msg.numel()).item()]
                decoder_acc = [((rec_r_msg >= 0).eq(robust_msg >= 0).sum().float() / robust_msg.numel()).item(), ((rec_f_msg >= 0).eq(fragile_msg >= 0).sum().float() / fragile_msg.numel()).item()]
                
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)

                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                
                norm2=mse_loss(wav_matrix.detach(),zero_tensor)
                print('-' * 100)
                print("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - no_attack_acc:[{:.8f},{:.8f}] - attack_acc:[{:.8f},{:.8f}] - snr:{:.8f} - norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    step, losses[0], losses[1], decoder_acc[0], decoder_acc[1],\
                    attack_decoder_acc[0], attack_decoder_acc[1],\
                    snr, norm2, sample["patch_num"].item(), sample["pad_num"].item(),\
                    wav_matrix.shape[2], d_loss_on_encoded, d_loss_on_cover))
                  
                wandb.log({'wav_loss': losses[0]})
                wandb.log({'msg_loss': losses[1]})
                wandb.log({'snr': snr})
                wandb.log({'d_loss_on_encoded': d_loss_on_encoded})
                wandb.log({'d_loss_on_cover': d_loss_on_cover})

                
        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["ckpt"], 'pth')
            else:
                path = os.path.join(train_config["path"]["ckpt"], 'pth_ab')
            save_op(path, ep, encoder, decoder, discriminator, en_de_optim, train_config["attack_type"])
            shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training scripts


        # Validation Stage
        """with torch.no_grad():
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            avg_acc = [0, 0]
            avg_snr = 0
            avg_wav_loss = 0
            avg_msg_loss = 0
            avg_d_loss_on_encoded = 0
            avg_d_loss_on_cover = 0
            count = 0
            for sample in track(val_audios_loader):
                count += 1
                # ---------------- build watermark
                msg = np.random.choice([0,1], [batch_size, 1, msg_length])
                msg = torch.from_numpy(msg).float()*2 - 1
                wav_matrix = sample["matrix"].to(device)
                msg = msg.to(device)
                encoded, carrier_wateramrked = encoder(wav_matrix, msg, global_step)
                decoded = decoder(y = encoded, global_step = global_step, attack_type = train_config["attack_type"])
                losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)
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

                decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                # norm2=mse_loss(wav_matrix.detach(),zero_tensor)
                avg_acc[0] += decoder_acc[0]
                avg_acc[1] += decoder_acc[1]
                avg_snr += snr
                avg_wav_loss += losses[0]
                avg_msg_loss += losses[1]
                avg_d_loss_on_cover += d_loss_on_cover
                avg_d_loss_on_encoded += d_loss_on_encoded
            avg_acc[0] /= count
            avg_acc[1] /= count
            avg_snr /= count
            avg_wav_loss /= count
            avg_msg_loss /= count
            avg_d_loss_on_encoded /= count
            avg_d_loss_on_cover /= count
            logging.info('#e' * 60)
            logging.info("epoch:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                ep, avg_wav_loss, avg_msg_loss, avg_acc[0], avg_acc[1], avg_snr, avg_d_loss_on_encoded, avg_d_loss_on_cover))"""

        with torch.no_grad():
            encoder.eval()
            decoder.eval()
            discriminator.eval()
            wm_avg_acc = [0, 0]
            attack_wm_avg_acc = [0, 0]
            unwm_avg_acc = [0, 0]
            avg_snr = 0
            wm_avg_wav_loss = 0
            unwm_avg_wav_loss = 0
            wm_avg_r_msg_loss = 0
            wm_avg_f_msg_loss = 0
            unwm_avg_msg_loss = 0
            avg_d_loss_on_encoded = 0
            avg_d_loss_on_cover = 0
            count = 0
            no_attack_msg_loss = 0
            
            # @TODO save it as track
            for i, sample in enumerate(val_audios_loader):
                if count % 500 == 0:
                    print("count = ", count, "global_step = ", global_step)
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
                wm_losses = loss.half_en_de_loss(wav_matrix, encoded, msg, wm_decoded[0], wm_decoded[1])
                unwm_losses = loss.half_en_de_loss(wav_matrix, wav_matrix, msg, unwm_decoded[0], unwm_decoded[1])
                # wm_losses = loss.en_de_loss(wav_matrix, encoded, msg, wm_decoded)
                # unwm_losses = loss.en_de_loss(wav_matrix, encoded, msg, unwm_decoded)

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

                robust_msg, fragile_msg = torch.chunk(input = msg, chunks = 2, dim = 2)
                att_rec_r_msg, att_rec_f_msg = torch.chunk(input = wm_decoded[0], chunks = 2, dim = 2)
                rec_r_msg, rec_f_msg = torch.chunk(input = wm_decoded[1], chunks = 2, dim = 2)

                # 在受到攻击后，鲁棒水印和脆弱水印的正确率
                attack_decoder_acc = [((att_rec_r_msg >= 0).eq(robust_msg >= 0).sum().float() / robust_msg.numel()).item(), ((att_rec_f_msg >= 0).eq(fragile_msg >= 0).sum().float() / fragile_msg.numel()).item()]
                # 没受到攻击时，鲁棒水印和脆弱水印的正确率
                wm_decoder_acc = [((rec_r_msg >= 0).eq(robust_msg >= 0).sum().float() / robust_msg.numel()).item(), ((rec_f_msg >= 0).eq(fragile_msg >= 0).sum().float() / fragile_msg.numel()).item()]

                # wm_decoder_acc = [((wm_decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((wm_decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                unwm_decoder_acc = [((unwm_decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((unwm_decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                # norm2=mse_loss(wav_matrix.detach(),zero_tensor)


                attack_wm_avg_acc[0] += attack_decoder_acc[0]
                attack_wm_avg_acc[1] += attack_decoder_acc[1]
                wm_avg_acc[0] += wm_decoder_acc[0]
                wm_avg_acc[1] += wm_decoder_acc[1]
                unwm_avg_acc[0] += unwm_decoder_acc[0]
                unwm_avg_acc[1] += unwm_decoder_acc[1]

                wm_avg_wav_loss += wm_losses[0]
                no_attack_msg_loss += wm_losses[1]
                wm_avg_r_msg_loss += wm_losses[2]
                wm_avg_f_msg_loss += wm_losses[3]

                unwm_avg_wav_loss += unwm_losses[0]
                unwm_avg_msg_loss += unwm_losses[1]

                avg_d_loss_on_cover += d_loss_on_cover
                avg_d_loss_on_encoded += d_loss_on_encoded
                avg_snr += snr

            attack_wm_avg_acc[0] /= count
            attack_wm_avg_acc[1] /= count

            wm_avg_acc[0] /= count
            wm_avg_acc[1] /= count
            unwm_avg_acc[0] /= count
            unwm_avg_acc[1] /= count

            wm_avg_wav_loss /= count
            wm_avg_r_msg_loss /= count
            wm_avg_f_msg_loss /= count
            no_attack_msg_loss /= count

            unwm_avg_wav_loss /= count
            unwm_avg_msg_loss /= count

            avg_d_loss_on_encoded /= count
            avg_d_loss_on_cover /= count
            avg_snr /= count
            

            wandb.log({"attack_watermark_r_decoder_acc": attack_wm_avg_acc[0]})
            wandb.log({"attack_watermark_f_decoder_acc": attack_wm_avg_acc[1]})
            wandb.log({"watermark_r_decoder_acc": wm_avg_acc[0]})
            wandb.log({'watermark_f_decoder_acc': wm_avg_acc[1]})
            wandb.log({'valid_unwatermark_msg_loss': unwm_avg_acc[0]})
            wandb.log({'attack_robust_msg_loss': wm_avg_r_msg_loss})
            wandb.log({'attack_fragile_msg_loss': wm_avg_f_msg_loss})
            wandb.log({'no_attack_real_msg_loss': no_attack_msg_loss})
            
            
            print('#e' * 60)
            print("epoch:{} - wm_wav_loss:{:.8f} - unwm_wav_loss:{:.8f} - wm_msg_r_loss:{:.8f} - wm_msg_f_loss:{:.8f} - unwm_msg_loss:{:.8f} - wm_acc:[{:.8f},{:.8f}] - unwm_acc:[{:.8f},{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    ep, wm_avg_wav_loss, unwm_avg_wav_loss, wm_avg_r_msg_loss, wm_avg_f_msg_loss, unwm_avg_msg_loss, wm_avg_acc[0], wm_avg_acc[1], unwm_avg_acc[0], unwm_avg_acc[1], avg_snr, avg_d_loss_on_encoded, avg_d_loss_on_cover))



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
    parser.add_argument(
        "--team_name", type=str
    )
    parser.add_argument(
        "--project_name", type=str
    )
    parser.add_argument(
        "--experiment_name", default= "original", type=str
    )
    parser.add_argument(
        "--scenario_name", type=str
    )
    parser.add_argument(
        "--seed",type=int,default=0
    )

    args = parser.parse_args()
    
    wandb.init(config=args,
               project=args.project_name,
               entity=args.team_name,
               notes=socket.gethostname(),
               name=args.experiment_name+"_"+str(args.seed),
               group=args.scenario_name,
               job_type="training",
               reinit=True)
    

    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(args, configs)