import os
import sys
import time
import numpy as np
sys.path.append('./')
import argparse
import math
import shutil
import socket
import torch
import json
import yaml
import wandb
import random
import logging
import torchaudio
import torch.nn as nn
import torch.nn.functional as F

from tqdm.auto import tqdm
from torch.optim import Adam
from itertools import chain
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR
from torch.nn.functional import mse_loss

from utils.tools import save_op, save_multi_decoder_finetune_op
from model.loss import Loss_identity
from dataset.data import wav_dataset as used_dataset
from My_model.model_SVD import load_model_for_svd
from My_model.modules import Encoder, Decoder, Discriminator
from utils.optimizer import ScheduledOptimMain, ScheduledOptimDisc, my_step

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
seed = 2024
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

logging_mark = "#" * 20
logging.basicConfig(level=logging.INFO, format = '%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_ckpt(encoder, robust_decoder, fragile_decoder, config):
    ckpt_path = config["load_ckpt_path"]
    model_state_dict = torch.load(ckpt_path, map_location=device)
    encoder.load_state_dict(model_state_dict["encoder"])
    robust_decoder.load_state_dict(model_state_dict["robust_decoder"])
    fragile_decoder.load_state_dict(model_state_dict["fragile_decoder"])

    encoder = encoder.to(device)
    robust_decoder = robust_decoder.to(device)
    fragile_decoder = fragile_decoder.to(device)
    return encoder, robust_decoder, fragile_decoder

def finetune(args, configs):
    print("start finetuning")
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
    msg_length = train_config["watermark"]["length"]

    encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
    robust_decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    fragile_decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)
    
    print("loading checkpointing")
    encoder, robust_decoder, fragile_decoder = load_ckpt(encoder, robust_decoder, fragile_decoder, train_config)
    
    # print("Start SVD")
    # learnable_parameters_encoder = load_model_for_svd(encoder)
    # learnable_parameters_robust_decoder = load_model_for_svd(robust_decoder)
    # learnable_parameters_fragile_decoder = load_model_for_svd(fragile_decoder)
    # params_to_optimize_encoder = chain(learnable_parameters_encoder["params"], learnable_parameters_encoder["params_1d"])
    # params_to_optimize_robust_decoder = chain(learnable_parameters_robust_decoder["params"], learnable_parameters_robust_decoder["params_1d"])
    # params_to_optimize_fragile_decoder = chain(learnable_parameters_fragile_decoder["params"], learnable_parameters_fragile_decoder["params_1d"])

    if train_config["finetune_both_encoder_decoder"] == True:
        # Decomposing the weight matrix using SVD
        # finetune_robust_optim = Adam(
        #     params = [{"params": params_to_optimize_encoder}, {"params": params_to_optimize_robust_decoder}],
        #     betas = train_config["optimize"]["betas"],
        #     eps = train_config["optimize"]["eps"],
        #     weight_decay=train_config["optimize"]["weight_decay"],
        #     lr = train_config["optimize"]["lr"]
        # )

        # finetune_fragile_optim = Adam(
        #     params = params_to_optimize_fragile_decoder,
        #     betas = train_config["optimize"]["betas"],
        #     eps = train_config["optimize"]["eps"],
        #     weight_decay=train_config["optimize"]["weight_decay"],
        #     lr = train_config["optimize"]["lr"]
        # )
        
        finetune_optim = Adam(
            params = chain(encoder.parameters(), robust_decoder.parameters(), fragile_decoder.parameters()),
            betas = train_config["optimize"]["betas"],
            eps = train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )

    else:
        finetune_optim = Adam(
            params = chain(robust_decoder.parameters(), fragile_decoder.parameters()),
            betas = train_config["optimize"]["betas"],
            eps = train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )

    
    lr_sched = StepLR(finetune_optim, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])
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
    epoch_num = train_config["iter"]["finetune_epoch"]
    save_circle = train_config["iter"]["save_circle"]
    save_step = train_config["iter"]["save_step"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    lambda_f_m = train_config["optimize"]["lambda_f_m"]
    global_step = 0
    train_len = len(train_audio_loader)
    
    if train_config["adv"]:
        discriminator = Discriminator(process_config).to(device)
        ckpt_path = train_config["load_ckpt_path"]
        model_state_dict = torch.load(ckpt_path)
        discriminator.load_state_dict(model_state_dict["discriminator"])

        d_op = Adam(
            params=chain(discriminator.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )
        lr_sched_d = StepLR(d_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])

    for ep in range(1, epoch_num + 1):
        encoder.train()
        robust_decoder.train()
        fragile_decoder.train()
        discriminator.train()
        step = 0
        print('Finetune Epoch {}/{}'.format(ep, epoch_num))

        for i, sample in enumerate(train_audio_loader):
            global_step += 1
            step += 1
            # ---------------- build watermark
            msg = np.random.choice([0, 1], [batch_size, 1, int(msg_length)])
            msg = torch.from_numpy(msg).float() * 2 - 1

            wav_matrix = sample["matrix"].to(device)
            msg = msg.to(device)
            encoded, carrier_wateramrked = encoder(wav_matrix, msg, global_step)
            robust_decoded = robust_decoder(encoded, global_step, train_config["attack_type"])
            fragile_decoded = fragile_decoder(encoded, global_step, train_config["attack_type"])

            # losses = loss.half_en_de_loss(wav_matrix, encoded, msg, decoded[0], decoded[1])
            losses = loss.multi_de_one_wm_loss(wav_matrix, encoded, msg, robust_decoded[0], robust_decoded[1], fragile_decoded[0], fragile_decoded[1])

            if global_step < pre_step:
                robust_sum_loss = lambda_m * losses[1] + lambda_m * losses[2]
                fragile_sum_loss = lambda_m * losses[3] + lambda_m * losses[4]
                sum_loss = lambda_m*losses[1] + lambda_m*losses[2] + lambda_m * losses[3] + lambda_m * losses[4]
            else:
                robust_sum_loss = lambda_e * losses[0] + lambda_m * losses[1] + lambda_m * losses[2]
                fragile_sum_loss = lambda_e * losses[0] + lambda_m * losses[3] + lambda_m * losses[4]
                sum_loss = lambda_e*losses[0] + lambda_m*losses[1] + lambda_m*losses[2] + lambda_m * losses[3] + lambda_m * losses[4]

            # View only the results of the discriminator
            if train_config["adv"]:
                d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)

                d_target_label_encoded = torch.full((batch_size, 1), 0, device=device).float()
                d_on_encoded = discriminator(encoded.detach())

                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                d_loss_on_encoded.backward()
                my_step(d_op, lr_sched_d, global_step, train_len)


            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"] # modify weights of m and a for better convergence
                g_target_label_encoded = torch.full((batch_size, 1), 1, device=device).float()

                d_on_encoded_for_enc = discriminator(encoded)
                # target label for encoded images should be 'cover', because we want to fool the discriminator

                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)

                robust_sum_loss += lambda_a * g_loss_adv
                fragile_sum_loss += lambda_a * g_loss_adv
                sum_loss += lambda_a*g_loss_adv

            robust_sum_loss.backward()
            # fragile_sum_loss.backward()
        
            my_step(finetune_optim, lr_sched, global_step, train_len)
            # my_step(finetune_fragile_optim, lr_fragile_sched, global_step, train_len)

            print("curr_step = ", step, "show_circle = ", show_circle, "train_set = ", len(train_audio_loader))

            if step % show_circle == 0:
                # save generated audios
                save_audio_path = os.path.join("results/wm_speech", "attack:{}_epoch:{}.wav".format(train_config["attack_type"], ep))
                torchaudio.save(save_audio_path, src = encoded.detach().squeeze(1).to("cpu"), sample_rate = sample["trans_sr"])

                
                att_rec_r_msg, att_rec_f_msg = robust_decoded[0], fragile_decoded[0]
                rec_r_msg, rec_f_msg = robust_decoded[1], fragile_decoded[1]
            
                # decoder_acc = (decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()
                attack_decoder_acc = [((att_rec_r_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((att_rec_f_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                decoder_acc = [((rec_r_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((rec_f_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)

                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                
                norm2=mse_loss(wav_matrix.detach(),zero_tensor)
                print('-' * 100)
                print("step:{} - wav_loss:{:.8f} - robust_msg_loss:{:.8f} - fragile_msg_loss:{:.8f} - no_attack_acc:[{:.8f},{:.8f}] - attack_acc:[{:.8f},{:.8f}] - snr:{:.8f} - norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    step, losses[0], losses[1], losses[3], decoder_acc[0], decoder_acc[1],\
                    attack_decoder_acc[0], attack_decoder_acc[1],\
                    snr, norm2, sample["patch_num"].item(), sample["pad_num"].item(),\
                    wav_matrix.shape[2], d_loss_on_encoded, d_loss_on_cover))
                  
                wandb.log({'wav_loss': losses[0]})
                wandb.log({'robust_msg_loss': losses[1]})
                wandb.log({'fragile_msg_loss': losses[3]})
                wandb.log({'snr': snr})
                wandb.log({'d_loss_on_encoded': d_loss_on_encoded})
                wandb.log({'d_loss_on_cover': d_loss_on_cover})

                wandb.log({'train_data_attack_robust_msg_loss': losses[2]})
                wandb.log({'train_data_attack_fragile_msg_loss': losses[4]})

        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["finetune_ckpt"], 'attack_type_' + str(train_config["attack_type"]))
            else:
                path = os.path.join(train_config["path"]["finetune_ckpt"], 'attack_type_ab_' + str(train_config["attack_type"]))
            save_multi_decoder_finetune_op(path, ep, encoder, robust_decoder, fragile_decoder, discriminator, finetune_optim, finetune_optim, train_config["attack_type"])
            shutil.copyfile(os.path.realpath(__file__), os.path.join(path, os.path.basename(os.path.realpath(__file__)))) # save training scripts

        with torch.no_grad():
            encoder.eval()
            robust_decoder.eval()
            fragile_decoder.eval()
            discriminator.eval()
            wm_avg_acc = [0, 0]
            attack_wm_avg_acc = [0, 0]
            unwm_avg_acc = 0
            avg_snr = 0
            wm_avg_wav_loss = 0
            unwm_avg_wav_loss = 0
            wm_avg_r_msg_loss = 0
            wm_avg_f_msg_loss = 0
            unwm_avg_r_msg_loss = 0
            unwm_avg_f_msg_loss = 0
            avg_d_loss_on_encoded = 0
            avg_d_loss_on_cover = 0
            count = 0
            no_attack_r_msg_loss = 0
            no_attack_f_msg_loss = 0
            
            # @TODO save it as track
            for i, sample in enumerate(val_audios_loader):
                if count % 500 == 0:
                    print("count = ", count, "global_step = ", global_step)
                count += 1
                global_step += 1
                # ---------------------- build watermark

                msg = np.random.choice([0, 1], [batch_size, 1, int(msg_length)])
                wav_matrix = sample["matrix"].to(device)
                msg = torch.from_numpy(msg).float() * 2 - 1
                msg = msg.to(device)


                encoded, carrier_wateramrked = encoder(wav_matrix, msg, global_step)
                robust_decoded = robust_decoder(encoded, global_step, train_config["attack_type"])
                fragile_decoded = fragile_decoder(encoded, global_step, train_config["attack_type"])
                unwm_robust_decoded = robust_decoder(wav_matrix, global_step, train_config["attack_type"])
                unwm_fragile_decoded = fragile_decoder(wav_matrix, global_step, train_config["attack_type"])

                eval_wm_losses = loss.multi_de_one_wm_loss(wav_matrix, encoded, msg, robust_decoded[0], robust_decoded[1], fragile_decoded[0], fragile_decoded[1])
                eval_unwm_losses = loss.multi_de_one_wm_loss(wav_matrix, encoded, msg, unwm_robust_decoded[0], unwm_robust_decoded[1], unwm_fragile_decoded[0], unwm_fragile_decoded[1])
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

                att_rec_r_msg, att_rec_f_msg = robust_decoded[0], fragile_decoded[0]
                rec_r_msg, rec_f_msg = robust_decoded[1], fragile_decoded[1]

                # 在受到攻击后，鲁棒水印和脆弱水印的正确率
                attack_decoder_acc = [((att_rec_r_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((att_rec_f_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                # 没受到攻击时，鲁棒水印和脆弱水印的正确率
                wm_decoder_acc = [((rec_r_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((rec_f_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]

                # wm_decoder_acc = [((wm_decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((wm_decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                unwm_rec_msg = unwm_robust_decoded[1]
                unwm_decoder_acc = [((unwm_rec_msg >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)
                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                # norm2=mse_loss(wav_matrix.detach(),zero_tensor)


                attack_wm_avg_acc[0] += attack_decoder_acc[0]
                attack_wm_avg_acc[1] += attack_decoder_acc[1]
                wm_avg_acc[0] += wm_decoder_acc[0]
                wm_avg_acc[1] += wm_decoder_acc[1]
                unwm_avg_acc += unwm_decoder_acc[0]

                wm_avg_wav_loss += eval_wm_losses[0]
                no_attack_r_msg_loss += eval_wm_losses[1]
                wm_avg_r_msg_loss += eval_wm_losses[2]
                no_attack_f_msg_loss += eval_wm_losses[3]
                wm_avg_f_msg_loss += eval_wm_losses[4]

                unwm_avg_wav_loss += eval_unwm_losses[0]
                unwm_avg_r_msg_loss += eval_unwm_losses[1]
                unwm_avg_f_msg_loss += eval_unwm_losses[3]

                avg_d_loss_on_cover += d_loss_on_cover
                avg_d_loss_on_encoded += d_loss_on_encoded
                avg_snr += snr

            attack_wm_avg_acc[0] /= count
            attack_wm_avg_acc[1] /= count

            wm_avg_acc[0] /= count
            wm_avg_acc[1] /= count
            unwm_avg_acc /= count

            wm_avg_wav_loss /= count
            wm_avg_r_msg_loss /= count
            wm_avg_f_msg_loss /= count
            no_attack_r_msg_loss /= count
            no_attack_f_msg_loss /= count

            unwm_avg_wav_loss /= count
            unwm_avg_r_msg_loss /= count
            unwm_avg_f_msg_loss /= count

            avg_d_loss_on_encoded /= count
            avg_d_loss_on_cover /= count
            avg_snr /= count
            

            wandb.log({"attack_watermark_r_decoder_acc": attack_wm_avg_acc[0]})
            wandb.log({"attack_watermark_f_decoder_acc": attack_wm_avg_acc[1]})
            wandb.log({"watermark_r_decoder_acc": wm_avg_acc[0]})
            wandb.log({'watermark_f_decoder_acc': wm_avg_acc[1]})
            wandb.log({'valid_unwatermark_msg_loss': unwm_avg_acc})
            wandb.log({'attack_robust_msg_loss': wm_avg_r_msg_loss})
            wandb.log({'attack_fragile_msg_loss': wm_avg_f_msg_loss})
            wandb.log({'no_attack_robust_msg_loss': no_attack_r_msg_loss})
            wandb.log({'no_attack_fragile_msg_loss': no_attack_f_msg_loss})

            wandb.log({'no_wm_audio_robust_msg_loss': unwm_avg_r_msg_loss})
            wandb.log({'no_wm_audio_fragile_msg_loss': unwm_avg_f_msg_loss})
            
            
            print('#e' * 60)
            print("epoch:{} - wm_wav_loss:{:.8f} - unwm_wav_loss:{:.8f} - wm_msg_r_loss:{:.8f} - wm_msg_f_loss:{:.8f} - unwm_r_msg_loss:{:.8f} - unwm_f_msg_loss:{:.8f} - wm_acc:[{:.8f},{:.8f}] - unwm_acc:[{:.8f}] - snr:{:.8f} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    ep, wm_avg_wav_loss, unwm_avg_wav_loss, wm_avg_r_msg_loss, wm_avg_f_msg_loss, unwm_avg_r_msg_loss, unwm_avg_f_msg_loss, wm_avg_acc[0], wm_avg_acc[1], unwm_avg_acc, avg_snr, avg_d_loss_on_encoded, avg_d_loss_on_cover))




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

    finetune(args, configs)
