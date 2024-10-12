import os
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

"""
修改于12.20，目标在于将水印分为两个部分：
1、分布（鲁棒）
2、比特串（脆弱）
"""

# 创建水印，选择一个分布来生成水印
def generate_watermark(batch_size, length, micro, sigma):
    sigmoid = torch.nn.Sigmoid()
    eye_matrix = np.eye(length)
    mask_convariance_maxtix = eye_matrix * (sigma ** 2)
    center = np.ones(length) * micro

    w_bit = multivariate_normal.rvs(mean = center, cov = mask_convariance_maxtix, size = [batch_size, 1])
    w_bit = torch.from_numpy(w_bit).float()
    return w_bit


def DiffVC_model(source_path, target_path):
    import sys
    use_gpu = torch.cuda.is_available()
    mel_basis = librosa_mel_fn(sr = 22050, n_fft = 1024, n_mels = 80, fmin = 0, fmax = 8000)

    pdb.set_trace()
    sys.path.append('Speech-Backbones/DiffVC/')
    import params
    from VCmodel import DiffVC

    import sys
    sys.path.append('Speech-Backbones/DiffVC/hifi-gan/')
    from env import AttrDict
    from models import Generator as HiFiGAN

    sys.path.append('Speech-Backbones/DiffVC/speaker_encoder/')
    from encoder import inference as spk_encoder
    from pathlib import Path

    def get_mel(wav_path):
        wav, _ = load(wav_path, sr=22050)
        wav = wav[:(wav.shape[0] // 256)*256]
        wav = np.pad(wav, 384, mode='reflect')
        stft = librosa.core.stft(y = wav, n_fft=1024, hop_length=256, win_length=1024, window='hann', center=False)
        stftm = np.sqrt(np.real(stft) ** 2 + np.imag(stft) ** 2 + (1e-9))
        mel_spectrogram = np.matmul(mel_basis, stftm)
        log_mel_spectrogram = np.log(np.clip(mel_spectrogram, a_min=1e-5, a_max=None))
        return log_mel_spectrogram

    def get_embed(wav_path):
        wav_preprocessed = spk_encoder.preprocess_wav(wav_path)
        embed = spk_encoder.embed_utterance(wav_preprocessed)
        return embed

    def noise_median_smoothing(x, w=5):
        y = np.copy(x)
        x = np.pad(x, w, "edge")
        for i in range(y.shape[0]):
            med = np.median(x[i:i+2*w+1])
            y[i] = min(x[i+w+1], med)
        return y

    def mel_spectral_subtraction(mel_synth, mel_source, spectral_floor=0.02, silence_window=5, smoothing_window=5):
        mel_len = mel_source.shape[-1]
        energy_min = 100000.0
        i_min = 0
        for i in range(mel_len - silence_window):
            energy_cur = np.sum(np.exp(2.0 * mel_source[:, i:i+silence_window]))
            if energy_cur < energy_min:
                i_min = i
                energy_min = energy_cur
        estimated_noise_energy = np.min(np.exp(2.0 * mel_synth[:, i_min:i_min+silence_window]), axis=-1)
        if smoothing_window is not None:
            estimated_noise_energy = noise_median_smoothing(estimated_noise_energy, smoothing_window)
        mel_denoised = np.copy(mel_synth)
        for i in range(mel_len):
            signal_subtract_noise = np.exp(2.0 * mel_synth[:, i]) - estimated_noise_energy
            estimated_signal_energy = np.maximum(signal_subtract_noise, spectral_floor * estimated_noise_energy)
            mel_denoised[:, i] = np.log(np.sqrt(estimated_signal_energy))
        return mel_denoised

    # loading voice conversion model
    vc_path = 'Speech-Backbones/DiffVC/checkpts/vc/vc_vctk_wodyn.pt' # path to voice conversion model

    generator = DiffVC(params.n_mels, params.channels, params.filters, params.heads,
                    params.layers, params.kernel, params.dropout, params.window_size,
                    params.enc_dim, params.spk_dim, params.use_ref_t, params.dec_dim,
                    params.beta_min, params.beta_max)
    if use_gpu:
        generator = generator.cuda()
        generator.load_state_dict(torch.load(vc_path))
    else:
        generator.load_state_dict(torch.load(vc_path, map_location='cpu'))
    generator.eval()

    print(f'Number of parameters: {generator.nparams}')

    # loading HiFi-GAN vocoder
    hfg_path = 'Speech-Backbones/DiffVC/checkpts/vocoder/' # HiFi-GAN path

    with open(hfg_path + 'config.json') as f:
        h = AttrDict(json.load(f))

    if use_gpu:
        hifigan_universal = HiFiGAN(h).cuda()
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator')['generator'])
    else:
        hifigan_universal = HiFiGAN(h)
        hifigan_universal.load_state_dict(torch.load(hfg_path + 'generator',  map_location='cpu')['generator'])

    _ = hifigan_universal.eval()
    hifigan_universal.remove_weight_norm()

    # loading speaker encoder
    enc_model_fpath = Path('Speech-Backbones/DiffVC/checkpts/spk_encoder/pretrained.pt') # speaker encoder path
    if use_gpu:
        spk_encoder.load_model(enc_model_fpath, device="cuda")
    else:
        spk_encoder.load_model(enc_model_fpath, device="cpu")

    # loading source and reference wavs, calculating mel-spectrograms and speaker embeddings
    src_path = 'Speech-Backbones/DiffVC/example/6415_111615_000012_000005.wav' # path to source utterance
    tgt_path = 'Speech-Backbones/DiffVC/example/8534_216567_000015_000010.wav' # path to reference utterance

    mel_source = torch.from_numpy(get_mel(src_path)).float().unsqueeze(0)
    if use_gpu:
        mel_source = mel_source.cuda()
    mel_source_lengths = torch.LongTensor([mel_source.shape[-1]])
    if use_gpu:
        mel_source_lengths = mel_source_lengths.cuda()

    mel_target = torch.from_numpy(get_mel(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        mel_target = mel_target.cuda()
    mel_target_lengths = torch.LongTensor([mel_target.shape[-1]])
    if use_gpu:
        mel_target_lengths = mel_target_lengths.cuda()

    embed_target = torch.from_numpy(get_embed(tgt_path)).float().unsqueeze(0)
    if use_gpu:
        embed_target = embed_target.cuda()

    # performing voice conversion
    mel_encoded, mel_ = generator.forward(mel_source, mel_source_lengths, mel_target, mel_target_lengths, embed_target,
                                        n_timesteps=30, mode='ml')
    mel_synth_np = mel_.cpu().detach().squeeze().numpy()
    mel_source_np = mel_.cpu().detach().squeeze().numpy()
    mel = torch.from_numpy(mel_spectral_subtraction(mel_synth_np, mel_source_np, smoothing_window=1)).float().unsqueeze(0)
    if use_gpu:
        mel = mel.cuda()

    with torch.no_grad():
        source_audio = hifigan_universal.forward(mel_source).cpu().squeeze().clamp(-1, 1)
        source_audio = hifigan_universal.forward(mel_target).cpu().squeeze().clamp(-1, 1)
        conversed_audio = hifigan_universal.forward(mel).cpu().squeeze().clamp(-1, 1)


def main(args, configs):
    logging.info('main function')
    process_config, model_config, train_config = configs
    pre_step = 0
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
    # ---------------- get train dataset
    audios = my_dataset(process_config=process_config, train_config=train_config, flag='train')
    val_audios = my_dataset(process_config=process_config, train_config=train_config, flag='val')

    batch_size = train_config["optimize"]["batch_size"]
    assert batch_size < len(audios)
    audios_loader = DataLoader(audios, batch_size=batch_size, shuffle=True)
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle=False)
    # ---------------- build model
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
    if model_config["structure"]["share"]:
        if model_config["structure"]["transformer"]:
            decoder.msg_decoder = encoder.encoder
        else:
            decoder.wav_encoder = encoder.wav_encoder
    # ---------------- optimizer -----------
    # en_de_op = Adam([
    #     {'params': decoder.parameters(), 'lr': args.lr_en_de},
    #     {'params': encoder.parameters(), 'lr': args.lr_en_de}
	# ])
    # en_de_op = ScheduledOptimMain(encoder, decoder, train_config, model_config, args.restore_step)
    en_de_op = Adam(
            params=chain(decoder.parameters(), encoder.parameters()),
            betas=train_config["optimize"]["betas"],
            eps=train_config["optimize"]["eps"],
            weight_decay=train_config["optimize"]["weight_decay"],
            lr = train_config["optimize"]["lr"]
        )
    lr_sched = StepLR(en_de_op, step_size=train_config["optimize"]["step_size"], gamma=train_config["optimize"]["gamma"])

    # ---------------- Loss
    loss = Loss_identity(train_config=train_config)
    # ---------------- Init logger
    for p in train_config["path"].values():
        os.makedirs(p, exist_ok=True)
    train_log_path = os.path.join(train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    # train_logger = SummaryWriter(train_log_path)
    # val_logger = SummaryWriter(val_log_path)

    # ---------------- train
    logging.info(logging_mark + "\t" + "Begin Trainging" + "\t" + logging_mark)
    epoch_num = train_config["iter"]["epoch"]
    save_circle = train_config["iter"]["save_circle"]
    show_circle = train_config["iter"]["show_circle"]
    lambda_e = train_config["optimize"]["lambda_e"]
    lambda_m = train_config["optimize"]["lambda_m"]
    global_step = 0
    train_len = len(audios_loader)
    for ep in range(1, epoch_num+1):
        encoder.train()
        decoder.train()
        discriminator.train()
        step = 0
        logging.info('Epoch {}/{}'.format(ep, epoch_num))
        # for sample in track(audios_loader):
        for i, sample in enumerate(audios_loader):
            global_step += 1
            step += 1
            # ---------------- build watermark
            distribution_msg = generate_watermark(batch_size, msg_length, train_config["micro"], train_config["sigma"])
            distribution_msg = distribution_msg.unsqueeze(0).unsqueeze(1)
            distribution_msg = distribution_msg.to(device)


            msg = np.random.choice([0,1], [batch_size, 1, msg_length])
            msg = torch.from_numpy(msg).float()*2 - 1
            # 把msg从0, 1变成-1，1
            wav_matrix = sample["matrix"].to(device)
            msg = msg.to(device)
            encoded, carrier_wateramrked = encoder(wav_matrix, msg, global_step)
            decoded = decoder(encoded, global_step, train_config["attack_type"])
            losses = loss.en_de_loss(wav_matrix, encoded, msg, decoded)


            distribution_encoded, _ = encoder(wav_matrix, distribution_msg, global_step)
            distribution_decoded = decoder(distribution_encoded, global_step, 1)
            distribution_losses = loss.en_de_loss(wav_matrix, distribution_encoded, distribution_msg, distribution_decoded)

            if global_step < pre_step:
                sum_loss = lambda_m*losses[1]
                distrib_loss = lambda_m*distribution_losses[1]
            else:
                sum_loss = lambda_e*losses[0] + lambda_m*losses[1]
                distrib_loss = lambda_e*distribution_losses[0] + lambda_m*distribution_losses[1]

            # adv
            if train_config["adv"]:
                lambda_a = lambda_m = train_config["optimize"]["lambda_a"] # modify weights of m and a for better convergence
                g_target_label_encoded = torch.full((batch_size, 1), 1, device=device).float()

                d_on_encoded_for_enc = discriminator(encoded)
                d_on_encoded_for_enc_distrib = discriminator(distribution_encoded)
                # target label for encoded images should be 'cover', because we want to fool the discriminator

                g_loss_adv = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc, g_target_label_encoded)
                g_loss_adv_distrib = F.binary_cross_entropy_with_logits(d_on_encoded_for_enc_distrib, g_target_label_encoded)

                sum_loss += lambda_a*g_loss_adv
                distrib_loss = lambda_a*g_loss_adv_distrib

            sum_loss.backward()
            distrib_loss.backward()

            my_step(en_de_op, lr_sched, global_step, train_len)

            if train_config["adv"]:
                d_target_label_cover = torch.full((batch_size, 1), 1, device=device).float()
                d_on_cover = discriminator(wav_matrix)
                d_loss_on_cover = F.binary_cross_entropy_with_logits(d_on_cover, d_target_label_cover)
                d_loss_on_cover.backward()

                d_target_label_encoded = torch.full((batch_size, 1), 0, device=device).float()
                d_on_encoded = discriminator(encoded.detach())
                d_on_encoded_distrib = discriminator(distribution_encoded.detach())

                # target label for encoded images should be 'encoded', because we want discriminator fight with encoder
                d_loss_on_encoded = F.binary_cross_entropy_with_logits(d_on_encoded, d_target_label_encoded)
                d_loss_on_encoded_distrib = F.binary_cross_entropy_with_logits(d_on_encoded_distrib, d_target_label_encoded)

                d_loss_on_encoded.backward()
                d_loss_on_encoded_distrib.backward()
                my_step(d_op, lr_sched_d, global_step, train_len)

            if step % show_circle == 0:
                save_audio_path = os.path.join("results/wm_speech", "attack:{}_epoch:{}_step:{}.wav".format(train_config["attack_type"], ep, step))
                torchaudio.save(save_audio_path, src = encoded.detach().squeeze(1).to("cpu"), sample_rate = sample["trans_sr"])

                # decoder_acc = (decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()
                decoder_acc = [((decoded[0] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item(), ((decoded[1] >= 0).eq(msg >= 0).sum().float() / msg.numel()).item()]
                zero_tensor = torch.zeros(wav_matrix.shape).to(device)

                snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), encoded.detach()))
                distribution_snr = 10 * torch.log10(mse_loss(wav_matrix.detach(), zero_tensor) / mse_loss(wav_matrix.detach(), distribution_encoded.detach()))

                norm2=mse_loss(wav_matrix.detach(),zero_tensor)
                logging.info('-' * 100)
                logging.info("step:{} - wav_loss:{:.8f} - msg_loss:{:.8f} - acc:[{:.8f},{:.8f}] - snr:{:.8f} - norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} - d_loss_on_encoded:{} - d_loss_on_cover:{}".format(\
                    step, losses[0], losses[1], decoder_acc[0], decoder_acc[1],\
                    snr, norm2, sample["patch_num"].item(), sample["pad_num"].item(),\
                    wav_matrix.shape[2], d_loss_on_encoded, d_loss_on_cover))
                

                print("step:{} - distrib_wav_loss:{:.8f} - distrib_msg_loss:{:.8f} - distrib_snr:{:.8f} - norm:{:.8f} - patch_num:{} - pad_num:{} - wav_len:{} - d_loss_on_encoded_distrib:{} - d_loss_on_cover_distrib:{}".format(\
                    step, distribution_losses[0], distribution_losses[1],\
                    distribution_snr, norm2, sample["patch_num"].item(), sample["pad_num"].item(),\
                    wav_matrix.shape[2], d_loss_on_encoded_distrib, d_loss_on_cover))
                

                """wandb.log({'distrib_wav_loss': distribution_losses[0]})
                wandb.log({'distrib_msg_loss': distribution_losses[1]})
                wandb.log({'distrib_snr': distribution_snr})
                wandb.log({'wav_len': wav_matrix.shape[2]})
                wandb.log({'d_loss_on_encoded_distrib': d_loss_on_encoded_distrib})
                wandb.log({'d_loss_on_cover_distrib': d_loss_on_cover})"""

        # if ep % save_circle == 0 or ep == 1 or ep == 2:
        if ep % save_circle == 0:
            if not model_config["structure"]["ab"]:
                path = os.path.join(train_config["path"]["ckpt"], 'pth')
            else:
                path = os.path.join(train_config["path"]["ckpt"], 'pth_ab')
            save_op(path, ep, encoder, decoder, discriminator, en_de_op, train_config["attack_type"])
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
            unwm_avg_acc = [0, 0]
            avg_snr = 0
            wm_avg_wav_loss = 0
            unwm_avg_wav_loss = 0
            wm_avg_msg_loss = 0
            unwm_avg_msg_loss = 0
            avg_d_loss_on_encoded = 0
            avg_d_loss_on_cover = 0
            count = 0
            for sample in track(val_audios_loader):
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
    """wandb.init(config=args,
               project=args.project_name,
               entity=args.team_name,
               notes=socket.gethostname(),
               name=args.experiment_name+"_"+str(args.seed),
               group=args.scenario_name,
               job_type="training",
               reinit=True)"""


    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, train_config)

    main(args, configs)
