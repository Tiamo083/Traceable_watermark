import os
import torch
import yaml
import logging
import argparse
import numpy as np
import soundfile as sf
import torchaudio as ta

# import wandb
from librosa.filters import mel as librosa_mel_fn
from torch.utils.data import DataLoader
from model.loss import Loss_identity
from torch.nn.functional import mse_loss
import random
import socket

from distortions.dl import distortion
from My_model.modules import Encoder, Decoder

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
    process_config, model_config, valid_config = configs

    # -------------- build model
    win_dim = process_config["audio"]["win_len"]
    embedding_dim = model_config["dim"]["embedding"]
    nlayers_encoder = model_config["layer"]["nlayers_encoder"]
    nlayers_decoder = model_config["layer"]["nlayers_decoder"]
    attention_heads_encoder = model_config["layer"]["attention_heads_encoder"]
    attention_heads_decoder = model_config["layer"]["attention_heads_decoder"]
    # msg_length:16bit, the last 6 bits are used to clarify robust watermark and fragile watermark
    msg_length = valid_config["watermark"]["length"]

    encoder = Encoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_encoder=nlayers_encoder, attention_heads=attention_heads_encoder).to(device)
    decoder = Decoder(process_config, model_config, msg_length, win_dim, embedding_dim, nlayers_decoder=nlayers_decoder, attention_heads=attention_heads_decoder).to(device)

    # shared parameters
    path_model = model_config["test"]["model_path"]
    model_name = model_config["test"]["model_name"]
    if model_name:
        model = torch.load(os.path.join(path_model, model_name))
        logging.info("model <<{}>> loadded".format(model_name))
    else:
        index = model_config["test"]["index"]
        model_list = os.listdir(path_model)
        model_list = sorted(model_list,key=lambda x:os.path.getmtime(os.path.join(path_model,x)))
        model_path = "/amax/home/Tiamo/half_vulnerabe/results/ckpt/pth/none-conv228_ep_20_2024-06-23_15_23_33.pth"
        # model = torch.load(model_path,map_location=torch.device('cpu'))
        model = torch.load(model_path)
        logging.info("model <<{}>> loadded".format(model_path))
        print("\n\nmodel <<{}>> loadded".format(model_path))
    # encoder = model["encoder"]
    # decoder = model["decoder"]
    encoder.load_state_dict(model["encoder"])
    decoder.load_state_dict(model["decoder"],strict=False)

    # -------------- Loss 
    # use MSELoss to evaluate msg_loss and embedding_loss
    loss = Loss_identity(train_config=valid_config)

    # ---------------- train
    print(logging_mark + "\t" + "Begin Trainging" + "\t" + logging_mark)
    global_step = 0
    ep = 1
    # Validation Stage
    with torch.no_grad():
        encoder.eval()
        decoder.eval()
        

        # ---------------------- build watermark
        wav_matrix, sr = ta.load("/amax/home/Tiamo/Traceable_watermark/dataset/LibriSpeech_wav/test/61-70968-0000.wav")
        resampler = ta.transforms.Resample(orig_freq=sr, new_freq=22050)
        wav_matrix = resampler(wav_matrix)
        # msg = torch.tensor([0, 0, 1, 0, 1, 1, 1, 0,\
        #                     0, 1, 0, 1, 1, 1, 0, 0,\
        #                     1, 1, 0, 1, 0, 0, 0, 1,\
        #                     1, 0, 0, 0, 1, 0, 1, 0])
        msg = np.random.choice([0, 1], [1, 1, msg_length])
        print("鲁棒部分:", msg[0, 0, :msg_length//2])
        print("脆弱部分:", msg[0, 0, msg_length//2:])

        msg = torch.from_numpy(msg).float()*2 - 1
        wav_matrix = wav_matrix.unsqueeze(0).to(device)
        msg = msg.to(device)
        encoded, carrier_watermarked = encoder(wav_matrix, msg, global_step)
        ta.save("demo/嵌入水印后的音频.wav", encoded[0].cpu(), 22050)

        # 把音频通过autovc进行变换
        dl = distortion(process_config)
        audio_source = encoded.clone()
        audio_vced = dl(audio_source, attack_choice = 28, ratio = 10, src_path = '/amax/home/Tiamo/Traceable_watermark/autovc/wavs/p225/p225_011.wav')
        ta.save("demo/受到VC攻击后的音频.wav", audio_vced[0].cpu(), 22050)

        # 提取水印
        wm_decoded = decoder(encoded, global_step, valid_config["attack_type"]) # 嵌入水印的音频

        robust_msg, fragile_msg = torch.chunk(input = msg, chunks = 2, dim = 2)
        att_rec_r_msg, att_rec_f_msg = torch.chunk(input = wm_decoded[0], chunks = 2, dim = 2)
        print("受到攻击后:")
        print("鲁棒部分:", att_rec_r_msg[0, 0, :msg_length//2])
        print("脆弱部分:", att_rec_f_msg[0, 0, msg_length//2:])
        rec_r_msg, rec_f_msg = torch.chunk(input = wm_decoded[1], chunks = 2, dim = 2)
        print("没有受到攻击:")
        print("鲁棒部分:", rec_r_msg[0, 0, :msg_length//2])
        print("脆弱部分:", rec_f_msg[0, 0, msg_length//2:])

        # 在受到攻击后，鲁棒水印和脆弱水印的正确率
        attack_decoder_acc = [((att_rec_r_msg >= 0).eq(robust_msg >= 0).sum().float() / robust_msg.numel()).item(), ((att_rec_f_msg >= 0).eq(fragile_msg >= 0).sum().float() / fragile_msg.numel()).item()]
        print("在受到攻击后，鲁棒水印和脆弱水印的正确率:", attack_decoder_acc)
        # 没受到攻击时，鲁棒水印和脆弱水印的正确率
        wm_decoder_acc = [((rec_r_msg >= 0).eq(robust_msg >= 0).sum().float() / robust_msg.numel()).item(), ((rec_f_msg >= 0).eq(fragile_msg >= 0).sum().float() / fragile_msg.numel()).item()]
        print("没有受到攻击时，鲁棒水印和脆弱水印的正确率:", wm_decoder_acc)


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
        "-t", "--valid_config", type=str, required=True, help="path to valid.yaml"
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
    

    # Read Config
    process_config = yaml.load(
        open(args.process_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    valid_config = yaml.load(open(args.valid_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, valid_config)

    main(args, configs)