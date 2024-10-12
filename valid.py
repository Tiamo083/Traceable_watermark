import os
import torch
import yaml
import logging
import argparse
import numpy as np
import soundfile as sf

# import wandb
from librosa.filters import mel as librosa_mel_fn
from torch.utils.data import DataLoader
from model.loss import Loss_identity
from torch.nn.functional import mse_loss
import random
import wandb
import socket


from My_model.modules import Encoder, Decoder
from dataset.data import wav_dataset as used_dataset

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

    # -------------- get train dataset 
    val_audios = used_dataset(process_config=process_config, train_config=valid_config, flag='test')
    
    batch_size = valid_config["optimize"]["batch_size"]
    val_audios_loader = DataLoader(val_audios, batch_size=batch_size, shuffle = False)

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
        model_path = os.path.join(path_model, model_list[index])
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
            if count % 10 == 0:
                print("count = ", count, "global_step = ", global_step)
                if count != 0:
                    print("attack_watermark_r_decoder_acc: ", attack_wm_avg_acc[0] / count)
                    print("attack_watermark_f_decoder_acc: ", attack_wm_avg_acc[1] / count)
                    print("watermark_r_decoder_acc: ", wm_avg_acc[0] / count)
                    print("watermark_f_decoder_acc: ", wm_avg_acc[1] / count)
                    print("valid_unwatermark_msg_loss: ", unwm_avg_acc[0] / count)
                    wandb.log({"attack_watermark_r_decoder_acc": attack_wm_avg_acc[0] / count})
                    wandb.log({"attack_watermark_f_decoder_acc": attack_wm_avg_acc[1] / count})
                    wandb.log({"watermark_r_decoder_acc": wm_avg_acc[0] / count})
                    wandb.log({'watermark_f_decoder_acc': wm_avg_acc[1] / count})
                    wandb.log({'valid_unwatermark_msg_loss': unwm_avg_acc[0] / count})
            count += 1
            global_step += 1
            # ---------------------- build watermark
            msg = np.random.choice([0, 1], [batch_size, 1, msg_length])
            msg = torch.from_numpy(msg).float()*2 - 1
            wav_matrix = sample["matrix"].to(device)
            msg = msg.to(device)
            encoded, carrier_watermarked = encoder(wav_matrix, msg, global_step)
            sf.write("results/before-vc.wav", encoded.squeeze(0).squeeze(0).cpu(), 22050)
            wm_decoded = decoder(encoded, global_step, valid_config["attack_type"]) # 嵌入水印的音频
            unwm_decoded = decoder(wav_matrix, global_step, valid_config["attack_type"]) # 没有嵌入水印的音频
            wm_losses = loss.half_en_de_loss(wav_matrix, encoded, msg, wm_decoded[0], wm_decoded[1])
            unwm_losses = loss.half_en_de_loss(wav_matrix, wav_matrix, msg, unwm_decoded[0], unwm_decoded[1])
            # wm_losses = loss.en_de_loss(wav_matrix, encoded, msg, wm_decoded)
            # unwm_losses = loss.en_de_loss(wav_matrix, encoded, msg, unwm_decoded)

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
    valid_config = yaml.load(open(args.valid_config, "r"), Loader=yaml.FullLoader)
    configs = (process_config, model_config, valid_config)

    main(args, configs)