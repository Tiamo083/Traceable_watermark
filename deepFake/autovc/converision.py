import os
import pickle
import torch
import numpy as np
from math import ceil
from model_vc import Generator


def pad_seq(x, base=32):
    len_out = int(base * ceil(float(x.shape[0])/base))
    len_pad = len_out - x.shape[0]
    assert len_pad >= 0
    return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

# metadata = pickle.load(open('metadata.pkl', "rb"))

# spect_vc = []

# for sbmt_i in metadata:
             
#     x_org = sbmt_i[2]
#     x_org, len_pad = pad_seq(x_org)
#     uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
#     emb_org = torch.from_numpy(sbmt_i[1][np.newaxis, :]).to(device)
    
#     for sbmt_j in metadata:
                   
#         emb_trg = torch.from_numpy(sbmt_j[1][np.newaxis, :]).to(device)
        
#         with torch.no_grad():
#             _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)
            
#         if len_pad == 0:
#             uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
#         else:
#             uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
        
#         spect_vc.append( ('{}x{}'.format(sbmt_i[0], sbmt_j[0]), uttr_trg) )
        
        
# with open('results.pkl', 'wb') as handle:
#     pickle.dump(spect_vc, handle)          

# 这里 org 应该是提供文字的人，trg 是提供声音的人
def get_vc_spect(src_spectrum, src_embedding, tgt_embedding, device):
    G = Generator(32,256,512,32).eval().to(device)

    g_checkpoint = torch.load('deepFake/autovc/pretrained_models/autovc.ckpt', map_location=device)
    G.load_state_dict(g_checkpoint['model'])
    x_org, len_pad = pad_seq(src_spectrum)
    uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
    emb_org = torch.from_numpy(src_embedding[np.newaxis, :]).to(device)
    emb_trg = torch.from_numpy(tgt_embedding[np.newaxis, :]).to(device)

    with torch.no_grad():
        _, x_identic_psnt, _ = G(uttr_org, emb_org, emb_trg)

    if len_pad == 0:
        uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
    else:
        uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()
    return uttr_trg