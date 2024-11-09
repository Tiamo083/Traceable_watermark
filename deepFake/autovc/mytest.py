import sys
import torchaudio as ta

from converision import get_vc_spect
from make_metadata import get_emb
from make_spect import get_spect, get_spect_from_wav

tgt_path = '/amax/home/zhaoxd/half_vulnerable/autovc/wavs/p225/p225_003.wav'
src_path = '/amax/home/zhaoxd/half_vulnerable/autovc/wavs/p226/p226_008.wav'
tgt_audio, sr = ta.load(tgt_path)
tgt_audio = tgt_audio.unsqueeze(0) # sr: 16000
device = 'cuda:0'


src_spectrum = get_spect(src_path)
src_embedding = get_emb(src_spectrum)

tgt_audio = tgt_audio.cpu().squeeze(0).numpy()
if tgt_audio.shape[0] == 1:
    tgt_audio = tgt_audio.squeeze(0)
tgt_spectrum = get_spect_from_wav(tgt_audio)
tgt_embdedding = get_emb(tgt_spectrum)

vc_spectrum = get_vc_spect(src_spectrum, src_embedding, tgt_embdedding, device)

sys.path.append('deepFake/autovc/hifi_gan')
from hifi_gan.inference_e2e import inference_from_spec

audio = inference_from_spec(vc_spectrum.T, device)
ta.save('vc.wav', audio, sr)
