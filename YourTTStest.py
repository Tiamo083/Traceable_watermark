import torch
from TTS.api import TTS
import torchaudio
import pdb
# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())

speaker_embedding, sample_rate = torchaudio.load("/amax/home/Tiamo/Traceable_watermark/test_speech.wav")
# Example voice cloning with YourTTS in English, French and Portuguese
pdb.set_trace()
tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False).to(device)
tts.tts_to_file("This is voice cloning.", speaker_wav=speaker_embedding, language="en", file_path="output.wav")

