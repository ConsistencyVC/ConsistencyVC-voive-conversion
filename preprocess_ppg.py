import os
import numpy as np
import argparse
import torch
from glob import glob
from tqdm import tqdm
from whisper.model import Whisper, ModelDimensions
from whisper.audio import load_audio, pad_or_trim, log_mel_spectrogram
import librosa
import soundfile as sf
def load_model(path) -> Whisper:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=device)
    dims = ModelDimensions(**checkpoint["dims"])
    model = Whisper(dims)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def pred_ppg(whisper: Whisper, wavPath, ppgPath):
    audio, sr = librosa.load(wavPath,sr=None)
    if len(audio) >= sr * 29:
        print(wavPath,"cut to 29s")
        audio = audio[:sr * 29]
        #librosa.output.write_wav("your_audio_file.wav", audio, sr)
        sf.write(wavPath, audio, sr)
    audio = load_audio(wavPath)
    audln = audio.shape[0]
    ppgln = audln // 320
    # audio = pad_or_trim(audio)
    mel = log_mel_spectrogram(audio).to(whisper.device)
    with torch.no_grad():
        ppg = whisper.encoder(mel.unsqueeze(0)).squeeze().data.cpu().float().numpy()
        
        if ppgln>ppg.shape[0]:
            print("ppgln>ppg.shape[0]")
        ppg = ppg[:ppgln,] # [length, dim=1024]
        #if audln // 320<ppg.shape[0]:
        #    print("audln // 320<ppg.shape[0]")
        np.save(ppgPath, ppg, allow_pickle=False)


if __name__ == "__main__":
    # 读取所有 .wav 文件
    data_dir = r".\dataset\crosslingual_emo_dataset\jvs"
    wav_files = glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    wav_files=sorted(wav_files)
    whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))

    for wav in tqdm(wav_files):
        ppg_path=wav.replace(r".wav",r"whisper.pt.npy")
        #print(wav,ppg_path)
        if not os.path.exists(ppg_path):
            pred_ppg(whisper, wav, ppg_path)
