import os
import argparse
import torch
import librosa
import time
from scipy.io.wavfile import write
from tqdm import tqdm
import soundfile as sf
import utils
from models import SynthesizerTrn
from mel_processing import mel_spectrogram_torch
import logging
logging.getLogger('numba').setLevel(logging.WARNING)
import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.
from preprocess_ppg import pred_ppg_c,load_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--hpfile", type=str, default="ConsistencyVC-voive-conversion/logs/config.json", help="path to json config file")
    parser.add_argument("--ptfile", type=str, default="ConsistencyVC-voive-conversion/logs/G_cvc-whispers-three-emo-loss.pth", help="path to pth file")
    parser.add_argument("--outdir", type=str, default="output/long", help="path to output dir")
    parser.add_argument("--use_timestamp", default=False, action="store_true")
    args = parser.parse_args()
    
    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda()
    _ = net_g.eval()
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None, True)


    
    src_wavs=[r"longaudio1.wav",
             r"longaudio2.wav"]
    

    tgt_wavs=["tgt_slice/20230712-092103-296_1.wav"]
    print("Processing text...")
    titles, srcs, tgts = [], [], []
    for src_wav in src_wavs:
        
        for tgt_wav in tgt_wavs:
            src_wav_name=os.path.basename(src_wav)[:-4]
            tgt_wav_name=os.path.basename(tgt_wav)[:-4]
            title="{}_to_{}".format(src_wav_name,tgt_wav_name)
            titles.append(title)
            srcs.append(src_wav)
            tgts.append(tgt_wav)
    #print(srcs)
    #print(tgts)
    #print(titles)
    #import sys
    #sys.exit()
    """
    with open(args.txtpath, "r") as f:
        for rawline in f.readlines():
            print(rawline)
            title, src, tgt = rawline.strip().split("|")
            titles.append(title)
            srcs.append(src)
            tgts.append(tgt)
    """

    print("Synthesizing...")
    with torch.no_grad():
        for line in tqdm(zip(titles, srcs, tgts)):
            title, src, tgt = line
            srcname,tgtname=title.split("to")
            # tgt
            wav_tgt, _ = librosa.load(tgt, sr=hps.data.sampling_rate)
            sf.write(os.path.join(args.outdir, f"{tgtname}.wav"), wav_tgt, hps.data.sampling_rate)
            #wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
     
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).cuda()
            mel_tgt = mel_spectrogram_torch(
                wav_tgt, 
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            # src
            audio, sr = librosa.load(src, sr=hps.data.sampling_rate)
            import numpy as np
            audio_result_sum=np.float32(np.zeros(len(audio)))
            #audio, sr = librosa.load(src_wav,sr=None)
            audiolen=audio.shape[0]
            print(audiolen)
            src_wav_wavs=[]
            num=int(audiolen/(sr*25))
            print(num)
            whisper = load_model(os.path.join("whisper_pretrain", "medium.pt"))
            for i in range(0,num+1):

                #print(i*20*sr,(i*20*sr+25*sr))
                tmp=audio[i*20*sr:(i*20*sr+25*sr)]
                sf.write(os.path.join(args.outdir, f"{srcname}_{i}.wav"), tmp, hps.data.sampling_rate)
        
                
                c=pred_ppg_c(whisper,os.path.join(args.outdir, f"{srcname}_{i}.wav"))#torch.from_numpy(np.load(c_filename))
                c=torch.from_numpy(c)
                c=c.transpose(1,0)
                c=c.unsqueeze(0)

                #print(c.size(),mel_tgt.size())
                audio_result = net_g.infer(c.cuda(), mel=mel_tgt)
                audio_result = audio_result[0][0].data.cpu().float().numpy()
                audio_result_sum[i*20*sr:(i*20*sr+audio_result.shape[0])]=audio_result
                #print(audio_result.dtype)
                #print(audio_result_sum.dtype)
                """
                if args.use_timestamp:
                    timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
                    write(os.path.join(args.outdir, "{}.wav".format(timestamp+"_"+title)), hps.data.sampling_rate, audio_result)
                else:
                    write(os.path.join(args.outdir, f"{title}_{i}.wav"), hps.data.sampling_rate, audio_result)
                """
            write(os.path.join(args.outdir, f"{title}_sum.wav"), hps.data.sampling_rate, audio_result_sum)
