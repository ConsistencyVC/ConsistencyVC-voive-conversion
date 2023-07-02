from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC 
#from datasets import load_dataset
import torch
import soundfile as sf
from glob import glob
import glob2
from tqdm import tqdm
import matplotlib.pyplot as plt
# load model and processor
processor = Wav2Vec2Processor.from_pretrained("speech31/wav2vec2-large-english-TIMIT-phoneme_v3")
model = Wav2Vec2ForCTC.from_pretrained("speech31/wav2vec2-large-english-TIMIT-phoneme_v3").cuda().eval()
files=glob2.glob(r".\dataset\crosslingual_emo_dataset\**\*.wav")
files=sorted(files)
print(len(files))
for file in tqdm(files):
    ppg_file=file.replace(r".wav",r"_eng_ppg.pt")
    # Read and process the input
    audio_input, sample_rate = sf.read(file)
    #audio_input=audio_input[:25000]
    #sf.write('3752_slice.wav',audio_input,sample_rate)
    inputs = processor(audio_input, sampling_rate=16_000, return_tensors="pt", padding=True)
    #print(inputs)
    #import matplotlib.pyplot as plt
    with torch.no_grad():
        logits = model(inputs.input_values.cuda()).logits
        logits_trans=logits.transpose(2,1)
        #print(logits_trans.size())
        #print(file,ppg_file)

        torch.save(logits_trans.cpu(),ppg_file)

        #lo=logits.squeeze(0).numpy()
        #print(lo)
        #lo=lo.transpose(1,0)
        #plt.imshow(lo)
        #plt.show()
        #hidden_states=model(inputs.input_values).hidden_states
        #print(hidden_states)
        #hidden=hidden_states.squeeze(0).numpy()
        #print(hidden)
        #plt.imshow(hidden)
        #plt.show()
    # Decode id into string
    #print(logits)
    ##predicted_ids = torch.argmax(logits, axis=-1)
    ##print(predicted_ids)      
    ##predicted_sentences = processor.batch_decode(predicted_ids)
    ##print(predicted_sentences)
