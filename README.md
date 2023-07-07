# ConsistencyVC-voive-conversion

## Using joint training speaker encoder with consistency loss to achieve cross-lingual voice conversion and expressive voice conversion

Demo page: https://consistencyvc.github.io/ConsistencyVC-demo-page

The whisper medium model can be downloaded here: https://drive.google.com/file/d/1PZsfQg3PUZuu1k6nHvavd6OcOB_8m1Aa/view?usp=drive_link

The pre-trained models are available here:https://drive.google.com/drive/folders/1KvMN1V8BWCzJd-N8hfyP283rLQBKIbig?usp=sharing


<!-- 科研好累。 -->

# Inference with the pre-trained models

Use whisperconvert_exp.py to achieve voice conversion using weo as content information.

Use ppgemoconvert_exp.py to achieve voice conversion using ppg as content information.

# Train models by your dataset

Use ppg.py to generate the PPG.

Use preprocess_ppg.py to generate the WEO.

## If you want to use WEO to train a cross-lingual voice conversion model:

First you need to train the model without speaker consistency loss for 100k steps:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/b5e8e984dffd5a12910d1846e25b128298933e40/train_whisper_emo.py#L214C11-L214C11) to 

```python
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl# + loss_emo
```

run the py file:

```python
python train_whisper_emo.py -c configs/cvc-whispers-multi.json -m cvc-whispers-three
```

Then change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/71cf17a5b65c12987ea7fba74d1d173ea1aae5cb/train_whisper_emo.py#L214) back to finetune this model with speaker consistency loss

```python
python train_whisper_emo.py -c configs/cvc-whispers-three-emo.json -m cvc-whispers-three
```

## If you want to use PPG to train an expressive voice conversion model:

First you need to train the model without speaker consistency loss for 100k steps:

change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/71cf17a5b65c12987ea7fba74d1d173ea1aae5cb/train_eng_ppg_emo_loss.py#L311) to 

```python
loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl# + loss_emo
```

run the py file:

```python
python train_eng_ppg_emo_loss.py -c configs/cvc-eng-ppgs-three-emo.json -m cvc-eng-ppgs-three-emo
```

Then change [this line](https://github.com/ConsistencyVC/ConsistencyVC-voive-conversion/blob/71cf17a5b65c12987ea7fba74d1d173ea1aae5cb/train_eng_ppg_emo_loss.py#L311) back to finetune this model with speaker consistency loss

```python
python train_eng_ppg_emo_loss.py -c configs/cvc-eng-ppgs-three-emo-cycleloss.json -m cvc-eng-ppgs-three-emo
```


# Reference

The code structure is based on [FreeVC-s](https://github.com/OlaWod/FreeVC). Suggestion: please follow the instruction of FreeVC to install python requirements.

The WEO content feature is based on [LoraSVC](https://github.com/PlayVoice/lora-svc).

The PPG is from the [phoneme recognition model](https://huggingface.co/speech31/wav2vec2-large-english-TIMIT-phoneme_v3).


