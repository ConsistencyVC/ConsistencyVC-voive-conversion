# ConsistencyVC-voive-conversion

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

If you want to use WEO to train a cross-lingual voice conversion model:
<!-- python train.py -c configs/freevc.json -m freevc -->

If you want to use PPG to train a expressive voice conversion model:
<!-- python train.py -c configs/freevc.json -m freevc -->
