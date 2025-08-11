# Toward Robust Speech Enhancement in Multilingual Environment (Submitted to ICASSP 2026)
### Yujie Xiong and Zhihua Huang
Audio samples are from THCHS+DNS dataset (mixed with THCHS-30 dataset and DNS-Challenge dataset). All wav files are resampled to 16kHz in our experiments.

 
## Pre-requisites
1. Python >= 3.9.
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](https://github.com/Yj-Xiong/TdSENet/blob/main/requirements.txt).
4. Download and extract the [VoiceBank+DEMAND dataset](https://datashare.ed.ac.uk/handle/10283/1942). 

## Training and Inference
### Step 1:

```pip install -r requirements.txt```

### Step 2:
Download VCTK-DEMAND dataset with 16 kHz, change the dataset dir:
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```
Or catalog other datasets following the above folder branches.

### Step 3:
If you want to train the model, run train.py
```
python train_td.py --data_dir <dir to VCTK-DEMAND dataset or your own dataset>
```

### Step 4:
Inference with the best ckpt for evaluation:
```
python inference_td.py --test_dir <dir to VCTK-DEMAND/test> --model_path <path to the best ckpt>
```

## Model Architecture
The overview of Td-SENet. <br><br>
<img src="https://github.com/Yj-Xiong/TdSENet/blob/main/models/TdSENet-Overview.png" width="600px">

The details of the modules. <br><br>
<img src="https://github.com/Yj-Xiong/TdSENet/blob/main/models/Modules.png" width="600px">


## Spectral Visualization
For Mandarin enhancement results with different denoising models, the spectral visualization uses B7_278.wav from THCHS-30 dataset.
<img src="https://github.com/Yj-Xiong/TdSENet/blob/main/data/Spectrograms.png" width="600px">

For more convincing, we visualize the spectrograms of the audio samples, where white boxes are added for the highlighted comparison. It can be observed that (1). the spectra in (c) and (d) retain more residual noise or lose more original components in the lower frequency bands; (2). artificial sounds are introduced in (d); (3). Due to the inherently low spectral energy density in high-frequency bands, the high-frequency components in both (c) and (d) exhibit relatively sparse distributions. In contrast, Td-SENet demonstrates superior high-frequency harmonic retention, visually validating the efficacy of HfCB.
## Acknowledgements
We referred to [CMGAN](https://github.com/ruizhecao96/CMGAN/).
