# Toward Robust Speech Enhancement in Multilingual Environment

**English** | [中文](README_CN.md)

Official PyTorch implementation of the paper *"Improving Local Features and High-frequency Information for Cross-lingual and Low-SNR Speech Enhancement"*. Audio samples are from THCHS+DNS dataset (mixed with THCHS-30 dataset and DNS-Challenge dataset). All wav files are resampled to 16kHz in our experiments.

### Yujie Xiong and Zhihua Huang

## Pre-requisites
1. Python >= 3.9.
2. Clone this repository.
3. Install python requirements. Please refer [requirements.txt](requirements.txt).
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
If you want to train the model, run [train_td.py](train_td.py):
```
python train_td.py --data_dir <dir to VCTK-DEMAND dataset or your own dataset>
```

### Step 4:
Inference with the best ckpt for evaluation:
```
python inference_td.py --test_dir <dir to VCTK-DEMAND/test> --model_path <path to the best ckpt>
```

## Model Architecture

### Overview of Td-SENet
<img src="figs/Fig2_TdSENet_overview.png" width="1200px">

### Time-Frequency LocConformer (TF-LocConformer)
<img src="figs/Fig3_TF_LocConformer.png" width="1200px">

### Convolution SwiGLU (ConvSwiGLU) Module
<img src="figs/Fig4_ConvSwiGLU.png" width="500px">

### High-frequency Channel Block (HfCB)
<img src="figs/Fig5_HfCB.png" width="1200px">


## Cross-lingual Speech Enhancement
An example of cross-lingual speech enhancement, where Td-SENet effectively handles multiple languages in challenging acoustic environments.
<img src="figs/Fig1_cross_lingual_example.png" width="1200px">

## Low SNR Mandarin Denoising
Further assessment for well-trained Mandarin models under low SNR and adverse conditions (400 unseen noises from DNS-challenge dataset and challenging babble noise).
<img src="data/SNR_Comparison.png" width="1800px">
Clearly, Td-SENet maintains overall superiority over the SOTA models in each case.

## Spectral Visualization
For Mandarin enhancement results with different denoising models, the spectral visualization uses B7_278.wav from THCHS-30 dataset.
<img src="figs/Fig6_Spectrograms.png" width="1800px">

For more convincing, we visualize the spectrograms of the audio samples, where white boxes are added for the highlighted comparison. It can be observed that:

1. the spectra in (c) and (d) retain more residual noise or lose more original components in the lower frequency bands;

2. artificial sounds are introduced in (d);

3. Due to the inherently low spectral energy density in high-frequency bands, the high-frequency components in both (c) and (d) exhibit relatively sparse distributions.

In contrast, Td-SENet demonstrates superior high-frequency harmonic retention, visually validating the efficacy of HfCB.

## Audio Demo
Listen to our speech enhancement demos: [Demo Page](demo.html)

## Acknowledgements
We referred to [CMGAN](https://github.com/ruizhecao96/CMGAN/).
