#  [Td-SENet](https://github.com/Yj-Xiong/TdSENet)

[English](README.md) | **中文**

论文 *"Improving Local Features and High-frequency Information for Cross-lingual and Low-SNR Speech Enhancement"* 的官方 PyTorch 实现。音频样本来自 THCHS+DNS 数据集（混合 THCHS-30 数据集与 DNS-Challenge 数据集）。所有实验中的 wav 文件均已重采样至 16kHz。

### 作者：Yujie Xiong, Zhihua Huang

## 跨语言语音增强
跨语言语音增强示例，Td-SENet 在多种语言的复杂声学环境中均能实现有效增强。
<img src="figs/Fig1_cross_lingual_example.png" width="1200px">

## 环境要求
1. Python >= 3.9。
2. 克隆本仓库。
3. 安装 Python 依赖，请参考 [requirements.txt](requirements.txt)。
4. 下载并解压 [VoiceBank+DEMAND 数据集](https://datashare.ed.ac.uk/handle/10283/1942)。

## 训练与推理
### 步骤 1：

```pip install -r requirements.txt```

### 步骤 2：
下载 16 kHz 的 VCTK-DEMAND 数据集，按如下结构组织数据目录：
```
-VCTK-DEMAND/
  -train/
    -noisy/
    -clean/
  -test/
    -noisy/
    -clean/
```
其他数据集也请按照上述文件夹分支结构进行组织。

### 步骤 3：
如需训练模型，运行 [train_td.py](train_td.py)：
```
python train_td.py --data_dir <VCTK-DEMAND数据集或自定义数据集的目录>
```

### 步骤 4：
使用最佳检查点进行推理评估：
```
python inference_td.py --test_dir <VCTK-DEMAND/test目录> --model_path <最佳检查点路径>
```

## 模型架构

### Td-SENet 总体架构
<img src="figs/Fig2_TdSENet_overview.png" width="1200px">

### 时频局部增强 Conformer（TF-LocConformer）
<img src="figs/Fig3_TF_LocConformer.png" width="1200px">

### 卷积 SwiGLU（ConvSwiGLU）模块
<img src="figs/Fig4_ConvSwiGLU.png" width="500px">

### 高频通道块（HfCB）
高频通道块采用通道注意力机制，专门增强高频特征，确保在低信噪比条件下实现高保真重建。
<img src="figs/Fig5_HfCB.png" width="1200px">


## 低信噪比普通话降噪
对训练好的普通话模型在低信噪比及恶劣条件下的进一步评估（使用 DNS-Challenge 数据集中 400 种未见噪声及具有挑战性的嘈杂人声）。
<img src="data/SNR_Comparison.png" width="1800px">
显然，Td-SENet 在每种情况下均保持了对 SOTA 模型的整体优势。

## 频谱可视化
针对不同降噪模型的普通话增强效果，使用 THCHS-30 数据集中的 B7_278.wav 进行频谱可视化。
<img src="figs/Fig6_Spectrograms.png" width="1800px">

为了更具说服力，我们可视化了音频样本的频谱图，其中添加了白色方框以突出对比。可以观察到：

1. （c）和（d）中的频谱在低频段保留了更多残留噪声或丢失了更多原始成分；

2. （d）中引入了人工伪音；

3. 由于高频段固有的低频谱能量密度，（c）和（d）中的高频分量均呈现相对稀疏的分布。

相比之下，Td-SENet 展现出优越的高频谐波保留能力，从视觉上验证了 HfCB 模块的有效性。

## 音频演示
试听我们的语音增强效果演示：[Demo 页面](https://yj-xiong.github.io/TdSENet/demo.html)

## 致谢
我们参考了 [CMGAN](https://github.com/ruizhecao96/CMGAN/)。
