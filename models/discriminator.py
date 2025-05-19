import numpy as np
import torch
import torch.nn as nn
from joblib import Parallel, delayed
from pesq import pesq
from tools.compute_metrics import *
from util import *


class Simam_module(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(Simam_module, self).__init__()
        self.act = nn.ReLU()  # 使用Sigmoid激活函数
        self.e_lambda = e_lambda  # 定义平滑项e_lambda，防止分母为0

    def forward(self, x):
        b, c, h, w = x.size()  # 获取输入x的尺寸
        n = w * h - 1  # 计算特征图的元素数量减一，用于下面的归一化
        # 计算输入特征x与其均值之差的平方
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        # 计算注意力权重y，这里实现了SimAM的核心计算公式
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5
        # 返回经过注意力加权的输入特征
        return x * self.act(y)
def pesq_loss(clean, noisy, sr=16000):
    try:
        pesq_score = pesq(sr, clean, noisy, 'wb')
    except:
        # error can happen due to silent period
        pesq_score = -1
    return pesq_score

def stoi_loss(clean, noisy, sr=16000):
    try:
        stoi_score = stoi(clean, noisy, sr)
    except:
        # error can happen due to silent period
        stoi_score = -1
    return stoi_score
def si_sdr(x, s, remove_dc=True):
    """
    Compute SI-SDR
    x: extracted signal
    s: reference signal(ground truth)
    """

    def vec_l2norm(x):
        return np.linalg.norm(x, 2)

    if remove_dc:
        x_zm = x - np.mean(x)
        s_zm = s - np.mean(s)
        t = np.inner(x_zm, s_zm) * s_zm / vec_l2norm(s_zm) ** 2
        n = x_zm - t
    else:
        t = np.inner(x, s) * s / vec_l2norm(s) ** 2
        n = x - t
    return 20 * np.log10(vec_l2norm(t) / vec_l2norm(n))
def batch_stoi(clean, noisy):
    stoi_score = Parallel(n_jobs=8)(delayed(stoi_loss)(c, n) for c, n in zip(clean, noisy))
    stoi_score = np.array(stoi_score)
    if -1 in stoi_score:
        return None
    stoi_score = (stoi_score + 3) / 4
    return torch.FloatTensor(stoi_score).to('cuda')
def stoi_value(utts_r, utts_g,sr):

    pesq_score = Parallel(n_jobs=8)(delayed(stoi_loss)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(),
                            sr)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score
def pesq_value(utts_r, utts_g,sr):

    pesq_score = Parallel(n_jobs=8)(delayed(pesq_loss)(
                            utts_r[i].squeeze().cpu().numpy(),
                            utts_g[i].squeeze().cpu().numpy(),
                            sr)
                          for i in range(len(utts_r)))
    pesq_score = np.mean(pesq_score)

    return pesq_score
def b_pesq(clean, noisy,device):
    pesq_score = Parallel(n_jobs=8)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to(device)
def batch_pesq(clean, noisy):
    pesq_score = Parallel(n_jobs=8)(delayed(pesq_loss)(c, n) for c, n in zip(clean, noisy))
    pesq_score = np.array(pesq_score)
    if -1 in pesq_score:
        return None
    pesq_score = (pesq_score - 1) / 3.5
    return torch.FloatTensor(pesq_score).to('cuda')
class TinyDiscriminator(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )


    def forward(self, x,y):

        #x/y = [b,f,t,2]
        x,y = torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2).unsqueeze(1), torch.sqrt(y[:,:,:,0]**2 + y[:,:,:,1]**2).unsqueeze(1)
        x,y = x.permute(0,1,3,2), y.permute(0,1,3,2)
        xy = torch.cat([x, y], dim=1)
        # if
        # print("Shape before AdaptiveMaxPool2d:", xy.shape)
        return self.layers(xy)
class NoisyDiscriminator(nn.Module):
    def __init__(self, ndf, in_channel=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )


    def forward(self, x):

        #x/y = [b,f,t,2]
        x = torch.sqrt(x[:,:,:,0]**2 + x[:,:,:,1]**2).unsqueeze(1)
        x = x.permute(0,1,3,2)
        # xy = torch.cat([x, y], dim=1)
        # if
        # print("Shape before AdaptiveMaxPool2d:", xy.shape)
        return self.layers(x)
class Discriminator(nn.Module):
    def __init__(self, ndf, in_channel=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(in_channel, ndf, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf, affine=True),
            nn.PReLU(ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf, ndf*2, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*2, affine=True),
            nn.PReLU(2*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*2, ndf*4, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*4, affine=True),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Conv2d(ndf*4, ndf*8, (4,4), (2,2), (1,1), bias=False)),
            nn.InstanceNorm2d(ndf*8, affine=True),
            nn.PReLU(8*ndf),

            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.utils.spectral_norm(nn.Linear(ndf*8, ndf*4)),
            nn.Dropout(0.3),
            nn.PReLU(4*ndf),
            nn.utils.spectral_norm(nn.Linear(ndf*4, 1)),
            LearnableSigmoid(1)
        )

    def forward(self, x, y):
        xy = torch.cat([x, y], dim=1)
        # if
        # print("Shape before AdaptiveMaxPool2d:", xy.shape)
        return self.layers(xy)