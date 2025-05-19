import torch
import torch.nn as nn
from transformer import TransformerBlock
# from utils import *
# from conformer import ConformerBlock as TransformerBlock
class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
class DilatedDenseNet(nn.Module):
    # orgin
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2 ** i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(self, 'pad{}'.format(i + 1), nn.ConstantPad2d((1, 1, pad_length, 0), value=0.))
            setattr(self, 'conv{}'.format(i + 1),
                    nn.Conv2d(self.in_channels * (i + 1), self.in_channels, kernel_size=self.kernel_size,
                              dilation=(dil, 1)))
            setattr(self, 'norm{}'.format(i + 1), nn.InstanceNorm2d(in_channels, affine=True))
            setattr(self, 'prelu{}'.format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, 'pad{}'.format(i + 1))(skip)
            out = getattr(self, 'conv{}'.format(i + 1))(out)
            out = getattr(self, 'norm{}'.format(i + 1))(out)
            out = getattr(self, 'prelu{}'.format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
       # out = out[:, :, :-1, :]         #Casual
        return out
class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
class BiasNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5):
        super(BiasNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps

        # 可学习的逐通道偏置 (b)
        self.bias = nn.Parameter(torch.zeros(num_features))

        # 可学习的标量参数 (γ)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        # 确保输入是 2D 或更高维度 (batch_size, num_features, ...)
        if x.dim() < 2:
            raise ValueError("Input must have at least 2 dimensions.")

        # 计算 x - b
        x_centered = x - self.bias.view(1, -1, *([1] * (x.dim() - 2)))

        # 计算 RMS[x - b] (在通道维度上计算均方根值)
        rms = torch.sqrt(torch.mean(x_centered ** 2, dim=1, keepdim=True) + self.eps)

        # 归一化并应用缩放
        x_norm = x / rms
        x_norm = x_norm * torch.exp(self.gamma)

        return x_norm
class DS_DDB(nn.Module):
    def __init__(self, in_channels=64,  depth=4):
        super(DS_DDB, self).__init__()
        self.in_channels = in_channels
        self.depth = depth
        # self.Deform_Embedding = Deform_Embedding(in_chans=h.dense_channel, embed_dim=h.dense_channel)
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dil = 2 ** i
            dense_conv = nn.Sequential(
                nn.Conv2d(self.in_channels*(i+1), self.in_channels*(i+1), (3, 3), dilation=(dil, 1),
                          padding=get_padding_2d((3, 3), dilation=(dil, 1)), groups=self.in_channels*(i+1), bias=True),
                nn.Conv2d(in_channels=self.in_channels*(i+1), out_channels=self.in_channels, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.InstanceNorm2d(self.in_channels, affine=True),
                nn.PReLU(self.in_channels)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):

        skip = x
        for i in range(self.depth):

            # if i == 0:
            #     x = self.Deform_Embedding(x)
            #     x = self.dense_block[i](x)
            # else:
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x
#Class FiLM(nn.Module):
#
#   def __init__(self, zdim, maskdim):
#      super(FiLM, self).__init__()
#
#      self.gamma = nn.Linear(zdim, maskdim)   # s
#      self.beta = nn.Linear(zdim, maskdim)    # t
#
#      self.down_sample=conv_1 = nn.Sequential(
#             nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
#             nn.InstanceNorm2d(channels, affine=True),      #实例归一化
#             nn.PReLU(channels)                             #PReLU激活
#         )
#   def forward(self, x, z):
#      z=self.down_sample(z).permute(0,3,2,1).contiguous() #(bs,C,T,F)
#      z=z.view(-1,1) #(bs*T*F,C)
#      gamma = self.gamma(z)
#      beta = self.beta(z)      #(bs*T*F,64)
#      gamma=gamma.view(x.shape[0],x.shape[3],x.shape[2],x.shape[1]) #(bs,C,T,F)
#      beta=beta.view(x.shape[0],x.shape[3],x.shape[2],x.shape[1])
#
#      x = gamma.permute(0,3,2,1) * x + beta.permute(0,3,2,1)
#
#      return x


class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=64):
        super(DenseEncoder, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, channels, (1, 1), (1, 1)),
            nn.InstanceNorm2d(channels, affine=True),      #实例归一化
            nn.PReLU(channels)                              #PReLU激活
        )
        self.dilated_dense = DilatedDenseNet(depth=4, in_channels=channels)
        self.conv_2 = nn.Sequential(
            nn.Conv2d(channels, channels, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(channels, affine=True),
            nn.PReLU(channels)
        )
    def forward(self, x):#
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        x = self.conv_2(x)

        return x

# class TSCB(nn.Module):
#     def __init__(self, num_channel=64):
#         super(TSCB, self).__init__()
#         self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#         self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
#                                              conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
#
#
#     def forward(self, x_in):
#         b, c, t, f = x_in.size()
#         x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
#         x_t = self.time_conformer(x_t) + x_t
#         x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
#         x_f = self.freq_conformer(x_f) + x_f
#         x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
#         return x_f
class TSTB(nn.Module):
    def __init__(self, num_channel=64,drop=0.):
        super(TSTB, self).__init__()
        # self.time_transformer = TransformerBlock(dim=num_channel, heads=4)
        # self.freq_transformer = TransformerBlock(dim=num_channel, heads=4)
        self.time_transformer = TransformerBlock(d_model=num_channel, n_heads=4)
        self.freq_transformer = TransformerBlock(d_model=num_channel, n_heads=4)

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_t = self.time_transformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_transformer(x_f) + x_f
        x_f = x_f.view(b, t, f, c).permute(0, 3, 1, 2)
        return x_f

class SPConvTranspose2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, r=1):
        super(SPConvTranspose2d, self).__init__()
        self.pad1 = nn.ConstantPad2d((1, 1, 0, 0), value=0.)
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels * r, kernel_size=kernel_size, stride=(1, 1))
        self.r = r

    def forward(self, x):
        x = self.pad1(x)
        out = self.conv(x)
        batch_size, nchannels, H, W = out.shape
        out = out.view((batch_size, self.r, nchannels // self.r, H, W))
        out = out.permute(0, 2, 3, 4, 1)
        out = out.contiguous().view((batch_size, nchannels // self.r, H, -1))
        return out

class MaskDecoder(nn.Module):
    def __init__(self, feature_size, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet( depth=4,in_channels=num_channel)
        self.mask_conv = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
            nn.Conv2d(num_channel, out_channel, (1, 2))
        )
        self.lsigmoid = LearnableSigmoid_2d(feature_size)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        # x = x.permute(0, 3, 2, 1) # [B, F, T, C]
        x = self.lsigmoid(x = x.permute(0, 1, 3, 2))
        return x.permute(0, 1, 3, 2)


class PhaseDecoder(nn.Module):
    def __init__(self, num_channel=64, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DilatedDenseNet( depth=4,in_channels=num_channel)
        self.phase_conv = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel)
        )
        self.phase_conv_r = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(num_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)

        # x = x.permute(0, 3, 2, 1).squeeze(-1) # [B, F, T]
        return x

class AdaptiveDecoder(nn.Module):
    def __init__(self, num_features,f_types, num_channel=64, ): #out_channel=[M,P,C]
        super(AdaptiveDecoder, self).__init__()
        self.dense_block = DS_DDB(depth=4, in_channels=num_channel)
        self.initial_conv = nn.Sequential(
            SPConvTranspose2d(num_channel, num_channel, (1, 3), 2),
            nn.InstanceNorm2d(num_channel, affine=True),
            nn.PReLU(num_channel),
        )
        self.mask_conv = nn.Conv2d(num_channel, 1, (1, 2))
        # self.mask_act = nn.Sequential(BiasNorm(1),nn.Softmax(dim=1))
        self.mask_act = LearnableSigmoid_2d(num_features)
        # self.LSigmoid2d = LearnableSigmoid_2d(num_features)
        self.weights = [nn.Parameter(torch.rand(1), requires_grad=True).cuda() , nn.Parameter(torch.rand(1), requires_grad=True).cuda()]
        if f_types == "C":
            self.complex_conv = nn.Conv2d(num_channel, 2, (1, 2))
            self.BiasNorm = BiasNorm(2)
        elif f_types == "P":
            self.phase_conv = nn.modules.ModuleList([nn.Conv2d(num_channel, 1, (1, 2)) for i in range(2)])
            self.BiasNorm = [BiasNorm(1),BiasNorm(1)]
        elif f_types == "CP":
            self.complex_conv = nn.Conv2d(num_channel, 2, (1, 2))
            self.BiasNorm = [BiasNorm(2),BiasNorm(1), BiasNorm(1)]
            self.phase_conv = nn.modules.ModuleList([nn.Conv2d(num_channel, 1, (1, 2)) for i in range(2)])
            self.weights.append(nn.Parameter(torch.rand(1), requires_grad=True).cuda())

    def forward(self, x, f_types):

        if f_types == "C":
            device = x.device
            self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x*self.weights[0]).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2, 1).unsqueeze(1)
            x_complex = self.BiasNorm(self.complex_conv(x*self.weights[1]))
            return x_mask, x_complex
        if f_types == "P":
            device = x.device
            self.BiasNorm = [bn.to(device) for bn in self.BiasNorm]
            self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x*self.weights[0]).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2, 1).unsqueeze(1)
            x_p = x * self.weights[1]
            x_r = self.BiasNorm[0](self.phase_conv[0](x_p))
            x_i = self.BiasNorm[1](self.phase_conv[1](x_p))
            x_pha = torch.atan2(x_i, x_r)
            return x_mask, x_pha
        if f_types == "CP":
            device = x.device
            self.weights = [w.to(device) for w in self.weights]
            x = self.dense_block(x)
            x = self.initial_conv(x)
            x_mask = self.mask_act(self.mask_conv(x * self.weights[0]).permute(0, 3, 2, 1).squeeze(-1)).permute(0, 2,1).unsqueeze(1)
            x_complex = self.BiasNorm[0](self.complex_conv(x * self.weights[1]))
            x_p = x * self.weights[2]
            x_r = self.BiasNorm[1](self.phase_conv[0](x_p))
            x_i = self.BiasNorm[2](self.phase_conv[1](x_p))
            x_pha = torch.atan2(x_i, x_r)
            return x_mask, x_complex, x_pha
class MPNet(nn.Module):
    def __init__(self,num_channel=64, num_features=201):
        super(MPNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=2, channels=num_channel)
        self.TSTB_1 = TSTB(num_channel=num_channel,drop=0.)
        self.TSTB_2 = TSTB(num_channel=num_channel,drop=0.)
        self.TSTB_3 = TSTB(num_channel=num_channel,drop=0.)
        self.TSTB_4 = TSTB(num_channel=num_channel,drop=0.)
        # self.dense_decoder = AdaptiveDecoder(num_features=num_features, f_types="P", num_channel=num_channel)
        self.mask_decoder = MaskDecoder( feature_size=num_features, num_channel=num_channel, out_channel=1)
        self.phase_decoder = PhaseDecoder(num_channel=num_channel)
    def forward(self, x, *args, **kwargs):
        # mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) #切片操作分别选择了 x 张量中第二个维度（通常用于表示实部和虚部）中的第一个和第二个通道
        # noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([x, x], dim=1) #(幅度和相位沿第二个维度拼接)
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSTB_1(out_1)
        out_3 = self.TSTB_2(out_2)
        out_4 = self.TSTB_3(out_3)
        out_5 = self.TSTB_4(out_4)
        # mask, denoised_pha = self.dense_decoder(out_5, f_types="P")
        # denoised_amp = mag * mask
        denoised_pha = self.phase_decoder(out_5)
        denoised_amp = x * self.mask_decoder(out_5)
        denoised_pha = self.phase_decoder(out_5)
        final_real, final_imag , = denoised_amp * torch.cos(denoised_pha),  denoised_amp * torch.sin(denoised_pha)

        return final_real, final_imag , denoised_pha
import time
import torch
import argparse
import json
import os
from attrdict import AttrDict
from thop import profile

def main():
    model = MPNet(num_channel=64).cuda()
    noisy_com = torch.randn(1, 1, 321, 201).cuda()
    print('Initializing Inference Process..')
    print('Warming up the model...')
    for i in range(10):
        with torch.no_grad():
            # model(input_data)
            model(noisy_com)
            # audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            # audio_g = audio_g / norm_factor

    print('Measuring inference speed...')
    total_time = 0
    num_iters = 100
    for i in range(num_iters):
        start_time = time.time()
        with torch.no_grad():
            model(noisy_com)

        end_time = time.time()
        total_time += end_time - start_time

    avg_time = total_time / num_iters
    print('Avg. Inference Time: {:.3f} seconds'.format(avg_time))
    noisy_com = torch.randn(1, 1, 321, 201).cuda()
    # Measure FLOPS
    flops, params = profile(model, inputs=(noisy_com,noisy_com))
    print('Number of FLOPs: {:.3f} GFLOPs'.format(flops / 1e9))
    print(f"Total Parameters: {params:,}")
    print(f"Total FLOPs: {flops:,}")
    max_memory = 0
    num_iters = 100
    for i in range(num_iters):
        torch.cuda.reset_max_memory_allocated()
        with torch.no_grad():
            model(noisy_com,noisy_com)
        max_memory = max(max_memory, torch.cuda.max_memory_allocated() / 1024**2)  # 转换为MB
    print('Data Length: {:.3f} seconds'.format(2))
    print('Avg. Inference Time: {:.3f} seconds'.format(avg_time))
    print('Number of FLOPs: {:.3f} GFLOPs'.format(flops / 1e9))
    print('Max Memory Usage: {:.2f} MB'.format(max_memory))
    print(f"Manual calculation of parameters: {params}")
if __name__ == '__main__':
    main()













