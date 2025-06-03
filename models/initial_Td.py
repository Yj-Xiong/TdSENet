import torch
import torch.fft
import torch.nn as nn
from utils import *
from .conformer import LocConformerBlock

class SelfCA(nn.Module):# F_in = [B,T*F,C]
    def __init__(self, c_in):
        super(SelfCA, self).__init__()
        self.q_conv = nn.Conv1d(c_in, c_in, 1)
        self.k_conv = nn.Conv1d(c_in, c_in, 1)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x_in):
        B, N, C = x_in.shape
        x = x_in.reshape(B, C, N)
        Q = self.q_conv(x)
        K = self.k_conv(x).transpose(1, 2)
        c_c = self.softmax(Q @ K)
        out = x_in @ c_c + x_in # residual connection
        return out      # F_out = [B,N,C]
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
class LearnableSigmoid_2d(nn.Module):
    def __init__(self, in_features, beta=2):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)
class ChannelFeatureBranch(nn.Module): # 通道注意力分支 F_in = [B,N,C]
    def __init__(self,chan1,chan2=4):
        super(ChannelFeatureBranch, self).__init__()
        self.conv1 =  nn.Conv1d(chan1,chan2,3,padding=1)
        self.ln1 = nn.LayerNorm(chan2)
        self.act1 = nn.Sigmoid()
        self.CA = SelfCA(chan2)
        self.conv2 = nn.Conv1d(chan2,chan1,3,padding=1)
        self.ln2 = nn.LayerNorm(chan1)
        self.act2 = nn.SiLU()
    def forward(self, x_in,batch):
        # B,N,C = x_in.shape
        # if B//batch is int :
        #     x = x_in.reshape(-1,C,B//batch*N)
        # else:
        #     x = x_in.reshape(-1,C,N)
        B,C,T,F = x_in.shape
        x = x_in.reshape(B,C,T*F)
        conved1 = self.conv1(x)
        conved1 = self.act1(self.ln1(conved1.permute(0,2,1)))
        x_ca = self.CA(conved1) #+ conved1
        conved2 = self.conv2(x_ca.permute(0,2,1))
        conved2 = self.act2(self.ln2(conved2.permute(0,2,1)))
        x_out = conved2.reshape(B,C,T,F)
        return x_out
#
#



class DilatedDenseNet(nn.Module):
    def __init__(self, depth=4, in_channels=64):
        super(DilatedDenseNet, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.pad = nn.ConstantPad2d((1, 1, 1, 0), value=0.0)
        self.twidth = 2
        self.kernel_size = (self.twidth, 3)
        for i in range(self.depth):
            dil = 2**i
            pad_length = self.twidth + (dil - 1) * (self.twidth - 1) - 1
            setattr(
                self,
                "pad{}".format(i + 1),
                nn.ConstantPad2d((1, 1, pad_length, 0), value=0.0),
            )
            setattr(
                self,
                "conv{}".format(i + 1),
                nn.Conv2d(
                    self.in_channels * (i + 1),
                    self.in_channels,
                    kernel_size=self.kernel_size,
                    dilation=(dil, 1),
                ),
            )
            setattr(
                self,
                "norm{}".format(i + 1),
                nn.InstanceNorm2d(in_channels, affine=True),
            )
            setattr(self, "prelu{}".format(i + 1), nn.PReLU(self.in_channels))

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            out = getattr(self, "pad{}".format(i + 1))(skip)
            out = getattr(self, "conv{}".format(i + 1))(out)
            out = getattr(self, "norm{}".format(i + 1))(out)
            out = getattr(self, "prelu{}".format(i + 1))(out)
            skip = torch.cat([out, skip], dim=1)
        return out




class DenseEncoder(nn.Module):
    def __init__(self, in_channel, channels=32):
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
        # self.mtra1 = MambaTRA(channels)
    def forward(self, x):#
        x = self.conv_1(x)
        x = self.dilated_dense(x)
        # self.mtra1(x)
        x = self.conv_2(x)

        return x

# #

class TSCB(nn.Module):
    def __init__(self, num_channel=64):
        super(TSCB, self).__init__()
        self.time_conformer = LocConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)
        self.freq_conformer = LocConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=0.2, ff_dropout=0.2)

    def forward(self, x_in):
        b, c, t, f = x_in.size()
        x_t = x_in.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x_t = self.time_conformer(x_t) + x_t
        x_f = x_t.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x_f = self.freq_conformer(x_f) + x_f
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
    def __init__(self, num_features, num_channel=64, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.conv_1 = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.norm = nn.InstanceNorm2d(out_channel, affine=True)
        self.prelu = nn.PReLU(out_channel)
        self.final_conv = nn.Conv2d(out_channel, out_channel, (1, 1))
        self.prelu_out = nn.PReLU(num_features, init=-0.25)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.conv_1(x)
        x = self.prelu(self.norm(x))
        x = self.final_conv(x).permute(0, 3, 2, 1).squeeze(-1)
        return self.prelu_out(x).permute(0, 2, 1).unsqueeze(1)
class ComplexDecoder(nn.Module):
    def __init__(self, num_channel=64):
        super(ComplexDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.sub_pixel = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.prelu = nn.PReLU(num_channel)
        self.norm = nn.InstanceNorm2d(num_channel, affine=True)
        self.conv = nn.Conv2d(num_channel, 2, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.sub_pixel(x)
        x = self.prelu(self.norm(x))
        x = self.conv(x)
        return x

class PhaseDecoder(nn.Module):
    def __init__(self,num_channel=64, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DilatedDenseNet(depth=4, in_channels=num_channel)
        self.phase_conv = SPConvTranspose2d(num_channel, num_channel, (1, 3), 2)
        self.phase_conv_r = nn.Conv2d(num_channel, out_channel, (1, 2))
        self.phase_conv_i = nn.Conv2d(num_channel, out_channel, (1, 2))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x
class TSCNet(nn.Module):
    def __init__(self,num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=4, channels=num_channel)
        self.hfcb = ChannelFeatureBranch(num_channel)
        self.hfcb1 = ChannelFeatureBranch(num_channel)
        self.hfcb2 = ChannelFeatureBranch(num_channel)
        self.TSCB_1 = TSCB(num_channel=num_channel)
        self.TSCB_2 = TSCB(num_channel=num_channel)
        self.TSCB_3 = TSCB(num_channel=num_channel)
        self.TSCB_4 = TSCB(num_channel=num_channel)
        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)
        self.phase_decoder = PhaseDecoder(num_channel=num_channel)

    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) #切片操作分别选择了 x 张量中第二个维度（通常用于表示实部和虚部）中的第一个和第二个通道
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)

        x_in = torch.cat([mag, x,noisy_phase], dim=1) #(M、P和RI沿第二个维度拼接) [B, 4, T, F]
        # x_in = F.pad(x_in,[0,0,2,0])
        out_1 = self.dense_encoder(x_in)
        d = out_1.shape[3]
        # 创建一个与 out1 其他维度相同、第四个维度大小为 d//2 的零张量
        zeros = torch.zeros(out_1.shape[0], out_1.shape[1], out_1.shape[2], d // 2).to(out_1.device)
        # 获取 out1 的第四个维度的后一半
        out_1_slice = out_1[:, :, :, (d // 2):]
        out_c = self.hfcb(out_1_slice, batch=4)
        # 在前面补零，并恢复成原来的形状
        out_c = torch.cat((zeros, out_c), dim=3)

        out_2 = self.TSCB_1(out_1)
        out_2_slice = out_2[:, :, :, (d // 2):]
        out_c1 = self.hfcb1(out_2_slice, batch=4)
        out_c1 = torch.cat((zeros, out_c1), dim=3)
        out_3 = self.TSCB_2(out_2)
        out_3_slice = out_3[:, :, :, (d // 2):]
        out_c2 = self.hfcb2(out_3_slice, batch=4)
        out_c2 = torch.cat((zeros, out_c2), dim=3)
        out_3 += out_c2

        out_4 = self.TSCB_3(out_3)
        out_4 += out_c1
        out_5 = self.TSCB_4(out_4)
        out_5 += out_c

        mask = self.mask_decoder(out_5)
        denoised_phase = self.phase_decoder(out_5)
        out_mag = mask * mag
        complex_out = self.complex_decoder(out_5)
        out_mag = out_mag
        complex_out = complex_out
        mag_real = out_mag * torch.cos(denoised_phase)
        mag_imag = out_mag * torch.sin(denoised_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)
        return final_real, final_imag , denoised_phase








