import math
import numbers

from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from utils import *
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout

from modules.mamba_simple import Mamba


# from KAN import GR_KAN
# from kan.KAN import KAN
# from .ParallelConformer.PC import CFTSA
# source: https://github.com/lucidrains/conformer/blob/master/conformer/conformer.py
# helper functions


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

import torch
import torch.nn as nn
from torch.nn import functional as F
class ConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.diff_ks = conv1d_kernel - conv1d_shift

        self.net = nn.Sequential(
            nn.Conv1d(dim, dim_inner, conv1d_kernel, stride=conv1d_shift),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """ConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        # b, s1, s2, h = x.shape
        # x = x.view(b * s1, s2, h)
        s2 = x.shape[1]
        x = x.transpose(-1, -2)
        x = self.net(x).transpose(-1, -2)
        x = x[..., self.diff_ks // 2 : self.diff_ks // 2 + s2, :]
        return x#.view(b, s1, s2, h)


class SwiGLUConvDeconv1d(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel, conv1d_shift, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift)

        self.swish = nn.SiLU()
        self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        # b, s1, s2, h = x.shape
        # x = x.contiguous().view(b * s1, s2, h)
        s2 = x.shape[1]
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.conv1d(x)
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        x = self.dropout(x)
        x = self.deconv1d(x).transpose(-1, -2)

        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x)#.view(b, s1, s2, h)
class DwSwiGLUConv(nn.Module):
    def __init__(self, dim, dim_inner, conv1d_kernel=4, conv1d_shift=1, dropout=0.0, **kwargs):
        super().__init__()

        self.conv1d = nn.Conv1d(dim, dim_inner * 2, conv1d_kernel, stride=conv1d_shift,groups=dim)
        self.pw_conv1d = nn.Conv1d(dim*2, dim_inner * 2, 1)

        self.swish = nn.SiLU()
        self.out_conv1d = nn.Conv1d(dim_inner, dim, 1)
        # self.deconv1d = nn.ConvTranspose1d(dim_inner, dim, conv1d_kernel, stride=conv1d_shift)
        self.dropout = nn.Dropout(dropout)
        self.dim_inner = dim_inner
        self.diff_ks = conv1d_kernel - conv1d_shift
        self.conv1d_kernel = conv1d_kernel
        self.conv1d_shift = conv1d_shift

    def forward(self, x):
        """SwiGLUConvDeconv1d forward

        Args:
            x: torch.Tensor
                Input tensor, (n_batch, seq1, seq2, channel)
                seq1 (or seq2) is either the number of frames or freqs
        """
        # b, s1, s2, h = x.shape
        # x = x.contiguous().view(b * s1, s2, h)
        s2 = x.shape[1]
        x = x.transpose(-1, -2)

        # padding
        seq_len = (
            math.ceil((s2 + 2 * self.diff_ks - self.conv1d_kernel) / self.conv1d_shift) * self.conv1d_shift
            + self.conv1d_kernel
        )
        x = F.pad(x, (self.diff_ks, seq_len - s2 - self.diff_ks))

        # conv-deconv1d
        x = self.pw_conv1d(self.conv1d(x))
        gate = self.swish(x[..., self.dim_inner :, :])
        x = x[..., : self.dim_inner, :] * gate
        # x = self.dropout(x)
        # x = self.deconv1d(x).transpose(-1, -2)
        x = self.out_conv1d(x).transpose(-1, -2)
        # cut necessary part
        x = x[..., self.diff_ks : self.diff_ks + s2, :]
        return self.dropout(x)#.view(b, s1, s2, h)
class SwiGLU(nn.Module):
    def __init__(self, beta=1.0):
        super().__init__()
        self.beta = beta  # 可设为可学习参数

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return (x1 * torch.sigmoid(self.beta * x1)) * x2
class BiMambaFFN(nn.Module):
    def __init__(self, d_model,gpu='cuda:0',dropout=0):
        super(MambaFFN, self).__init__()
        self.state = 2*d_model
        self.FMamba= Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=d_model,  # Model dimension d_model
            d_state= self.state,  # SSM state expansion factor
            d_conv= 4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.BMamba = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=d_model,  # Model dimension d_model
            d_state=self.state,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.fchannel_scale = nn.Parameter(torch.ones(1, 1, d_model))
        self.bchannel_scale = nn.Parameter(torch.zeros(1, 1, d_model))
        self.rearrange1 = Rearrange('b s c -> b c s')
        self.rearrange2 = Rearrange('b c s -> b s c')
        # if bidirectional:
        #     self.linear = Linear(d_model * 2 * 2, d_model)
        # else:
        #     self.linear = Linear(d_model * 2, d_model)
        self.conv_forward = nn.Conv1d(self.state, 2*self.state, 1,bias=True)
        self.dw_conv = nn.Conv1d(in_channels=2*self.state, out_channels=2*self.state, kernel_size=3, padding=1, stride=1,
                               groups=2*self.state,
                               bias=True)
        self.gate = SwiGLU()
        self.conv_out = nn.Conv1d(self.state, d_model, 1,bias=True)
        self.Fpre_norm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5,device=gpu)
        self.Bpre_norm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5,device=gpu)
        self.FMamba = self.Fpre_norm(self.FMamba)
        self.BMamba = self.Bpre_norm(self.BMamba)
        self.rmsg_norm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5,device=gpu)
    def forward(self, x):
        # self.Mamba.flatten_parameters()
        x_b = x.flip(1)
        x_forward = self.FMamba(x)
        x_forward = x + x_forward*self.fchannel_scale
        x_backforward = self.BMamba(x_b)
        x_backforward = x_b + x_backforward*self.bchannel_scale
        x = torch.cat([x_forward, x_backforward], dim=-1)
        x = self.rearrange1(x)
        x = self.conv_forward(x)
        x = self.dw_conv(x)
        x = self.gate(x)
        x = self.conv_out(x)
        x = self.rearrange2(x)
        x = self.rmsg_norm(x)

        return x
class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim, eps=1e-8, bias=False,device='cuda:0'):
        """
        Root Mean Square Group Normalization (RMSGroupNorm).
        Unlike Group Normalization in vision, RMSGroupNorm
        is applied to each TF bin.

        Args:
            num_groups: int
                Number of groups
            dim: int
                Number of dimensions
            eps: float
                Small constant to avoid division by zero.
            bias: bool
                Whether to add a bias term. RMSNorm does not use bias.

        """
        super().__init__()
        self.device = device
        assert dim % num_groups == 0, (dim, num_groups)
        self.num_groups = num_groups
        self.dim_per_group = dim // self.num_groups

        self.gamma = nn.Parameter(torch.Tensor(dim).to(torch.float32)).to(self.device)
        nn.init.ones_(self.gamma)

        self.bias = bias
        if self.bias:
            self.beta = nn.Parameter(torch.Tensor(dim).to(torch.float32)).to(self.device)
            nn.init.zeros_(self.beta)
        self.eps = eps
        self.num_groups = num_groups

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        others = input.shape[:-1]
        input = input.view(others + (self.num_groups, self.dim_per_group))

        # normalization
        norm_ = input.norm(2, dim=-1, keepdim=True)
        rms = norm_ * self.dim_per_group ** (-1.0 / 2)
        output = input / (rms + self.eps)

        # reshape and affine transformation
        output = output.view(others + (-1,))
        output = output * self.gamma
        if self.bias:
            output = output + self.beta

        return output
class GMamba(nn.Module):
    """Grouped Bidirectional Mamba"""

    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True, bidirectional=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.mamba1 = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=input_size//2,  # Model dimension d_model
            d_state=self.hidden_size,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2, # Block expansion factor
        )
        self.mamba1 = PreNorm(input_size//2, self.mamba1)
        self.mamba2 = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=input_size//2,  # Model dimension d_model
            d_state=self.hidden_size,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.mamba2 = PreNorm(input_size//2, self.mamba2)
        self.act = nn.GELU()
        self.post_norm = nn.LayerNorm(input_size)
        self.conv = ConformerConvModule(dim=input_size, causal=False, expansion_factor=2, kernel_size=31, dropout=0.)
        # self.conv1 = ConformerConvModule(dim=input_size//2, causal=False, expansion_factor=2, kernel_size=15, dropout=0.)
        # self.conv2 = ConformerConvModule(dim=input_size // 2, causal=False, expansion_factor=2, kernel_size=15,
        #                                 dropout=0.)
        # self.ac1 = nn.SiLU()
        # self.ac2 = nn.SiLU()
        # self.pn1 = nn.LayerNorm(input_size//2)
        # self.pn2 = nn.LayerNorm(input_size//2)

        # self.mra = MambaTRA(channels=self.hidden_size)
    def forward(self, x, h=None):
        """
        x: (B, seq_length, input_size)
        h: (num_layers, B, hidden_size)
        """

        x1, x2 = torch.chunk(x, chunks=2, dim=-1)
        x2 = torch.flip(x2, dims=[-1])
        y1,y2  = self.mamba1(x1), torch.flip(self.mamba2(x2), dims=[-1])
        # y1, y2 = self.conv1(y1)+ y1, self.conv2(y2) + y2
        # y1, y2 = self.ac1(self.pn1(y1)), self.ac2(self.pn2(y2))
        y = torch.cat([y1, y2], dim=-1)
        y = self.conv(y) + y
        y = self.act(self.post_norm(y))
        # h = torch.cat([h1, h2], dim=-1)
        # y = self.mra(y)
        return y
class LocConMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.norm = [RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5,device='cuda:0'),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5,device='cuda:0'),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5,device='cuda:0')]
        self.ff1 = SwiGLUConvDeconv1d(64,256,4,1)
        self.Mamba = GMamba(input_size=64, hidden_size=128, num_layers=1, batch_first=True, bidirectional=False)
        # self.Mamba = Mamba(
        #     # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
        #     d_model=dim,  # Model dimension d_model
        #     d_state=dim,  # SSM state expansion factor
        #     d_conv=ff_mult,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        self.drop = nn.Dropout(attn_dropout)
        # self.Mamba = PreNorm(dim, self.MambaBlock)
        # self.CFB = ChannelFeatureBranch(dim)
        # self.conv = MKGU(dim)
        # self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = SwiGLUConvDeconv1d(64, 256, 4, 1)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = self.norm[0](x)
        x = 0.5*self.ff1(x) + x
        x = self.norm[1](x)
        x = self.Mamba(x) + x
        x = self.drop(x)
        # x = self.conv(x)   +x
        x = self.norm[2](x)
        x = 0.5*self.ff2(x) + x
        x = self.post_norm(x)

        return x
class LocPreNormConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        # self.norm = [RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5)]
        self.ff1 = SwiGLUConvDeconv1d(64,256,4,1)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.CFB = ChannelFeatureBranch(dim)
        # self.conv = MKGU(dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = SwiGLUConvDeconv1d(64, 256, 4, 1)
        self.ff1 = PreNorm(dim, self.ff1)
        self.attn = PreNorm(dim, self.attn)
        self.ff2 = PreNorm(dim, self.ff2)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = 0.5*self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = 0.5*self.ff2(x) + x
        x = self.post_norm(x)

        return x
class RMSGConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.norm = [RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5)]
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.CFB = ChannelFeatureBranch(dim)
        # self.conv = MKGU(dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = self.norm[0](x)
        x = 0.5*self.ff1(x) + x
        x = self.norm[1](x)
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = self.norm[2](x)
        x = 0.5*self.ff2(x) + x
        x = self.post_norm(x)

        return x
class MLocConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.norm = [RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5)]
        self.ff1 = MambaFFN(dim, gpu='cuda:0')
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.CFB = ChannelFeatureBranch(dim)
        # self.conv = MKGU(dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # self.ff2 = SwiGLUConvDeconv1d(64, 256, 4, 1)
        # self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = self.norm[0](x)
        # x = 0.5*self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = self.norm[1](x)
        x = 0.5*self.ff1(x) + x
        x = self.norm[2](x)

        return x
class LocConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.norm = [RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5),RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5)]
        self.ff1 = SwiGLUConvDeconv1d(64,256,4,1)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = SwiGLUConvDeconv1d(64, 256, 4, 1)
        # self.post_norm = nn.LayerNorm(dim)
        # self.channel_scale1 = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        # self.channel_scale2 = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        # self.channel_scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.post_norm = RMSGroupNorm(num_groups=4, dim=dim,eps=1e-5)

    def forward(self, x, mask = None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = self.norm[0](x)
        x = 0.5*self.ff1(x) + x
        x = self.norm[1](x)
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = self.norm[2](x)
        x = 0*self.ff2(x) + x
        x = self.post_norm(x)

        return x


class KANConvConformerBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0.
    ):
        super().__init__()
        self.norm = [RMSGroupNorm(num_groups=4, dim=dim, eps=1e-5), RMSGroupNorm(num_groups=4, dim=dim, eps=1e-5),
                     RMSGroupNorm(num_groups=4, dim=dim, eps=1e-5), RMSGroupNorm(num_groups=4, dim=dim, eps=1e-5)]
        self.ff1 = SwiGLUConvDeconv1d(64, 256, 4, 1)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.ff2 = SwiGLUConvDeconv1d(64, 256, 4, 1)
        # self.post_norm = nn.LayerNorm(dim)
        self.kan = GR_KAN(in_features=64,hidden_features=128,out_features=64)
        self.post_norm = RMSGroupNorm(num_groups=4, dim=dim, eps=1e-5)

    def forward(self, x, mask=None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = self.norm[0](x)
        x = 0.5 * self.ff1(x) + x
        x = self.norm[1](x)
        x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x
        x = self.norm[2](x)
        x = 0.5 * self.ff2(x) + x
        x = self.post_norm(x)

        return x
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
        return out      #F_out = [B,N,C]


class ChannelFeatureBranch(nn.Module): # 通道注意力分支 F_in = [B,N,C]
    def __init__(self,chan1,chan2):
        super(ChannelFeatureBranch, self).__init__()
        self.conv1 =  nn.Conv1d(chan1,chan2,3,padding=1)
        self.ln1 = nn.LayerNorm(chan2)
        self.act1 = nn.ReLU()
        self.CA = SelfCA(chan2)
        self.conv2 = nn.Conv1d(chan2,chan1,3,padding=1)
        self.ln2 = nn.LayerNorm(chan1)
        self.act2 = nn.ReLU()
    def forward(self, x_in,batch):
        B,N,C = x_in.shape
        if B//batch is int :
            x = x_in.reshape(-1,C,B//batch*N)
        else:
            x = x_in.reshape(-1,C,N)

        conved1 = self.conv1(x)
        conved1 = self.act1(self.ln1(conved1.permute(0,2,1)))
        x_ca = self.CA(conved1) + conved1
        conved2 = self.conv2(x_ca.permute(0,2,1))
        conved2 = self.act2(self.ln2(conved2.permute(0,2,1)))
        x_out = conved2.reshape(B,N,C)
        return x_out
class CAConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0,
        batch
    ):
        super().__init__()
        self.batch_size = batch
        self.CAB = ChannelFeatureBranch(chan1=dim)
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x_ca = self.CAB(x,batch= self.batch_size)
        x = self.ff1(x) + x
        x = 0.5*x + 0.5*x_ca
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
class CMKConMambaBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0,
        batch
    ):
        super().__init__()
        self.batch_size = batch
        self.CAB = ChannelFeatureBranch(chan1=dim)

        # self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.attn = PreNorm(dim, self.attn)
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state=dim,  # SSM state expansion factor
            d_conv=ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.Mamba = PreNorm(dim, self.MambaBlock)
        self.conv = MKGU( dim,dropout=0.2)
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x_ca = self.CAB(x,batch= self.batch_size)
        # x = self.ff1(x) + x
        x = x + x_ca
        x = self.Mamba(x) + x
        x = self.conv(x)  + x
        # x = self.ff2(x) + x
        x = self.post_norm(x)
        return x
class MCFBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0,
            batch
    ):
        super().__init__()
        self.batch_size = batch
        self.CAB1 = ChannelFeatureBranch(chan1=dim)
        # self.CAB2 = ChannelFeatureBranch(chan1=dim)
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state=dim,  # SSM state expansion factor
            d_conv=ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.Mamba = PreNorm(dim, self.MambaBlock)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.conv = MKGU(dim)
        self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)

        # self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, mask=None):

        # x = self.attn1(x) + x
        x_ca1 = self.CAB1(x,batch= self.batch_size)
        x = x + x_ca1
        x = self.drop(self.MambaBlock(x)) + x
        x = self.conv(x) + x
        # x = self.ff2(x) + x
        # x_ca2 = self.CAB2(x,batch= self.batch_size)
        # x = x + x_ca2
        x = self.ff1(x) + x
        x = self.post_norm(x)

        return x
class CBMambaformerBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0,
            batch
    ):
        super().__init__()
        self.batch_size = batch
        self.CAB = ChannelFeatureBranch(chan1=dim)
        # self.CAB2 = ChannelFeatureBranch(chan1=dim)
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state=dim,  # SSM state expansion factor
            d_conv=ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.Mamba = PreNorm(dim, self.MambaBlock)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)

        # self.ff2 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.post_norm = nn.LayerNorm(dim)
        # self.drop = nn.Dropout(0.2)

    def forward(self, x, mask=None):
        x_ca1 = self.CAB(x,batch= self.batch_size)
        x = self.ff1(x) + x
        x = x + x_ca1
        # x = self.attn1(x) + x
        # x_ca2 = self.CAB2(x,batch= self.batch_size)
        # x = x + x_ca2
        x = self.MambaBlock(x) + x
        x = self.conv(x) + x
        # x = self.ff2(x) + x
        x = self.post_norm(x)

        return x
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class GLU(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        out, gate = x.chunk(2, dim=self.dim)
        return out * gate.sigmoid()

class DepthWiseConv1d(nn.Module):
    def __init__(self, chan_in, chan_out, kernel_size, padding):
        super().__init__()
        self.padding = padding
        self.conv = nn.Conv1d(chan_in, chan_out, kernel_size, groups = chan_in)

    def forward(self, x):
        x = F.pad(x, self.padding)
        return self.conv(x)

# attention, feedforward, and conv module
class MKGU(nn.Module):
    def __init__(self, in_channel,causal = False,kernel_size = 31,dropout = 0.):
        super().__init__()
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)
        self.chan_proj = nn.Sequential(nn.LayerNorm(in_channel),nn.Linear(in_channel, in_channel*2),Swish(),Rearrange('b n c -> b c n'))
        self.c1 = nn.Conv1d(in_channel, in_channel//4, kernel_size=7, padding=3)
        self.c2 = nn.Conv1d(in_channel, in_channel//4, kernel_size=15, padding=7)
        self.c3 = nn.Conv1d(in_channel, in_channel//4, kernel_size=23, padding=11)
        self.c4 = nn.Conv1d(in_channel, in_channel//4, kernel_size=31, padding=15)
        self.bn = nn.BatchNorm1d(in_channel)
        self.depthwise_conv = nn.Sequential(DepthWiseConv1d(in_channel, in_channel, kernel_size=31, padding=padding),
                                            nn.BatchNorm1d(in_channel),Swish(),)
        self.drop = nn.Sequential(Rearrange('b c n -> b n c'),nn.Dropout(dropout))
    def forward(self, x):
        x = self.chan_proj(x)
        x_splitted = torch.split(x,x.shape[1]//2,dim=1)
        x_dc = x_splitted[0]
        x_mc = x_splitted[1]
        x_mc = torch.concat([self.c1(x_mc),self.c2(x_mc),self.c3(x_mc),self.c4(x_mc)],dim=1)
        x = F.silu(self.bn(self.depthwise_conv(x_mc)+x_mc )) * (x_dc)
        # x = (self.depthwise_conv(x_dc)+x_dc)*F.silu(self.bn(x_mc) )
        x = self.drop(x)
        return x

        # x = self.chan_proj(x)
        # x1,x2 = torch.chunk(x,2,dim=1)
        # x = torch.concat([self.c1(x1),self.c2(x1),self.c3(x1),self.c4(x1)],dim=1)
        # x = x * x2
        # x = self.depthwise_conv(x) + x
        # x = self.drop(x)
        # return x
class Scale(nn.Module):
    def __init__(self, scale, fn):
        super().__init__()
        self.fn = fn
        self.scale = scale

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)



class Attention(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)

        self.dropout = nn.Dropout(dropout)

    # def forward(self, x, context = None, mask = None, context_mask = None):
    def forward(self, x, context=None, mask=None, context_mask=None,Drop_key = False):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context) #n是t或f，h是头数
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))     #d是每个头的维度

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
        if Drop_key == True:
            m_r = torch.ones_like(dots) * 0.3
            dots = dots + torch.bernoulli(m_r) * -1e12
        attn = dots.softmax(dim = -1)
        # attn = torch.max(attn,attn1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)

class GRU_FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super(GRU_FFN, self).__init__()
        self.gru = GRU(d_model, d_model * 2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(d_model * 2 * 2, d_model)
        else:
            self.linear = Linear(d_model * 2, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        mult = 4,
        dropout = 0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            Swish(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class ConformerConvModule(nn.Module):
    def __init__(
        self,
        dim,
        causal = False,
        expansion_factor = 2,
        kernel_size = 31,
        dropout = 0.):
        super().__init__()

        inner_dim = dim * expansion_factor
        padding = calc_same_padding(kernel_size) if not causal else (kernel_size - 1, 0)

        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n c -> b c n'),
            nn.Conv1d(dim, inner_dim * 2, 1),
            GLU(dim=1),
            DepthWiseConv1d(inner_dim, inner_dim, kernel_size = kernel_size, padding = padding),
            nn.BatchNorm1d(inner_dim) if not causal else nn.Identity(),
            Swish(),
            nn.Conv1d(inner_dim, dim, 1),
            Rearrange('b c n -> b n c'),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)
class Attention2(nn.Module):
    def __init__(
        self,
        dim,
        heads = 8,
        dim_head = 64,
        dropout = 0.,
        max_pos_emb = 512
    ):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads= heads
        self.scale = dim_head ** -0.5
        self.to_q = nn.Linear(dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, dim)

        self.max_pos_emb = max_pos_emb
        self.rel_pos_emb = nn.Embedding(2 * max_pos_emb + 1, dim_head)
        softplus = nn.Softplus()
        self.dropout = nn.Dropout(dropout)

    # def forward(self, x, context = None, mask = None, context_mask = None):
    def forward(self, x, context=None, mask=None, context_mask=None,Drop_key = True):
        n, device, h, max_pos_emb, has_context = x.shape[-2], x.device, self.heads, self.max_pos_emb, exists(context) #n是t或f，h是头数
        context = default(context, x)

        q, k, v = (self.to_q(x), *self.to_kv(context).chunk(2, dim = -1))
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))     #d是每个头的维度

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        # shaw's relative positional embedding
        seq = torch.arange(n, device = device)
        dist = rearrange(seq, 'i -> i ()') - rearrange(seq, 'j -> () j')
        dist = dist.clamp(-max_pos_emb, max_pos_emb) + max_pos_emb
        rel_pos_emb = self.rel_pos_emb(dist).to(q)
        pos_attn = einsum('b h n d, n r d -> b h n r', q, rel_pos_emb) * self.scale
        dots = dots + pos_attn

        if exists(mask) or exists(context_mask):
            mask = default(mask, lambda: torch.ones(*x.shape[:2], device = device))
            context_mask = default(context_mask, mask) if not has_context else default(context_mask, lambda: torch.ones(*context.shape[:2], device = device))
            mask_value = -torch.finfo(dots.dtype).max
            mask = rearrange(mask, 'b i -> b () i ()') * rearrange(context_mask, 'b j -> b () () j')
            dots.masked_fill_(~mask, mask_value)
        if Drop_key == True:
            m_r = torch.ones_like(dots) * 0.3
            dots = dots + torch.bernoulli(m_r) * -1e12
        out_numerator = torch.ones_like(dots)+dots+(0.5+ 1e-7)*(dots**2)
        out_denominator = (int(dots.shape[3])) *(torch.ones_like(dots)+dots +(0.5+ 1e-7)*(dots**2))
        attn1 = torch.div(out_numerator, out_denominator)
        attn = dots.softmax(dim = -1)
        attn = torch.max(attn,attn1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)



# Conformer Block


class ConformerBlock1(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):

        x = self.ff1(x) + x
        #print(x.shape)
        x = self.attn(x, mask = mask) + x
        # x = self.conv(x) + x
        #
        # x = self.conv(x) + x
        x = self.ff2(x) + x
        x = self.attn1(x, mask=mask) + x
        x = self.ff3(x) + x
        x = self.post_norm(x)

        return x

class MamformerBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            LayerNorm_type='BiasFree',
            ff_dropout=0.,
            conv_dropout=0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.attn1 = Attention2(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        # self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
        #                                 kernel_size=conv_kernel_size, dropout=conv_dropout)
        # self.ffn = metFeedForward(dim, 2.66, bias=False)
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state= dim,  # SSM state expansion factor
            d_conv= ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.MambaBlock =PreNorm(dim, self.MambaBlock) #5.6 +scale
        self.drop = nn.Dropout(0.2)
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.post_norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        x = self.drop(self.MambaBlock(x)) + x
        x = self.attn(x, mask=mask) + x
        # b,n,c = x.shape
        # x = x.reshape(1,c,b,n)
        # x_a = self.ffn(self.norm1(x)) + x
        # x = x_a.reshape(b,n,c)
        # x = self.conv(x) + x

        # x = self.attn(x, mask=mask) + x

        # x0 =x

        x = self.ff1(x) + x
        x = self.post_norm(x)
        return x
class ConMambaBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
                                        kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state= dim,  # SSM state expansion factor
            d_conv= ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.MambaBlock = PreNorm(dim, self.MambaBlock)
        self.drop = nn.Dropout(0.2)
        # self.attn = PreNorm(dim, self.attn)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))

        self.post_norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None):

        x = self.drop(self.MambaBlock(x)) + x
        # x = self.attn(x, mask=mask) + x
        x = self.conv(x) + x

        # x = self.attn(x, mask=mask) + x
        x = self.ff1(x) + x
        # x = self.ff3(x) + x
        x = self.post_norm(x)
        return x
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0,device='cuda:1'):
        super(TransformerBlock, self).__init__()
        self.norm = [LayerNorm(d_model).to(device), LayerNorm(d_model).to(device),LayerNorm(d_model).to(device),]
        # self.norm = [RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5), RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5),
        #              RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5)]
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout)
        # self.ffn = SwiGLUConvDeconv1d(64, 256, 4, 1)
        self.ffn = GRU_FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm[0](x)
        xt, _ = self.attention(xt, xt, xt,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm[1](x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm[2](x)

        return x
class ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        # self.CFB = ChannelFeatureBranch(dim)
        # self.conv = MKGU(dim)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))
        # self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        # X_cb = self.CFB(x,batch = 4)
        # g = 0.5
        # x = g*x + (1-g)*X_cb
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = self.ff2(x) + x
        x = self.post_norm(x)

        return x
    
class reduceConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))
        self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x1 = self.ff1(x) - x
        x2 = self.attn(x1, mask = mask) + x -x1
        x3 = self.conv(x2)   -x2
        # x1 = self.ff1(x) + x
        # x2 = self.attn(x, mask = mask) *  self.act(x1)
        # x = self.conv(x2)   +x
        x4 = self.ff2(x3) + x2 -x3
        x = self.post_norm(x4)

        return x

class reduce1ConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))
        self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x1 = self.ff1(x) - x
        x2 = self.attn(x1, mask = mask) + x
        x3 = self.conv(x2)   -x2
        x4 = self.ff2(x3) + x2
        x = self.post_norm(x4)

        return x
class MKConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        self.conv = MKGU(dim)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.attn(x, mask = mask) + x
        x = self.conv(x)   +x
        x = self.ff2(x) + x
        x = self.post_norm(x)

        return x


class MambaformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()

        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state=dim,  # SSM state expansion factor
            d_conv=ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        # self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.Mamba = PreNorm(dim, self.MambaBlock)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        # x = self.attn1(x)   +x
        x = self.drop(self.MambaBlock(x)) + x
        x = self.ff2(x) + x
        x = self.post_norm(x)

        return x
class FConformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.CFA = CFTSA(dropout=0.2)
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        # self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.attn = PreNorm(dim, self.attn)
        self.CFA = PreNorm(dim,self.CFA)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))

        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):

        x = self.ff1(x) + x
        #print(x.shape)
        # x = self.attn(x, mask = mask) + x
        x = self.CFA(x) + x
        #
        # x = self.conv(x) + x
        x = self.ff2(x) + x
        # x = self.attn1(x, mask=mask) + x
        # x = self.ff3(x) + x
        x = self.post_norm(x)

        return x
class FMamformerBlock(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8,
            ff_mult=4,
            conv_expansion_factor=2,
            conv_kernel_size=31,
            attn_dropout=0.,
            ff_dropout=0.,
            conv_dropout=0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        # self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=conv_expansion_factor,
        #                                 kernel_size=conv_kernel_size, dropout=conv_dropout)
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state= dim,  # SSM state expansion factor
            d_conv= ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.CFA = CFTSA(dropout=0.2)
        self.CFA = PreNorm(dim,self.CFA)
        self.MambaBlock =PreNorm(dim, self.MambaBlock) #5.6 +scale
        self.drop = nn.Dropout(0.25)
        self.attn = PreNorm(dim, self.attn)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))

        self.post_norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None):

        x = self.drop(self.MambaBlock(x)) + x
        x = self.CFA(x)+x
        # x = self.attn(x, mask=mask)+x

        x = self.ff1(x) + x
        # x = self.ff3(x) + x
        x = self.post_norm(x)
        return x
class ConformerBlock_star(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 31,
        attn_dropout = 0.,
        ff_dropout = 0.,
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.attn = Attention(dim = dim, dim_head = dim_head, heads = heads, dropout = attn_dropout)
        # self.attn1 = Attention2(dim=dim, dim_head=dim//4, heads=4, dropout=attn_dropout)
        # self.attn = MultiDilatelocalAttention(dim=dim, num_heads=heads, attn_drop=attn_dropout)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn = PreNorm(dim, self.attn)
        # self.attn1 = PreNorm(dim, self.attn1)
        # self.attn1 = PreNorm(dim, self.attn1)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        # self.ff3 = Scale(0.5, PreNorm(dim, self.ff3))
        self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)

    def forward(self, x, mask = None):

        x = self.ff1(x) + x
        x11 = self.attn(x, mask = mask) + x
        x12= self.conv(x)  * x
        x = x12 + self.act(x11)
        x = self.ff2(x) + x
        x = self.post_norm(x)

        return x
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias





class metFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(metFeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class refine_att(nn.Module):
    """Convolutional relative position encoding."""
    def __init__(self, Ch, h, window):

        super().__init__()

        if isinstance(window, int):
            # Set the same window size for all attention heads.
            window = {window: h}
            self.window = window
        elif isinstance(window, dict):
            self.window = window
        else:

            raise ValueError()

        self.conv_list = nn.ModuleList()
        self.head_splits = []
        for cur_window, cur_head_split in window.items():
            dilation = 1  # Use dilation=1 at default.
            padding_size = (cur_window + (cur_window - 1) *
                            (dilation - 1)) // 2
            cur_conv = nn.Conv2d(
                cur_head_split * Ch*2,
                cur_head_split,
                kernel_size=(cur_window, cur_window),
                padding=(padding_size, padding_size),
                dilation=(dilation, dilation),
                groups=cur_head_split,
            )

            self.conv_list.append(cur_conv)
            self.head_splits.append(cur_head_split)
        self.channel_splits = [x * Ch*2 for x in self.head_splits]

    def forward(self, q,k, v, size):
        """foward function"""
        B, h, N, Ch = q.shape
        H, W = size

        # We don't use CLS_TOKEN
        q_img = q
        k_img = k
        v_img = v

        # Shape: [B, h, H*W, Ch] -> [B, h*Ch, H, W].
        q_img = rearrange(q_img, "B h (H W) Ch -> B h Ch H W", H=H, W=W)
        k_img = rearrange(k_img, "B h Ch (H W) -> B h Ch H W", H=H, W=W)
        qk_concat = torch.cat((q_img,k_img),2)
        qk_concat = rearrange(qk_concat, "B h Ch H W -> B (h Ch) H W", H=H, W=W)
        # Split according to channels.
        qk_concat_list = torch.split(qk_concat, self.channel_splits, dim=1)
        qk_att_list = [
            conv(x) for conv, x in zip(self.conv_list, qk_concat_list)
        ]

        qk_att = torch.cat(qk_att_list, dim=1)
        # Shape: [B, h*Ch, H, W] -> [B, h, H*W, Ch].
        qk_att = rearrange(qk_att, "B (h Ch) H W -> B h (H W) Ch", h=h)

        return qk_att


class metAttention(nn.Module):
    def __init__(self, dim, num_heads, bias,shared_refine_att=None,qk_norm=1):
        super(metAttention, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias),
            nn.Conv2d(dim * 2, dim, kernel_size=3, stride=1, padding=1,
                      groups=dim, bias=bias),
        )
        self.norm = qk_norm
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.sigmoid = nn.Sigmoid()
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        if num_heads == 8:
            crpe_window = {
                3: 2,
                5: 3,
                7: 3
            }
        elif num_heads == 1:
            crpe_window = {
                3: 1,
            }
        elif num_heads == 2:
            crpe_window = {
                3: 2,
            }
        elif num_heads == 4:
            crpe_window = {
                3: 2,
                5: 2,
            }
        self.refine_att = refine_att(Ch=dim // num_heads,
                                     h=num_heads,
                                     window=crpe_window)

    def forward(self, x):
        xsca = self.sca(x)
        x1 = self.dwconv(x)

        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)

        #q = torch.nn.functional.normalize(q, dim=-1)
        q_norm=torch.norm(q,p=2,dim=-1,keepdim=True)/self.norm+1e-6
        q=torch.div(q,q_norm)
        k_norm=torch.norm(k,p=2,dim=-2,keepdim=True)/self.norm+1e-6
        k=torch.div(k,k_norm)
        #k = torch.nn.functional.normalize(k, dim=-2)

        refine_weight = self.refine_att(q,k, v, size=(h, w))
        #refine_weight=self.Leakyrelu(refine_weight)
        refine_weight = self.sigmoid(refine_weight)
        attn = k@v

        out_numerator = torch.sum(v, dim=-2).unsqueeze(2)+(q@attn)
        out_denominator = torch.full((h*w,c//self.num_heads),h*w).to(q.device)\
                          +q@torch.sum(k, dim=-1).unsqueeze(3).repeat(1,1,1,c//self.num_heads)+1e-6

        out = torch.div(out_numerator, out_denominator) * self.temperature
        out = out * refine_weight
        out = rearrange(out, 'b head (h w) c-> b (head c) h w', head=self.num_heads, h=h, w=w)
        #csa
        out = out * xsca * F.gelu(x1)
        out = self.project_out(out)
        return out
class ConMETformerBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        LayerNorm_type='BiasFree',
        conv_dropout = 0.
    ):
        super().__init__()
        # self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.attn = metAttention(dim, 8, bias=False,shared_refine_att=None,qk_norm=1)
        self.attn1 = Attention2(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.MambaBlock = Mamba(
        #     # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
        #     d_model=dim,  # Model dimension d_model
        #     d_state=dim,  # SSM state expansion factor
        #     d_conv=4,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        # self.MambaBlock = PreNorm(dim, self.MambaBlock)
        # self.drop = nn.Dropout(0.2)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.attn1 = PreNorm(dim, self.attn1)
        self.post_norm = nn.LayerNorm(dim)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.inter = Interaction(chan2=dim,chan1=dim)
    def forward(self, x, mask = None):
        b,n,c = x.shape
        x0 =x
        x = x.view(1,c,b,n)
        x_a = self.attn(self.norm1(x)) + x
        x = x_a.view(b,n,c)
        # x = F.sigmoid(x_m * x_1)
        x_c = self.conv(x)   +x
        x = self.attn1(x_a.reshape(b,n,c)*F.sigmoid(x_c)) + x0
        x = self.post_norm(x)
        return x
class ConMet(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        LayerNorm_type='BiasFree',
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        # self.attn = metAttention(dim, 4, bias=False,shared_refine_att=None,qk_norm=1)
        self.ffn = metFeedForward(dim, 2.66, bias= False)
        # self.attn2 = metAttention(dim, 4, bias=False,shared_refine_att=None,qk_norm=1)
        # self.ffn2 = metFeedForward(dim, 2.66, bias= False)
        self.attn1 = Attention2(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.MambaBlock = Mamba(
        #     # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
        #     d_model=dim,  # Model dimension d_model
        #     d_state=dim,  # SSM state expansion factor
        #     d_conv=4,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        # self.MambaBlock = PreNorm(dim, self.MambaBlock)
        # self.drop = nn.Dropout(0.2)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)

        self.attn1 = PreNorm(dim, self.attn1)
        self.post_norm = nn.LayerNorm(dim)
        # self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        # self.inter = Interaction(chan2=dim,chan1=dim)
    def forward(self, x, mask = None):
        b,n,c = x.shape
        x = x.reshape(1,c,b,n)
        # x = self.attn(self.norm1(x)) + x
        x = self.ffn(self.norm2(x)) + x
        x = x.reshape(b,n,c)
        xa = self.attn1(x)+x
        x = self.conv(xa)   - xa
        x = self.ff1(xa*F.sigmoid(x)) + x
        # x = F.sigmoid(x_m * x_1)

        # x = x.reshape(1,c,b,n)x
        # x = self.attn2(self.norm1(x)) + x
        # x = self.ffn1(self.norm2(x)) + x
        # x = x.reshape(b,n,c)
        x = self.post_norm(x)
        return x
class ConMetBlock(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        ff_mult = 4,
        conv_expansion_factor = 2,
        conv_kernel_size = 16,
        attn_dropout = 0.,
        ff_dropout = 0.,
        LayerNorm_type='BiasFree',
        conv_dropout = 0.
    ):
        super().__init__()
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.attn = metAttention(dim, 8, bias=False,shared_refine_att=None,qk_norm=1)

        # self.MambaBlock = Mamba(
        #     # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
        #     d_model=dim,  # Model dimension d_model
        #     d_state=dim,  # SSM state expansion factor
        #     d_conv=4,  # Local convolution width
        #     expand=2,  # Block expansion factor
        # )
        # self.MambaBlock = PreNorm(dim, self.MambaBlock)
        # self.drop = nn.Dropout(0.2)
        self.conv = ConformerConvModule(dim = dim, causal = False, expansion_factor = conv_expansion_factor, kernel_size = conv_kernel_size, dropout = conv_dropout)
        # self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        # self.ff3 = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        # self.attn1 = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)
        # self.attn = PreNorm(dim, self.attn1)
        self.post_norm = nn.LayerNorm(dim)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        # self.inter = Interaction(chan2=dim,chan1=dim)
    def forward(self, x, mask = None):
        b,n,c = x.shape
        x0 =x
        x = x.view(1,c,b,n)
        x_a = self.attn(self.norm1(x)) + x
        x = x_a.view(b,n,c)
        # x = F.sigmoid(x_m * x_1)
        # x_a = self.attn(x) + x
        x_c = self.conv(x)   + x
        # x = self.ff1(x_a*F.sigmoid(x_c)) + x_c
        x = self.ff1(x_a.reshape(b,n,c)*F.sigmoid(x_c)) + x_c
        x = self.post_norm(x)
        return x
class metTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1):
        super(metTransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = metAttention(dim, num_heads, bias,shared_refine_att=shared_refine_att,qk_norm=qk_norm)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = metFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
class metConformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1):
        super(metConformerBlock, self).__init__()
        self.conv = ConformerConvModule(dim=dim, causal=False, expansion_factor=2,
                                        kernel_size=31, dropout=0)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = metAttention(dim, num_heads, bias,shared_refine_att=shared_refine_att,qk_norm=qk_norm)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = metFeedForward(dim, ffn_expansion_factor, bias)
        self.silu = nn.SiLU()
        # self.interact = Interaction(chan1=dim,chan2=dim)
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        # x_a = x
        b,c,h,w = x.shape
        x = x.view(b,h*w,c).contiguous()
        x = self.conv(x)+x
        x = x.reshape(b,c,h,w)
        x = self.silu(self.norm2(x))
        # x = self.interact(x_a,x_c)
        return x
##########################################################################
class mmetTransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type,shared_refine_att=None,qk_norm=1):
        super(mmetTransformerBlock, self).__init__()
        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state=dim,  # SSM state expansion factor
            d_conv=4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.MambaBlock =PreNorm(dim, self.MambaBlock)
        self.drop = nn.Dropout(0.2)
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = metAttention(dim, num_heads, bias,shared_refine_att=shared_refine_att,qk_norm=qk_norm)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = metFeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        #add Mamba
        b,c,t,f = x.shape
        x = x.view(b,t*f,c)
        x = self.drop(self.MambaBlock(x))+x
        x = x.view(b,c,t,f)

        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
def main():
    x = torch.ones(4, 100, 64).cuda()
    conformer = MKConformerBlock(dim=64).cuda()
    # conformer = CBMambaformerBlock(dim=64,batch=4).cuda()
    print(conformer)
    print(x.shape)
    print(conformer(x).shape)

if __name__ == '__main__':
    main()