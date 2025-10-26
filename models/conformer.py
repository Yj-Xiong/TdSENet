import math
import numbers
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from utils import *
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout
from modules.mamba_simple import Mamba


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def calc_same_padding(kernel_size):
    pad = kernel_size // 2
    return (pad, pad - (kernel_size + 1) % 2)

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
        x = 0.5*self.ff2(x) + x
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
        self.attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout)

        self.MambaBlock = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=dim,  # Model dimension d_model
            d_state= dim,  # SSM state expansion factor
            d_conv= ff_mult,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        self.MambaBlock =PreNorm(dim, self.MambaBlock) #5.6 +scale
        self.drop = nn.Dropout(0.2)
        self.attn = PreNorm(dim, self.attn)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.post_norm = nn.LayerNorm(dim)
    def forward(self, x, mask=None):
        x = self.drop(self.MambaBlock(x)) + x
        x = self.attn(x, mask=mask) + x
        # b,n,c = x.shape


        x = self.ff1(x) + x
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
        self.ff2 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.Mamba = PreNorm(dim, self.MambaBlock)
        self.ff1 = FeedForward(dim = dim, mult = ff_mult, dropout = ff_dropout)
        self.ff1 = Scale(0.5, PreNorm(dim, self.ff1))
        self.ff2 = Scale(0.5, PreNorm(dim, self.ff2))
        self.act = nn.SiLU()
        self.post_norm = nn.LayerNorm(dim)
        self.drop = nn.Dropout(0.2)

    def forward(self, x, mask = None):
        x = self.ff1(x) + x
        x = self.drop(self.MambaBlock(x)) + x
        x = self.ff2(x) + x
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





def main():
    x = torch.ones(4, 100, 64).cuda()
    conformer = ConformerBlock(dim=64).cuda()
    print(conformer)
    print(x.shape)
    print(conformer(x).shape)

if __name__ == '__main__':
    main()
