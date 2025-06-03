# from modules.mamba_simple import Mamba
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import einsum
from torch.nn import MultiheadAttention, GRU, Linear, LayerNorm, Dropout


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


class RMSGroupNorm(nn.Module):
    def __init__(self, num_groups, dim,device, eps=1e-8, bias=False,):
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
class FFN(nn.Module):
    def __init__(self, d_model, bidirectional=True, dropout=0):
        super(FFN, self).__init__()
        self.gru = GRU(d_model, d_model*2, 1, bidirectional=bidirectional)
        if bidirectional:
            self.linear = Linear(d_model*2*2, d_model)
        else:
            self.linear = Linear(d_model*2, d_model)
        self.dropout = Dropout(dropout)
    
    def forward(self, x):
        self.gru.flatten_parameters()
        x, _ = self.gru(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x

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
class MambaFFN(nn.Module):
    def __init__(self, d_model,gpu,dropout=0):
        super(MambaFFN, self).__init__()
        self.Mamba= Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=d_model,  # Model dimension d_model
            d_state= 1*d_model,  # SSM state expansion factor
            d_conv= 4,  # Local convolution width
            expand=2,  # Block expansion factor
        )
        # if bidirectional:
        #     self.linear = Linear(d_model * 2 * 2, d_model)
        # else:
        #     self.linear = Linear(d_model * 2, d_model)
        self.ff1 = SwiGLUConvDeconv1d(64,256,4,1)
        self.dropout = Dropout(dropout)
        self.rmsg_norm = RMSGroupNorm(num_groups=4, dim=d_model, eps=1e-5,device=gpu)
    def forward(self, x):
        # self.Mamba.flatten_parameters()
        x = self.Mamba(x)
        x = F.leaky_relu(x)
        x = self.dropout(x)
        x = self.rmsg_norm(x)
        x += 0.5*self.ff1(x)

        return x

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, bidirectional=True, dropout=0):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attention = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout1 = Dropout(dropout)
        
        self.norm2 = LayerNorm(d_model)
        self.ffn = FFN(d_model, bidirectional=bidirectional)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt, _ = self.attention(xt, xt, xt,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask)
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x

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
        attn = dots.softmax(dim = -1)
        # attn = torch.max(attn,attn1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return self.dropout(out)
class TransformambaBlock(nn.Module):
    def __init__(self, d_model, n_heads,gpu_id, dropout=0):
        super(TransformambaBlock, self).__init__()

        self.norm1 = LayerNorm(d_model)
        self.attn = Attention(dim = d_model, dim_head = d_model//n_heads, heads = n_heads, dropout = dropout)
        self.dropout1 = Dropout(dropout)

        self.norm2 = LayerNorm(d_model)
        self.ffn = MambaFFN(d_model, gpu=gpu_id)
        self.dropout2 = Dropout(dropout)

        self.norm3 = LayerNorm(d_model)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        xt = self.norm1(x)
        xt=self.attn(xt, mask=attn_mask) + x
        x = x + self.dropout1(xt)

        xt = self.norm2(x)
        xt = self.ffn(xt)
        x = x + self.dropout2(xt)

        x = self.norm3(x)

        return x
def main():
    x = torch.randn(4, 64, 401, 201)
    b, c, t, f = x.size()
    x = x.permute(0, 3, 2, 1).contiguous().view(b, f*t, c)
    transformer = TransformerBlock(d_model=64, n_heads=4)
    x = transformer(x)
    x =  x.view(b, f, t, c).permute(0, 3, 2, 1)
    print(x.size())


if __name__ == '__main__':
    main()