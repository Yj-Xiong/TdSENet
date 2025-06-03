# ------------------------------------------------------------------------
# Copyright (c) 2022 Murufeng. All Rights Reserved.
# ------------------------------------------------------------------------
'''
@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


# from basicsr.models.archs.arch_util import LayerNorm2d
# from basicsr.models.archs.local_arch import Local_Base
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        B, C, T = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1) * y + bias.view(1, C, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        B, C, T = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=2).sum(dim=0), grad_output.sum(dim=2).sum(
            dim=0), None


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


#
# class ffn(nn.Module):
#     def __init__(self, in_channels, FFN_Expand=2,  dropout=0.):
#         super(ffn, self).__init__()
#
#         self.sg = SimpleGate()
#
#         ffn_channel = FFN_Expand * in_channels
#         self.conv4 = nn.Conv1d(in_channels=in_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1,
#                                groups=1,
#                                bias=True)
#         self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=in_channels, kernel_size=1, padding=0,
#                                stride=1,
#                                groups=1, bias=True)
#
#
#         self.norm2 = LayerNorm1d(in_channels)
#
#
#         self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
#
#
#         self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
#
#     def forward(self, x):
#         inp3 = x
#         x = self.conv4(self.norm2(inp3))
#         x = self.sg(x)
#         x = self.conv5(x)
#
#         x = self.dropout2(x)
#         x = inp3 + x * self.gamma
#         return x


class LayerNorm1d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm1d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class LKFCA_Block(nn.Module):
    def __init__(self, in_channels, DW_Expand=2, FFN_Expand=1, drop_out_rate=0., type='sca', kernel_size=31):
        super().__init__()

        dw_channel = in_channels * DW_Expand
        # ConvModule
        # self.lnorm0 = LayerNorm1d(in_channels)
        # self.cmconv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel*2, kernel_size=1)
        # self.cmconv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=kernel_size,
        #           padding=get_padding(kernel_size), groups=dw_channel)  # DepthWiseConv1d
        # self.cmnorm = nn.InstanceNorm1d(dw_channel, affine=True)
        # self.cmconv3 = nn.Conv1d(in_channels=in_channels, out_channels=in_channels, kernel_size=1)

        # 注意力

        # self.inter = int(dw_channel // 4)

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1,
                               bias=True)
        self.conv2 = nn.Conv1d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv1d(in_channels=dw_channel // 2, out_channels=in_channels, kernel_size=1, padding=0,
                               stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.type = type
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        # ffn_channel = FFN_Expand * in_channels
        # self.conv4 = nn.Conv1d(in_channels=in_channels, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv1d(in_channels=ffn_channel // 2, out_channels=in_channels, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm1d(in_channels)
        self.norm2 = LayerNorm1d(in_channels)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.lamda = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channels, 1)), requires_grad=True)
        # self.LKEFN = FeedForward(in_channels, FFN_Expand, kernel_size=kernel_size, bias=False)
        self.GCGFN = GCGFN(in_channels)

    def forward(self, x):
        inp2 = x
        # x = self.lnorm0(x)
        # x = self.cmconv1(x)
        # x = self.sg(x)
        # x = self.cmconv2(x)
        # x = self.cmnorm(x)
        # x = self.sg(x)
        # x = self.cmconv3(x)
        # inp2 = inp + x * self.lamda

        x = self.norm1(inp2)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        inp3 = inp2 + x * self.beta

        x = self.GCGFN(inp3)
        # x = self.sg(x)
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)
        # x = inp3 + x * self.gamma

        # # x = self.norm1(x)#末尾再加一层LN
        # x = self.Rearrange2(x)
        return x


class GCGFN(nn.Module):
    def __init__(self, n_feats, fnn_expend=4):
        super().__init__()
        i_feats = fnn_expend * n_feats

        self.n_feats = n_feats
        self.i_feats = i_feats

        self.norm = LayerNorm1d(n_feats)
        self.scale = nn.Parameter(torch.zeros((1, n_feats, 1)), requires_grad=True)

        # Multiscale Large Kernel Attention (replaced with 1D convolutions)
        self.LKA9 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA7 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA5 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.LKA3 = nn.Sequential(
            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4),

            nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=1))

        self.X3 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=3, padding=get_padding(3), groups=i_feats // 4)
        self.X5 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=11, padding=get_padding(11), groups=i_feats // 4)
        self.X7 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=23, padding=get_padding(23), groups=i_feats // 4)
        self.X9 = nn.Conv1d(i_feats // 4, i_feats // 4, kernel_size=31, padding=get_padding(31), groups=i_feats // 4)

        self.proj_first = nn.Sequential(
            nn.Conv1d(n_feats, i_feats, kernel_size=1))

        self.proj_last = nn.Sequential(
            nn.Conv1d(i_feats, n_feats, kernel_size=1))

    def forward(self, x):
        shortcut = x.clone()
        x = self.norm(x)
        x = self.proj_first(x)
        # a, x = torch.chunk(x, 2, dim=1)
        a_1, a_2, a_3, a_4 = torch.chunk(x, 4, dim=1)
        x = torch.cat([self.LKA3(a_1) * self.X3(a_1), self.LKA5(a_2) * self.X5(a_2), self.LKA7(a_3) * self.X7(a_3),
                       self.LKA9(a_4) * self.X9(a_4)], dim=1)
        x = self.proj_last(x) * self.scale + shortcut
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, kernel_size=3, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv1d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv1d(hidden_features * 2, hidden_features * 2, kernel_size=kernel_size, stride=1,
                                padding=get_padding(kernel_size),
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv1d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class GPFCA(nn.Module):

    def __init__(self, in_channels, num_blocks=2):
        super().__init__()

        self.naf_blocks = nn.ModuleList([LKFCA_Block(in_channels) for _ in range(num_blocks)])

        # self.norm1 = LayerNorm1d(in_channels)
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')

    def forward(self, x):
        # x = self.norm1(x)

        x = self.Rearrange1(x)

        for block in self.naf_blocks:
            x = block(x)

        x = self.Rearrange2(x)

        return x


class PolaLinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1,
                 kernel_size=5, alpha=4, max_pos_embeddings=321):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim

        self.qg = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)

        self.power = nn.Parameter(torch.zeros(size=(1, self.num_heads, 1, self.head_dim)))
        self.alpha = alpha

        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, max_pos_embeddings, dim)))
        nn.init.uniform_(self.positional_encoding, -0.1, 0.1)
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, alpha, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q, g = self.qg(x).reshape(B, N, 2, C).unbind(2)
        # 使用循环位置编码
        positional_encoding = self.positional_encoding.repeat(1, int(torch.ceil(
            N / self.positional_encoding.shape[1]).item()), 1)
        positional_encoding = positional_encoding[:, :N, :]
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        n = k.shape[1]
        # print(k.shape)
        k = k + positional_encoding
        kernel_function = nn.ReLU()

        scale = nn.Softplus()(self.scale)
        power = 1 + self.alpha * torch.sigmoid(self.power)

        q = q / scale
        k = k / scale
        q = q.reshape(B, N, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()
        v = v.reshape(B, n, self.num_heads, -1).permute(0, 2, 1, 3).contiguous()

        q_pos = kernel_function(q) ** power
        q_neg = kernel_function(-q) ** power
        k_pos = kernel_function(k) ** power
        k_neg = kernel_function(-k) ** power

        q_sim = torch.cat([q_pos, q_neg], dim=-1)
        q_opp = torch.cat([q_neg, q_pos], dim=-1)
        k = torch.cat([k_pos, k_neg], dim=-1)

        v1, v2 = torch.chunk(v, 2, dim=-1)

        z = 1 / (q_sim @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v1 * (n ** -0.5))
        x_sim = q_sim @ kv * z
        z = 1 / (q_opp @ k.mean(dim=-2, keepdim=True).transpose(-2, -1) + 1e-6)
        kv = (k.transpose(-2, -1) * (n ** -0.5)) @ (v2 * (n ** -0.5))
        x_opp = q_opp @ kv * z

        x = torch.cat([x_sim, x_opp], dim=-1)
        x = x.transpose(1, 2).reshape(B, N, C)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.transpose(-2, -1).reshape(B * self.num_heads, -1, n), size=N,
                                          mode='linear').reshape(B, self.num_heads, -1, N).transpose(-2, -1)

        v = v.reshape(B * self.num_heads, H, W, -1).view(B, H * W, N, self.num_heads)
        v = self.dwc(v).reshape(B, C, N).permute(0, 2, 1)
        x = x + v
        x = x * g

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


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


class Attn_Residual(nn.Module):
    def __init__(self, fn, scale_factor=1.0):
        super(Attn_Residual, self).__init__()
        self.fn = fn
        self.scale_factor = scale_factor

    def forward(self, x):
        return x + self.fn(x, 4, 4) * self.scale_factor


class PAGPF(nn.Module):

    def __init__(self, in_channels, ):
        super().__init__()
        self.Rearrange1 = Rearrange('b n c -> b c n')
        self.Rearrange2 = Rearrange('b c n -> b n c')
        self.norm1 = nn.LayerNorm(in_channels)
        self.PL_attn = PolaLinearAttention(dim=64, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
                                           proj_drop=0., sr_ratio=1,
                                           kernel_size=5, alpha=3)
        self.PL_attn = Attn_Residual(self.PL_attn, 1)
        self.GCGFN1 = nn.Sequential(self.Rearrange1,
                                    GCGFN(in_channels),
                                    self.Rearrange2)
        self.GCGFN2 = nn.Sequential(self.Rearrange1,
                                    GCGFN(in_channels),
                                    )

        # self.norm2 = nn.LayerNorm(in_channels)
        # self.ff_block1 = nn.Sequential(Residual(self.norm1,
        #                                PolaLinearAttention(dim=64,  num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
        #                               proj_drop=0., sr_ratio=1,
        #                               kernel_size=5, alpha=3),1),
        #                                self.Rearrange1,
        #                                GCGFN(in_channels),
        #                                self.Rearrange2)
        # self.ff_block2 = nn.Sequential(Residual(self.norm2,
        #                                PolaLinearAttention(dim=64,  num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0.,
        #                               proj_drop=0., sr_ratio=1,
        #                               kernel_size=5, alpha=3),1),
        #                                self.Rearrange1,
        #                                GCGFN(in_channels),
        #                                self.Rearrange2)
        # self.net = self.ff_block1 #if self.ff_block2 is None else nn.Sequential(self.ff_block1,self.ff_block2)
        self.post_norm = BiasNorm(in_channels)

    def forward(self, x):
        # x = self.norm1(x)
        x = self.GCGFN1(x)
        x = self.norm1(x)
        x = self.PL_attn(x)
        x = self.GCGFN2(x)
        x = self.post_norm(x)
        return self.Rearrange2(x)