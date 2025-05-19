import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from util import *

from modules.mamba_simple import Mamba
from .conformer import ConformerBlock, PreNorm, ConformerConvModule


# from ..utils import LearnableSigmoid

# class TimeRecursive_Mamba(nn.Module):#双
#     def __init__(self,  channel_num, dim_state, conv_width,expand):
#         super(TimeRecursive_Mamba, self).__init__()
#         self.MambaBlock = Mamba(
#     #uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
#     d_model=channel_num, # Model dimension d_model
#     d_state=dim_state,  # SSM state expansion factor
#     d_conv=conv_width,    # Local convolution width
#     expand=expand,    # Block expansion factor
# )
#         # self.t_len = time_length
#         self.post_norm = nn.LayerNorm(channel_num)
#         self.drop = nn.Dropout(0.2)
#     def forward(self, x_0):
#         x_t0 = x_0
#         x_tm1 = self.MambaBlock(x_t0)
#         x_t1 = self.post_norm(self.drop(x_tm1)) +x_t0
#         x_t01 = torch.cat((x_t0,x_t1),dim=1)
#         x_tm2 = self.MambaBlock(x_t01)
#         x_t01 = x_tm2[:,x_tm2.shape[1]//2:,:]
#         x_t = self.post_norm(self.drop(x_t01)) + x_t0
#         return x_t
class tfEnergy(nn.Module):   #CRA 通道循环能量注意力 CREA
    def __init__(self,channels):
        super(tfEnergy, self).__init__()
        self.fc1=nn.Linear(channels,int(channels*2))
        self.fc2=nn.Linear(int(channels*2),channels)
        # self.conv = nn.Conv2d(channels,channels,1)
        self.bn = nn.InstanceNorm2d(affine= True,num_features=channels)
        self.act= LearnableSigmoid(channels)
        # self.tanh = nn.Tanh()
    def forward(self, x):
        x0 = x
        b,c,t,f = x0.shape
        x_n = self.bn(x)
        x = x_n.view(b,c,t*f)
        x = torch.sqrt(torch.mean(x**2,dim=-1))  #b,c,1
        x = self.fc2(self.fc1(x))
        x = self.act(x).unsqueeze(2)
        # print(x.shape)
        x= torch.repeat_interleave(x,(t*f),dim=2).reshape_as(x0)
        q = x0
        k = torch.div(x_n,x)
        qk =  q*k
        v =   F.relu(self.bn(qk))
        # print(x.shape)
        return v
class TimeRecursive_Mamba(nn.Module):
    def __init__(self, time_length , channel_num, dim_state, conv_width,expand):
        super(TimeRecursive_Mamba, self).__init__()
        self.MambaBlock1 = Mamba(
    #uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
    d_model=channel_num, # Model dimension d_model
    d_state=dim_state,  # SSM state expansion factor
    d_conv=conv_width,    # Local convolution width
    expand=expand,    # Block expansion factor
)
        self.MambaBlock2 = Mamba(
            # uses roughly 3 * expand * d_model^2 parameters    x_in.shape = (batch, length, dim)
            d_model=channel_num,  # Model dimension d_model
            d_state=2*dim_state,  # SSM state expansion factor
            d_conv=2*conv_width,  # Local convolution width
            expand=expand,  # Block expansion factor
        )
        self.act = nn.ReLU()
        self.post_norm = nn.LayerNorm(channel_num)
        self.drop = nn.Dropout(0.2)
        self.t_len = time_length
        self.MambaBlock1 = PreNorm(channel_num,self.MambaBlock1)
        self.MambaBlock2 = PreNorm(channel_num, self.MambaBlock2)
        self.conv = ConformerConvModule(dim = channel_num, causal = False, expansion_factor = 2, kernel_size = 31, dropout = 0.)
    def forward(self, x_t0):
        skip1 = x_t0
        out_final = x_t0
        out_f = 0
        for i in range (self.t_len):
            out = self.MambaBlock1(skip1)
            out= self.post_norm(self.drop(out)) + skip1
            out_ = torch.cat((out_final, out), dim=1)
            out = self.MambaBlock2(out_)
            skip1 = self.post_norm(self.drop(out))+ out_
            # skip1 = (skip1[:, skip1.shape[1] // 2:, :]+skip1[:, :skip1.shape[1] // 2, :])/2
            skip1 = skip1[:, skip1.shape[1] // 2:, :]
            out_final = skip1
            # out_final = self.act(skip1 + out1)
            out_final = self.conv(out_final) + out_final
            out_final = self.act(self.post_norm(out_final))
            out_f += out_final
            out_f = self.post_norm(out_f)
        return out_f

#累加cat
class FiLM1C(nn.Module):        #沿通道仿射变换
    def __init__(self, input_size, feature_size):
        super(FiLM1C, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
      #  dil=1*2*4*8
        self.conv_1 = nn.Sequential(
            RefConv(3, 64, 3, (1, 1)),
            nn.InstanceNorm2d(64, affine=True),  # 实例归一化
            nn.PReLU(64)  # PReLU激活
        )
    def forward(self, input, context):
        #print(input.shape)
        context=self.conv_1(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
       # print(context.shape)
        gamma = self.gamma(context)  # 生成调制参数
     #   print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM1F(nn.Module):    ##沿频率仿射变换
    def __init__(self, input_size, feature_size):
        super(FiLM1F, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        #  dil=1*2*4*8
        self.conv_1 = nn.Sequential(
            RefConv(3, 64, 3, (1, 1)),
            nn.InstanceNorm2d(64, affine=True),  # 实例归一化
            nn.PReLU(64)  # PReLU激活
        )

    def forward(self, input, context):
        # print(input.shape)
        context = self.conv_1(context).contiguous()
       # print(context.shape)
        context = context.view(-1, context.shape[3])

        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM2C(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FiLM2C, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
      #  dil=1*2*4*8
        self.conv_2 = nn.Sequential(
            RefConv(64, 64, 3, (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
    def forward(self, input, context):
        #print(input.shape)
        context=self.conv_2(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
        gamma = self.gamma(context)  # 生成调制参数
       # print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input

class FiLM2F(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FiLM2F, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        #  dil=1*2*4*8
        self.conv_2 = nn.Sequential(
            RefConv(64, 64, 3, (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )

    def forward(self, input, context):
        # print(input.shape)
        context = self.conv_2(context).contiguous()
        # print(context.shape)
        context = context.view(-1, context.shape[3])
        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM11(nn.Module):        #沿通道仿射变换
    def __init__(self, input_size, feature_size):
        super(FiLM11, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
      #  dil=1*2*4*8
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1), (1, 1)),
            nn.InstanceNorm2d(64, affine=True),  # 实例归一化
            nn.PReLU(64)  # PReLU激活
        )
    def forward(self, input, context):
        #print(input.shape)
        context=self.conv_1(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
       # print(context.shape)
        gamma = self.gamma(context)  # 生成调制参数
     #   print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM12(nn.Module):    ##沿频率仿射变换
    def __init__(self, input_size, feature_size):
        super(FiLM12, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        #  dil=1*2*4*8
        self.conv_1 = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1), (1, 1)),
            nn.InstanceNorm2d(64, affine=True),  # 实例归一化
            nn.PReLU(64)  # PReLU激活
        )

    def forward(self, input, context):
        # print(input.shape)
        context = self.conv_1(context).contiguous()
       # print(context.shape)
        context = context.view(-1, context.shape[3])
        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class FiLM21(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FiLM21, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)   # 用于生成偏移参数的全连接层
      #  dil=1*2*4*8
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),
            nn.PReLU(64)
        )
    def forward(self, input, context):
        #print(input.shape)
        context=self.conv_2(context).contiguous()
       # print(context.shape)
        context=context.view(-1,context.shape[1])
        gamma = self.gamma(context)  # 生成调制参数
       # print(gamma.shape)
        beta = self.beta(context)    # 生成偏移参数
       # print(beta.shape)
        gamma=gamma.view(input.shape[0],input.shape[1],input.shape[2],input.shape[3])
       # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input

class FiLM22(nn.Module):
    def __init__(self, input_size, feature_size):
        super(FiLM22, self).__init__()
        self.gamma = nn.Linear(input_size, feature_size)  # 用于生成调制参数的全连接层
        self.beta = nn.Linear(input_size, feature_size)  # 用于生成偏移参数的全连接层
        #  dil=1*2*4*8
        self.conv_2 = nn.Sequential(
            nn.Conv2d(64, 64, (1, 3), (1, 2), padding=(0, 1)),
            nn.InstanceNorm2d(64, affine=True),

            nn.PReLU(64)
        )

    def forward(self, input, context):
        # print(input.shape)
        context = self.conv_2(context).contiguous()
        # print(context.shape)
        context = context.view(-1, context.shape[3])

        gamma = self.gamma(context)  # 生成调制参数
        # print(gamma.shape)
        beta = self.beta(context)  # 生成偏移参数
        # print(beta.shape)
        gamma = gamma.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # print(gamma)
        beta = beta.view(input.shape[0], input.shape[1], input.shape[2], input.shape[3])
        # 对特征映射进行调制
        modulated_input = input * gamma + beta

        return modulated_input
class GlobleCModule(nn.Module):
    def __init__(self,channels):
        super(GlobleCModule, self).__init__()
        self.ws=nn.Linear(channels,channels)
        self.wg=nn.Linear(channels,channels)
        self.pool = nn.Sequential(
                                  Rearrange('f t c -> c t f '),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c t f -> f c t '),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('f c t -> f t c'),
                                  )
        self.act= LearnableSigmoid(channels)
    def forward(self, x):
        input_p=self.pool(x)
        ws=self.ws(input_p)     # input_p.shape = 1，1,fea_num
        wg=self.wg(input_p)
        input_ws = ws * input_p
        input_wg = wg * input_p
        # input_wsg = wg * input_ws
        a= x
        b= (self.act(input_ws) + input_wg)
        moduled = torch.mul(a, b.expand_as(a))
        return moduled
class Fmodule(nn.Module):
    def __init__(self, fea_num):
        super(Fmodule, self).__init__()
        self.ws=nn.Linear(fea_num,fea_num)
        self.wg=nn.Linear(fea_num,fea_num)
        self.pool = nn.Sequential(nn.AdaptiveAvgPool1d(1),
                                  Rearrange('b f c -> c f b'),
                                  nn.AdaptiveAvgPool1d(1),
                                  Rearrange('c f b -> b c f '),
                                  )
        self.act= LearnableSigmoid(101)
    def forward(self, x):
        input_p=self.pool(x)
        # print(x.shape)# = 321,101,64
        # print(input_p.shape)    #1,1,101
        ws=self.ws(input_p)     # input_p.shape = 1，1,fea_num
        wg=self.wg(input_p)
        input_ws = ws * input_p
        input_wg = wg * input_p
        # input_wsg = wg * input_ws
        a= x.permute(0,2,1)
        b= (self.act(input_ws) + input_wg)
        moduled = torch.mul(a, b.expand_as(a))
        return moduled
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
class TSCB(nn.Module):
    def __init__(self, num_channel=64,drop=0.):
        super(TSCB, self).__init__()
        self.time_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=drop, ff_dropout=drop)
        self.freq_conformer = ConformerBlock(dim=num_channel, dim_head=num_channel//4, heads=4,
                                             conv_kernel_size=31, attn_dropout=drop, ff_dropout=drop)

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
     #Origin
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


class TSCNet(nn.Module):
    def __init__(self,num_channel=64, num_features=201):
        super(TSCNet, self).__init__()
        self.dense_encoder = DenseEncoder(in_channel=3, channels=num_channel)
        self.TSCB_1 = TSCB(num_channel=num_channel,drop=0.2)
        self.TSCB_2 = TSCB(num_channel=num_channel,drop=0.2)
        self.TSCB_3 = TSCB(num_channel=num_channel,drop=0.2)
        self.TSCB_4 = TSCB(num_channel=num_channel,drop=0.2)
        self.mask_decoder = MaskDecoder(num_features, num_channel=num_channel, out_channel=1)
        self.complex_decoder = ComplexDecoder(num_channel=num_channel)
    def forward(self, x):
        mag = torch.sqrt(x[:, 0, :, :]**2 + x[:, 1, :, :]**2).unsqueeze(1) #切片操作分别选择了 x 张量中第二个维度（通常用于表示实部和虚部）中的第一个和第二个通道
        noisy_phase = torch.angle(torch.complex(x[:, 0, :, :], x[:, 1, :, :])).unsqueeze(1)
        x_in = torch.cat([mag, x], dim=1) #(幅度和相位沿第二个维度拼接)
        out_1 = self.dense_encoder(x_in)
        out_2 = self.TSCB_1(out_1)
        out_3 = self.TSCB_2(out_2)
        out_4 = self.TSCB_3(out_3)
        out_5 = self.TSCB_4(out_4)

        mask = self.mask_decoder(out_5)
        out_mag = mask * mag
        complex_out = self.complex_decoder(out_5)
        mag_real = out_mag * torch.cos(noisy_phase)
        mag_imag = out_mag * torch.sin(noisy_phase)
        final_real = mag_real + complex_out[:, 0, :, :].unsqueeze(1)
        final_imag = mag_imag + complex_out[:, 1, :, :].unsqueeze(1)

        return final_real, final_imag
















