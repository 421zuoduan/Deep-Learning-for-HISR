import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch.nn import functional as F
from fvcore.nn import FlopCountAnalysis, flop_count_table
import numpy as np
from math import sqrt

def ka_window_partition(x, window_size):
    """
    input: (B, H*W, C)
    output: (num_windows, B, C, window_size, window_size)
    """
    B, L, C = x.shape
    H, W = int(sqrt(L)), int(sqrt(L))
    x = x.reshape(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(1, 3, 0, 5, 2, 4).contiguous().view(-1, B, C, window_size, window_size)
    return windows


def ka_window_reverse(windows, window_size, H, W):
    """
    input: (num_windows, B, C, window_size, window_size)
    output: (B, H*W, C)
    """
    B = windows.shape[1]
    x = windows.contiguous().view(H // window_size, W // window_size, B, -1, window_size, window_size)
    x = x.permute(2, 0, 4, 1, 5, 3).contiguous().view(B, H*W, -1)
    return x


class SELayer_KA(nn.Module):
    def __init__(self, channel):
        super(SELayer_KA, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(channel, channel // 16 if channel >= 64 else channel, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(channel // 16 if channel >= 64 else channel, channel, kernel_size=1),
                                nn.Sigmoid(), )

    def forward(self, x):
        channel_weight = self.se(x)
        x = x * channel_weight
        return x
    

class ConvLayer(nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ConvLayer, self).__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=stride, padding=padding)

    def forward(self, x, kernels=None):

        x = self.conv(x)

        return x, self.conv.weight.view(self.conv.weight.shape[0], self.conv.weight.shape[1], -1).permute(0, 2, 1)

class KernelAttention(nn.Module):
    """
    第一个分组卷积产生核，然后计算核的自注意力，调整核，第二个分组卷积产生输出，skip connection
    
    Args:
        dim: 输入通道数
        window_size: 窗口大小
        num_heads: 注意力头数
        qkv_bias: 是否使用偏置
        qk_scale: 缩放因子
        attn_drop: 注意力dropout
        proj_drop: 输出dropout
        ka_window_size: kernel attention window size
        kernel_size: 卷积核大小
        kernel_dim_scale: 卷积核通道数缩放因子, 未使用
        stride: 卷积步长
        padding: 卷积padding
    """

    def __init__(self, dim, input_resolution, num_heads, ka_win_num=4, kernel_size=3, stride=1, padding=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.win_num = ka_win_num
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        self.scale = qk_scale or (dim//num_heads) ** (-0.5)
        self.window_size = int(input_resolution // sqrt(ka_win_num))

        self.num_layers = self.win_num
        self.convlayers = nn.ModuleList()
        self.selayers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ConvLayer(self.dim, self.kernel_size, stride=stride, padding=padding)
            self.convlayers.append(layer)
        for j_layer in range(self.num_layers):
            layer = SELayer_KA(self.dim)
            self.selayers.append(layer)

        self.proj_qkv = nn.Linear(self.dim, self.dim*3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_out = nn.Linear(self.dim, self.dim)


    def forward(self, x):
        """
        x: B, L, C
        """
        B, L, C = x.shape
        H, W = self.input_resolution, self.input_resolution

        # x_windows:  win_num, bs, c, wh, ww
        x_windows = ka_window_partition(x, self.window_size)


        ### 下面对每个窗口进行卷积并获取卷积核
        # TODO: 这里如何写成并行运算的方法
        i = 0
        kernels = []
        
        for convlayer in self.convlayers:
            # kernel: out_c, k_size**2, in_c
            _, kernel = convlayer(x_windows[i], kernels=None)
            kernels.append(kernel)
            i = i + 1
        # kernels:  列表中有win_num个 out_c, k_size**2, in_c 的张量


        ### 下面想要计算所有卷积核间的自注意力
        # kernels:  out_c, win_num*k_size**2, in_c
        kernels = torch.cat(kernels, 1)

        # kernels_qkv:  3, out_c, num_heads, win_num*k_size**2, in_c/num_heads
        kernels_qkv = self.proj_qkv(kernels).reshape(self.dim, self.win_num*self.kernel_size**2, 3, self.num_heads, self.dim//self.num_heads).permute(2, 0, 3, 1, 4)

        # out_c, num_heads, win_num*k_size**2, in_c/num_heads
        kernels_q, kernels_k, kernels_v = kernels_qkv[0], kernels_qkv[1], kernels_qkv[2]
        kernels_q = kernels_q * self.scale

        # attn: out_c, num_heads, win_num*k_size**2, win_num*k_size**2
        attn = (kernels_q @ kernels_k.transpose(-2, -1))
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)

        # kernels:  out_c, win_num*k_size**2, in_c
        kernels = (attn @ kernels_v).transpose(1, 2).reshape(self.dim, self.win_num*self.kernel_size**2, self.dim)

        # TODO check: 此处kernels由win_num*k_size**2拆开，win_num的维度是在k_size前面还是后面
        # kernels:  win_num, out_c, in_c, k_size, k_size
        kernels = self.proj_out(kernels).reshape(self.dim, self.win_num, self.kernel_size, self.kernel_size, self.dim).permute(1, 0, 4, 2, 3)

        ### 下面计算SELayer输出并重新进行卷积
        # kernels:  win_num, out_c, in_c, k_size, k_size
        # x_windows_origin:  win_num, bs, c, wh, ww
        i = 0
        x_windows_out = []

        for selayer in self.selayers:
            # kernel:  out_c, in_c, k_size, k_size
            kernel = selayer(kernels[i])
            # x_window:  bs, c, wh, ww
            x_window = F.conv2d(x_windows[i], weight=kernel, bias=None, stride=self.stride, padding=self.padding).unsqueeze(0)

            # TODO check: 此处由1, bs*c, h, w变为1, bs, c, h, w的操作是否正确
            # x_window:  1, bs, c, wh, ww
            # x_window = x_window.view(B, self.dim, self.window_size, self.window_size).unsqueeze(0)
            x_windows_out.append(x_window)
            i = i + 1

        # x_windows:  win_num, bs, c, wh, ww
        x_windows = torch.cat(x_windows_out, 0)

        # x:  bs, h*w, c
        x = ka_window_reverse(x_windows, self.window_size, H, W)

        return x
    
input = torch.randn(1, 64*64, 24)

model = KernelAttention(24, 64, 4, ka_win_num=4, kernel_size=3, stride=1, padding=1, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.)

print(flop_count_table(FlopCountAnalysis(model, input)))