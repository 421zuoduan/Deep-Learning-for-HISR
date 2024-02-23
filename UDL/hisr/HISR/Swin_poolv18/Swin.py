# -*- encoding: utf-8 -*-
"""
@File    : Swin.py
@Time    : 2021/11/11 9:17
@Author  : Shangqi Deng
@Email   : dengsq5856@126.com
@Software: PyCharm
"""
import numpy as np
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange, repeat
from math import sqrt
import torch.nn.functional as F


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N).cuda() + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)
        attn_map = attn

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_map

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops
    

def pad_feature_map(input, window_size=8):
    _, L, C = input.size()
    H, W = int(math.sqrt(L)), int(math.sqrt(L))
    input = input.reshape(-1, H, W, C).permute(0, 3, 1, 2)
    
    # 计算要添加的上下左右 padding
    pad_h = (window_size - (H % window_size)) % window_size
    pad_w = (window_size - (W % window_size)) % window_size
    
    # 进行 padding
    padded_feature_map = torch.nn.functional.pad(input, (0, pad_w, 0, pad_h), 'constant', 0)

    output = padded_feature_map.permute(0, 2, 3, 1).reshape(-1, (H+pad_h)*(W+pad_w), C)
    
    return output


class WinKernel_Reweight(nn.Module):
    def __init__(self, dim, win_num=4):
        super().__init__()
        
        self.dim = dim
        self.win_num = win_num
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.downchannel = nn.Conv2d(win_num*dim, win_num, kernel_size=1, groups=win_num)
        self.linear1 = nn.Conv2d(win_num, win_num*4, kernel_size=1)
        self.gelu = nn.GELU()
        self.linear2 = nn.Conv2d(win_num*4, win_num, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, kernels, windows):
        """
        kernels:  win_num*c, 1, k ,k
        windows:  bs, win_num*c, wh, ww
        """

        B = windows.shape[0]

        # win_weight:  bs, win_num*c, 1, 1
        win_weight = self.pooling(windows)

        # win_weight:  bs, win_num, 1, 1
        win_weight = self.downchannel(win_weight)

        win_weight = self.linear1(win_weight)
        win_weight = win_weight.permute(0, 2, 3, 1).reshape(B, 1, -1)
        win_weight = self.gelu(win_weight)
        win_weight = win_weight.transpose(1, 2).reshape(B, -1, 1, 1)
        win_weight = self.linear2(win_weight)
        # weight:  bs, win_num, 1, 1, 1, 1
        weight = self.sigmoid(win_weight).unsqueeze(-1).unsqueeze(-1)

        # kernels:  bs, win_num, c, 1, k, k
        kernels = kernels.reshape(B, self.win_num, self.dim, 1, kernels.shape[-2], kernels.shape[-1])

        # kernels:  bs, win_num, c, 1, k, k
        kernels = weight * kernels

        # kernels:  bs*win_num*c, 1, k, k
        kernels = kernels.reshape(-1, 1, kernels.shape[-2], kernels.shape[-1])

        return kernels


class ConvLayer(nn.Module):

    def __init__(self, dim, k_size, stride, padding, ka_win_num, if_global=True):
        super().__init__()

        self.dim = dim
        self.k_size = k_size
        self.stride = stride
        self.padding = padding
        self.nW = ka_win_num
        self.if_global = if_global

    def forward(self, x, kernels, B):

        if self.if_global:
            G = B * (self.nW+1) * self.dim
        else:
            G = B * self.nW * self.dim

        x = F.conv2d(x, kernels, stride=self.stride, padding=self.padding, groups=G)

        return x


class WindowInterAttention(nn.Module):

    def __init__(self, input_resolution, dim, ka_win_num, k_size, k_stride, k_padding, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.nW = ka_win_num
        # self.nW = input_resolution[0] * input_resolution[1] // window_size // window_size
        self.window_size = int(input_resolution[0] // sqrt(ka_win_num))
        self.num_heads = num_heads
        self.k_size = k_size
        self.k_stride = k_stride
        self.k_padding = k_padding

        self.pooling = nn.AdaptiveAvgPool2d((k_size, k_size))

        self.pooling_attn_spacial = nn.AdaptiveAvgPool2d((k_size**2, k_size**2))
        self.pooling_attn_nW = nn.AdaptiveAvgPool1d(ka_win_num)
        # self.scale = qk_scale or dim ** -0.5
        # self.proj_qkv = nn.Linear(dim*k_size*k_size, dim*k_size*k_size * 3, bias=qkv_bias)
        self.proj_v = nn.Linear(dim*k_size*k_size, dim*k_size*k_size, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_out = nn.Linear(dim*k_size*k_size, dim*k_size*k_size)
        self.proj_out_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)

        self.gk_generation = nn.Conv2d(in_channels=self.nW, out_channels=1, kernel_size=1, stride=1, padding=0)

        # self.se = nn.Sequential(
        #     nn.Conv2d(in_channels=dim, out_channels=dim//16, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Conv2d(in_channels=dim//16, out_channels=dim, kernel_size=1, stride=1, padding=0),
        #     nn.Sigmoid()
        # )

        self.convlayer = ConvLayer(dim, k_size, k_stride, k_padding, self.nW, if_global=True)
        self.wink_reweight = WinKernel_Reweight(dim, win_num=self.nW)
        self.fusion = nn.Conv2d((self.nW+1)*self.dim, self.dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x, attn):
        '''
        x:  b, h*w, c
        attn:  b*nW, num_heads, wh*ww, wh*ww
        '''
        B, L, C = x.shape
        H, W = self.input_resolution
        x = x.view(B, H, W, C)

        # x_windows: B*nW, window_size, window_size, C
        x_windows = window_partition(x, self.window_size)
        # x_windows: B*nW, C, window_size, window_size
        x_windows = x_windows.permute(0, 3, 1, 2).contiguous()


        ### 池化为分组卷积参数
        # kernels: B*nW, C, k_size, k_size
        kernels = self.pooling(x_windows)
        # kernels: B, nW, C*k_size*k_size
        kernels = kernels.reshape(B, self.nW, C*self.k_size*self.k_size)


        ### kernels进行自注意力计算
        # kernels_v:  B*nW*num_heads, k_size**2, C/num_heads
        kernels_v = self.proj_v(kernels).reshape(B, self.nW, self.num_heads, self.dim//self.num_heads, self.k_size**2).transpose(4, 3)
        kernels_v = kernels_v.reshape(-1, self.k_size**2, self.dim//self.num_heads)

        # attn_qk:  B*nW*num_heads, k_size**2, k_size**2
        attn_qk = self.pooling_attn_spacial(attn).reshape(-1, self.k_size**2, self.k_size**2)

        # attn_qk:  B*nW*num_heads, k_size**2, k_size**2
        attn_qk = attn_qk.reshape(B, -1, self.num_heads, self.k_size**2, self.k_size**2).permute(0, 2, 3, 4, 1).reshape(B*self.num_heads*self.k_size**4, -1)
        attn_qk = self.pooling_attn_nW(attn_qk).reshape(B, self.num_heads, self.k_size**2, self.k_size**2, self.nW).permute(0, 4, 1, 2, 3).reshape(-1, self.k_size**2, self.k_size**2)

        # kernels:  B*num_heads, nW, k_size**2, C/num_heads
        kernels = attn_qk @ kernels_v

        # kernels:  B, nW, C*k_size**2
        kernels = kernels.reshape(B, self.num_heads, self.nW, self.k_size**2, self.dim//self.num_heads).permute(0, 2, 1, 4, 3)
        kernels = kernels.reshape(B, self.nW, self.dim*self.k_size**2)

        # kernels:  B, nW, C*k_size**2
        kernels = self.proj_out(kernels)


        # ### 对kernels进行SE
        # kernels:  B*nW, C, k_size, k_size
        # kernels = kernels.reshape(B, self.nW, self.kernel_size, self.kernel_size, self.dim).permute(0, 1, 4, 2, 3).reshape(B*self.nW, self.dim, self.kernel_size, self.kernel_size)

        # # kernels:  bs*win_num, c, k_size, k_size
        # kernels = self.se(kernels)


        ### 生成全局卷积核
        # x_windows:  B, nW*C, win_size, win_size
        x_windows = x_windows.reshape(B, self.nW*C, self.window_size, self.window_size)

        # kernels:  B*nW*C, 1, k_size, k_size
        kernels = kernels.reshape(B*self.nW*C, 1, self.k_size, self.k_size).unsqueeze(1)

        # kernels:  B*nW*C, 1, k_size, k_size
        kernels = self.wink_reweight(kernels, x_windows)

        # kernels:  B*C, nW, k_size, k_size
        kernels = kernels.reshape(B, self.nW, self.dim, self.k_size, self.k_size).transpose(1, 2).reshape(B*self.dim, self.nW, self.k_size, self.k_size)

        # global_kernel:  B*C, 1, k_size, k_size
        global_kernel = self.gk_generation(kernels)

        # global_kernel:  B, 1, C, 1, k_size, k_size
        global_kernel = global_kernel.reshape(B, self.dim, 1, self.k_size, self.k_size).unsqueeze(1)
        
        # kernels:  B, nW, C, 1, k_size, k_size
        kernels = kernels.reshape(B, self.dim, self.nW, 1, self.k_size, self.k_size).transpose(1, 2)

        # kernels:  B, nW+1, C, 1, k_size, k_size
        kernels = torch.cat([kernels, global_kernel], dim=1)

        # kernels:  B*(nW+1)*C, 1, k_size, k_size
        kernels = kernels.reshape(-1, 1, self.k_size, self.k_size)


        ### 卷积核与输入特征计算卷积
        # x:  B, C, H, W
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        # x:  B, (nW+1)*C, H, W
        x = x.repeat(1, self.nW+1, 1, 1)

        # x:  1, B*(nW+1)*C, H, W
        x = x.reshape(1, B*(self.nW+1)*C, H, W)

        # x:  1, B*(nW+1)*C, H, W
        x = self.convlayer(x, kernels, B)
        
        # x:  B, (nW+1)*C, H, W
        x = x.reshape(B, (self.nW+1)*C, H, W)

        # x:  B, C, H, W
        x = self.fusion(x)

        # x:  B, H*W, C
        x = x.permute(0, 2, 3, 1).reshape(B, L, C)

        return x


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.__class__.__name__ = "SwinTEB"
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size).cuda()  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.attn_mask = attn_mask
        # self.register_buffer("attn_mask", attn_mask)

        if self.shift_size == 0:
            self.window_inter_attn = WindowInterAttention(input_resolution, dim=dim, ka_win_num=16, k_size=3, k_stride=1, k_padding=1, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)


    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows, attn_map = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        if self.shift_size == 0:
            x = self.window_inter_attn(x, attn_map)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops

class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            # self.downsample = None
        else:
            self.downsample = None

        # self.window_inter_attn = WindowInterAttention(input_resolution, dim=dim, ka_win_num=16, window_size=window_size, k_size=3, k_stride=1, k_padding=1, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop)


    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)

        # x = self.window_inter_attn(x)

        if self.downsample is not None:
            x = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

class T(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=31,
                 embed_dim=96, depths=[2, 4], num_heads=[3, 3],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        '''
        :param patch_size: for the embed conv
        :param in_chans: for the embed conv

        '''
        super(T, self).__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        patches_resolution = [img_size//patch_size, img_size//patch_size]

        # split image into non-overlapping patches
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

    def forward(self, x):
        B, _, H, _ = x.shape
        x = self.conv(x)
        x = rearrange(x, 'B C H W -> B (H W) C ')
        for layer in self.layers:
            x = layer(x)
        H = H // (2 ** (self.num_layers - 1))
        x = rearrange(x, 'B (H W) C -> B C H W', H=H)
        return x

if __name__ == '__main__':
    from UDL.Basis.auxiliary.torchstat.statistics import stat

    input = torch.randn(32, 34, 64, 64).cuda()
    t = T(img_size=64, patch_size=1, in_chans=34, embed_dim=32, depths=[2, 4], num_heads=[8, 8], window_size=4).cuda()
    output = t(input)
    print(output.shape)
    out = stat(t, input_size=[[1, 34, 64, 64]])





