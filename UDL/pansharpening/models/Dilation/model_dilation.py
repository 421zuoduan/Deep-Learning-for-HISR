# -*- encoding: utf-8 -*-
"""
@File    : model_dilation.py
@Time    : 2022/3/19 9:45
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

def Win_Shuffle(x, win_size):
    """
    :param x: B H W C
    :param win_size:
    :return: y: B H W C
    """
    B, H, W, C = x.shape
    dilation = win_size*2
    resolution = H
    assert resolution % win_size == 0, 'resolution of input should be divisible'
    assert win_size % 2 == 0, 'win_size should be the multiple of two'
    "input size BxCxHxW"
    "shuffle"

    x = window_partition(x, dilation)  # BN x d x d x c
    xt = torch.zeros_like(x)
    x0 = x[:, 0::2, 0::2, :]  # B n/2 n/2 d2c
    x1 = x[:, 0::2, 1::2, :]  # B n/2 n/2 d2c
    x2 = x[:, 1::2, 0::2, :]  # B n/2 n/2 d2c
    x3 = x[:, 1::2, 1::2, :]  # B n/2 n/2 d2c

    xt[:, 0:win_size, 0:win_size, :] = x0  # B n/2 n/2 d2c
    xt[:, 0:win_size, win_size:dilation, :] = x1  # B n/2 n/2 d2c
    xt[:, win_size:dilation, 0:win_size, :] = x2  # B n/2 n/2 d2c
    xt[:, win_size:dilation, win_size:dilation, :] = x3  # B n/2 n/2 d2c
    xt = xt.reshape(-1, dilation, dilation, C)
    xt = window_reverse(xt, dilation, H, W)

    return xt

def Win_Reshuffle(x, win_size):
    """
        :param x: B H W C
        :param win_size:
        :return: y: B H W C
    """
    B, H, W, C = x.shape
    dilation = win_size*2
    assert H % win_size == 0, 'resolution of input should be divisible'
    assert win_size % 2 == 0, 'win_size should be the multiple of two'

    x = window_partition(x, dilation)  # BN x d x d x c
    xt = torch.zeros_like(x)
    xt[:, 0::2, 0::2, :] = x[:, 0:win_size, 0:win_size, :]  # B n/2 n/2 d2c
    xt[:, 0::2, 1::2, :] = x[:, 0:win_size, win_size:dilation, :]  # B n/2 n/2 d2c
    xt[:, 1::2, 0::2, :] = x[:, win_size:dilation, 0:win_size, :]  # B n/2 n/2 d2c
    xt[:, 1::2, 1::2, :] = x[:, win_size:dilation, win_size:dilation, :]  # B n/2 n/2 d2c
    xt = xt.reshape(-1, dilation, dilation, C)
    xt = window_reverse(xt, dilation, H, W)

    return xt


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

class Attention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
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
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Stage(nn.Module):
    def __init__(self, dim=32, input_resolution=(16, 16), num_heads=8, window_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=4, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio

        assert 0 <= self.window_size <= input_resolution[0], "input_resolution should be larger than window_size"
        assert 0 <= self.window_size <= input_resolution[1], "input_resolution should be larger than window_size"

        self.attn1 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn2 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.attn3 = Attention(
            dim, num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)
        self.mlp2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.norm5 = norm_layer(dim)
        self.norm6 = norm_layer(dim)
        self.mlp3 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn1(x_windows)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp1(self.norm2(x)))

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        shifted_x = Win_Shuffle(x, self.window_size)
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn2(x_windows)  # nW*B, window_size*window_size, C
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = Win_Reshuffle(shifted_x, self.window_size)
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp2(self.norm4(x)))

        shortcut = x
        x = self.norm5(x)
        x = x.view(B, H, W, C)
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        attn_windows = self.attn3(x_windows)  # nW*B, window_size*window_size, C
        x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
        x = x.view(B, H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp3(self.norm6(x)))

        return x

class Bottleneck(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=31,
                 embed_dim=96, num_heads=[],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        '''
        :param patch_size: for the embed conv
        :param in_chans: for the embed conv

        '''
        super(Bottleneck, self).__init__()
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.mlp_ratio = mlp_ratio
        self.img_size = img_size
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=3, stride=1, padding=1)
        patches_resolution = [img_size//patch_size, img_size//patch_size]

        # split image into non-overlapping patches
        downsample = PatchMerging
        self.downsample = downsample(dim=self.embed_dim, input_resolution=(self.img_size, self.img_size), norm_layer=norm_layer)
        self.upsample = nn.Sequential(
            nn.Conv2d(self.embed_dim*2, self.embed_dim * 8, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True))
        self.conv1 = nn.Conv2d(self.embed_dim*3, self.embed_dim*2, 1, 1, 0)
        self.stage1 = Stage(dim=self.embed_dim, input_resolution=(self.img_size, self.img_size), num_heads=num_heads[0], window_size=window_size)
        self.stage2 = Stage(dim=self.embed_dim*2, input_resolution=(self.img_size//2, self.img_size//2), num_heads=num_heads[1], window_size=window_size)
        self.stage3 = Stage(dim=self.embed_dim*2, input_resolution=(self.img_size, self.img_size), num_heads=num_heads[2], window_size=window_size)


    def forward(self, x):
        B, _, H, _ = x.shape
        x = self.conv(x)  # B x em x H x W
        x = rearrange(x, 'B C H W -> B (H W) C ')
        stage1_out = self.stage1(x)
        stage1_out_shape = rearrange(stage1_out, 'B (H W) C -> B C H W', H=H)
        stage1_out = self.downsample(stage1_out)
        H = H // 2
        stage2_out = self.stage2(stage1_out)
        stage2_out_shape = rearrange(stage2_out, 'B (H W) C -> B C H W', H=H)
        stage2_up = self.upsample(stage2_out_shape)
        cat = torch.cat([stage1_out_shape, stage2_up], dim=1)
        cat = self.conv1(cat)
        cat = rearrange(cat, 'B C H W -> B (H W) C ')
        stage3_out = self.stage3(cat)
        H = H*2
        stage3_out_shape = rearrange(stage3_out, 'B (H W) C -> B C H W', H=H)
        return stage3_out_shape

if __name__ == '__main__':
    input = torch.randn(1, 34, 64, 64).cuda()
    t = Bottleneck(img_size=64, patch_size=1, in_chans=34, embed_dim=48, num_heads=[8, 8, 8], window_size=4).cuda()
    output = t(input)
    print(output.shape)

