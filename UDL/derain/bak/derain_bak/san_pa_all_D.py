import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import numpy as np
# from einops import rearrange
from typing import Optional, List
from torch import Tensor


# from image_preprocessing_t import PatchifyAugment

######################################################
# 构造PatchesEmbedding，先得到patch再考虑如何处理patch
######################################################

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor, compatiable=True):
        super().__init__()
        # self.downscaling_factor = downscaling_factor
        self.downscaling_factor = downscaling_factor = int(downscaling_factor[0])  # list(downscaling_factor.numpy())[0]
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)
        self.C = out_channels
        self.comp = compatiable

    def forward(self, x, bs):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h * new_w).transpose(2, 1)
        # unsqueeze 与HirePatch的维度对齐，传统的需要压缩patch维度，所以这里p=1
        x = self.linear(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return x, (new_h, new_w, self.C)


class PatchConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_stride=16, stride=16, in_chans=3, embed_dim=768, compatiable=False):
        super().__init__()
        img_size = _pair(img_size)
        patch_stride = _pair(patch_stride)
        stride = _pair(stride)

        self.img_size = img_size
        self.patch_stride = patch_stride
        assert img_size[0] % patch_stride[0] == 0 and img_size[1] % patch_stride[1] == 0, \
            f"img_size {img_size} should be divided by patch_stride {patch_stride}."
        self.H, self.W = img_size[0] // patch_stride[0], img_size[1] // patch_stride[1]
        # self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_stride, stride=patch_stride)
        self.norm = nn.LayerNorm(embed_dim)
        # self.pa = PatchifyAugment(False, self.H)
        self.comp = compatiable

    def forward(self, x, bs):
        B, C, H, W = x.shape
        p = B // bs

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_stride[0], W // self.patch_stride[1]
        if len(x.shape) == 3:
            x = x.reshape(bs, p, H * W, -1)

        return x, (H, W, x.shape[-1])


class HPatchConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_stride=16, stride=16, in_chans=3, embed_dim=768, num_patch=None,
                 compatiable=False):
        super().__init__()
        img_size = _pair(img_size)
        patch_stride = _pair(patch_stride)
        stride = _pair(stride)
        self.embed_dim = embed_dim
        self.img_size = img_size
        self.patch_stride = patch_stride
        self.num_patch = (1 if num_patch is None else (img_size - patch_stride) // patch_stride + 1) ** 2
        if self.num_patch == 1:
            Warning("HPatchConvEmbed will get non-local info from local windows")
        else:
            Warning("HPatchConvEmbed will get more patch, equal to HirePatch, but share the weight among patchs")
        assert img_size[0] % patch_stride[0] == 0 and img_size[1] % patch_stride[1] == 0, \
            f"img_size {img_size} should be divided by patch_stride {patch_stride}."
        self.H, self.W = img_size[0] // patch_stride[0], img_size[1] // patch_stride[1]
        # self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim * num_patch, kernel_size=patch_stride, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
        # self.pa = PatchifyAugment(False, self.H)
        self.comp = compatiable

    def forward(self, x, bs):
        num_patch = self.num_patch
        patch_stride = self.patch_stride
        B, C, H, W = x.shape
        p = B // bs

        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // patch_stride[0], W // patch_stride[1]
        if len(x.shape) == 3:
            x = x.reshape(bs, p, H * W, num_patch, -1).permute(0, 1, 3, 2, 4).\
                reshape(bs, p * num_patch, H * W, -1).contiguous()

        return x, (H, W, x.shape[-1])


class IdentityModule(nn.Module):

    def __init__(self, patch_func=None, residual_func=None):
        super(IdentityModule, self).__init__()

        if patch_func is not None:
            self.patch_func = patch_func
        else:
            self.patch_func = lambda x: x

        if residual_func is not None:
            self.residual_func = residual_func
        else:
            self.residual_func = lambda x: x

    def forward(self, x, residual):

        out_ps = self.patch_func(x)
        residual = self.residual_func(residual)

        out_ps = out_ps + residual

        return out_ps


class AttentionFromPvt(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            # 步进8卷积
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q, k, v = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# FeedForward

# class FeedForward(nn.Module):
#     def __init__(self, dim, hidden_dim):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.GELU(),
#             nn.Linear(hidden_dim, dim),
#         )
#
#     def forward(self, x):
#         return self.net(x)

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        b, p, N, C = x.shape
        # L = int(math.sqrt(N))
        x = x.reshape(-1, H, W, C).permute(0, 3, 1, 2).contiguous()
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(b, p, -1, C).contiguous()
        return x


# class DWConv(nn.Module):
#     def __init__(self, dim):
#         super(DWConv, self).__init__()
#         self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
#
#     def forward(self, x, H, W):
#         B, p, N, C = x.shape
#         x = x.transpose(1, 2).reshape(B, C, H, W).contiguous()
#         x = self.dwconv(x)
#         x = x.permute(0, 2, 3, 1)#.flatten(2).transpose(1, 2)
#
#         return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.hidden_features = hidden_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HyFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 compatiable=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.hidden_features = hidden_features
        self.conv = DWConv(hidden_features)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.conv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PvtBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # the effect of sr_ratio
        self.attn = AttentionFromPvt(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.attn(self.norm1(x), H, W)  # self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.mlp(self.norm2(x), H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x


############################################################################################
# Stem network, 从事分割、超分、去x任务，可基于U-net/Hrnet/EffDet(FPN)
############################################################################################


class Block(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, image_size, windows_size, stride, pad, in_channels, out_channels, mid_channels=None,
                 split_patch=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if split_patch:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, image_size, windos_size, stride, pad, in_channels, out_channels, split_patch=True):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            Block(image_size, windos_size, stride, pad, in_channels, out_channels, split_patch=split_patch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, image_size, windos_size, stride, pad, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 逐层回放
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = Block(image_size, windos_size, stride, pad, in_channels, out_channels, in_channels // 2,
                              split_patch=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = Block(image_size, windos_size, stride, pad, in_channels, out_channels, split_patch=False)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


'''
改造PVT以对应DETR中的transformer
这是一个U-shaped结构的Transformer
DETR: backbone + top-down Transformer + ConvNetSegmentation 
s.t. backbone + Pvt + ConvNetSegmentation
'''


class Pvt(nn.Module):
    def __init__(self, patch_size, patch_stride, in_channels, out_channels, hidden_dim,
                 num_heads, mlp_ratios, depths, sr_ratio,
                 norm_pre=False, bilinear=True):
        super(Pvt, self).__init__()

        self.norm_pre = norm_pre
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.inc = PrimalConvBlock(in_channels, hidden_dim[0])

        self.patch_embed1 = PatchConvEmbed(img_size=patch_size, patch_sride=patch_stride[0], in_chans=hidden_dim[0],
                                           embed_dim=hidden_dim[1])
        self.patch_embed2 = PatchConvEmbed(img_size=patch_size // 2, patch_sride=patch_stride[1],
                                           in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2])
        self.patch_embed3 = PatchConvEmbed(img_size=patch_size // 4, patch_sride=patch_stride[2],
                                           in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3])
        self.patch_embed4 = PatchConvEmbed(img_size=patch_size // 8, patch_sride=patch_stride[3],
                                           in_chans=hidden_dim[3],
                                           embed_dim=hidden_dim[4])
        # self.patch_embed5 = PatchConvEmbed(img_size=3, patch_size=2, in_chans=256,
        #                                     embed_dim=256)

        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, hidden_dim[1]))
        # self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, hidden_dim[2]))
        # self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, hidden_dim[3]))
        # self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, hidden_dim[4]))
        # self.pos_drop4 = nn.Dropout(p=drop_rate)

        self.down1 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[0])
            for i in range(depths[0])])
        self.down2 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[1])
            for i in range(depths[1])])
        self.down3 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=True, qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[2])
            for i in range(depths[2])])
        factor = 2 if bilinear else 1
        # TODO: 扩维更合适
        self.down4 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=True, qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[3])
            for i in range(depths[3])])

        # self.norm = nn.LayerNorm(64, eps=1e-6)
        self.norm = nn.LayerNorm([hidden_dim[1], patch_size, patch_size], eps=1e-6)
        # self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)

        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        self.up2 = PrimalUpBlock(512 // factor + hidden_dim[2], 256 // factor, bilinear)  # 512
        self.up3 = PrimalUpBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        self.up4 = PrimalUpBlock(128 // factor + hidden_dim[0], 64, bilinear)
        self.up5 = PrimalUpBlock(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # init weights
        nn.init.trunc_normal_(self.pos_embed1, std=.02)
        nn.init.trunc_normal_(self.pos_embed2, std=.02)
        nn.init.trunc_normal_(self.pos_embed3, std=.02)
        nn.init.trunc_normal_(self.pos_embed4, std=.02)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_pre(self, x):
        ...

    def forward_post(self, x):
        # x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        bs = x.shape[0]
        # x = self.pa(x)

        x0 = self.inc(x)  # 32
        # stage 1 -64
        x, (H, W, C) = self.patch_embed1(x0, bs)  # x
        x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, H, W)
        x1 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2 -128
        x, (H, W, C) = self.patch_embed2(x1, bs)
        x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, H, W)
        x2 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3 -320
        x, (H, W, C) = self.patch_embed3(x2, bs)
        x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, H, W)
        x3 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4 -512
        x, (H, W, C) = self.patch_embed4(x3, bs)
        x = x + self.pos_embed4
        for blk in self.down4:
            x = blk(x, H, W)
        x4 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # x = self.up5(x, x0)
        x = self.up1(x4, x3, bs)
        x = self.up2(x, x2, bs)
        x = self.up3(x, x1, bs)
        x = self.up4(x, x0, bs)
        # B, H, W, _ = x.shape
        # x = x.flatten(2).transpose(2, 1)
        x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        logits = self.outc(x)
        # logits = self.norm(logits)
        return logits

    def forward(self, x):

        if self.norm_pre:
            return self.forward_pre(x)
        return self.forward_post(x)


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, image_size, windows_size, stride, pad, in_channels, out_channels, mid_channels=None,
                 split_patch=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if split_patch:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class PrimalUpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 逐层回放
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # self.conv = nn.Sequential(#nn.ModuleList([
            #     nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #     ResBlock(out_channels, kernel_size=3),#,
            #     # ResBlock(out_channels, kernel_size=3)
            # )
        #])
            self.conv = PrimalConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = PrimalConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, bs):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class PrimalConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            # nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ResBlock(nn.Module):
    def __init__(
        self, n_feats, kernel_size=3,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size,
                    padding=(kernel_size//2), bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, image_size, windos_size, stride, pad, in_channels, out_channels, split_patch=True):
        super().__init__()

        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(image_size, windos_size, stride, pad, in_channels, out_channels, split_patch=split_patch)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, image_size, windos_size, stride, pad, in_channels, out_channels, bilinear=True):
        super().__init__()

        # 逐层回放
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = ConvBlock(image_size, windos_size, stride, pad, in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = ConvBlock(image_size, windos_size, stride, pad, in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)


def partial_load_checkpoint(state_dict, amp, dismatch_list):
    pretrained_dict = {}
    # dismatch_list = ['dwconv']
    if amp is not None:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            k = 'amp.'.join(k)
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})
    else:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})

    return pretrained_dict


class HAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, p, N, C = x.shape
        out = torch.zeros_like(x)
        for i, (blk, sub_x) in enumerate(zip(self.H_module, x.chunk(p, dim=1))):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(sub_x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # if self.sr_ratio > 1:
            #     x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            #     # 步进8卷积
            #     x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            #     x_ = self.norm(x_)
            #     kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # else:
            kv = kv(sub_x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            sub_x = (attn @ v).transpose(1, 2).reshape(B, N, C)

            sub_x = self.proj(sub_x)
            sub_x = self.proj_drop(sub_x)

            out[:, i, ...] = sub_x

        return out


class PAttention_V2(nn.Module):
    '''
    # Case 2: 1，N, D x p,D,N = p, N, N
    #         p, N, N x p, N, D = N, D
    # 意义上就不好
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x_s, x_h, H, W):
        B, p, N, C = x_h.shape
        _, _, N_q, D = x_s.shape
        out = torch.zeros(B, p, N_q, C).to(x_h.device)
        # x_h = x_h.reshape(B, p, -1)
        # D_h = N * C

        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            # B, heads, N, C
            q = q(x_s).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

            # 2, B, heads, p, N_q, C'= D
            kv = kv(x_h).reshape(B, p, N, 2, self.num_heads, D // self.num_heads).permute(3, 0, 4, 1, 2, 5)

            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, p, N, N
            attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            attn = attn.permute(0, 1, 3, 2, 4).reshape(B, self.num_heads, N_q, -1)
            # pN * pN
            x_h = (attn @ v).transpose(1, 2).reshape(B, N_q, D)

            x_h = self.proj(x_h)
            x_h = self.proj_drop(x_h)

            # out[:, i, ...] = x_h
            out = x_h

        return out


class PAttention(nn.Module):
    '''
    QKV的W仅操作了通道方向，这适用于CP->D的情况，否则效果不好
        x_s     x_h
    QxK 1, NC x NC, p, = 1, p
    KxV 1,p x p, N, C = 1, N, C = x_s.shape
    '''

    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, num_patch=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(num_patch)])
        # for i in range(downscaling_factor):
        #     self.q = nn.Linear(dim, dim, bias=qkv_bias)
        #     self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        #     H_module.append(self.q)
        #     H_

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x_s, x_h, H, W):
        B, p, N, C = x_h.shape
        _, p_s, N_q, D = x_s.shape
        # x_s = x_s.reshape(B, 1, -1)
        # N_xs = N_q * D

        out = torch.zeros(B, 1, N_q, C).to(x_h.device)

        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(x_s).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q.reshape(B, self.num_heads, 1, -1)  # B, heads, N_q*D

            kv = kv(x_h).reshape(B, p, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2,
                                                                                          5)  # 2, B, heads, p, N_q, C'= D
            kv = kv.reshape(2, B, self.num_heads, p, -1)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale  # B, heads, 1, p
            # attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            x_s = (attn @ v).transpose(1, 2).reshape(B, N_q, D)

            x_s = self.proj(x_s)
            x_s = self.proj_drop(x_s)

            out[:, i, ...] = x_s

        return out


class HBlock(nn.Module):

    def __init__(self, downscaling_factor, norm_dim, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        # the effect of sr_ratio
        self.attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, x, H, W):
        # x: (b,P,N,c)
        x = x + self.attn(self.norm1(x), H, W)  # self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.ffn(self.norm2(x), H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class HPBlock(nn.Module):

    def __init__(self, down_scale, norm_dim, dim, num_heads, num_patch,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim
        # the effect of sr_ratio
        self.attn = PAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        self.norm3 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, x_s, x_h, H, W):
        # x: (b,P,N,c)
        # x_s = self.norm1(x_s)
        # p_x = x_s + self.attn(x_s, self.norm_h1(x_h), H, W)
        x = x_s + self.self_attn(self.norm1(x_s), H, W)
        x = x + self.ffn(self.norm2(x), H, W)

        return x

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


class HirePatch(nn.Module):
    def __init__(self, downs_scale, patch_stride, in_chans, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.downscaling_factor = list(downs_scale)
        # self.patch_merge = nn.Unfold(kernel_size=int(self.downscaling_factor[0]), stride=int(self.downscaling_factor[0]), padding=0)
        self.linear = nn.Linear(in_chans, embed_dim)

    def forward(self, x, bs, axes=[2, 3]):  # h, w
        b, c, h, w = x.shape
        # b, c, *hw = x.shape
        p = b // bs
        # shape = [hw[axis] // self.downscaling_factor[axis] for axis in axes]
        new_h, new_w = shape = [h // self.downscaling_factor[0], w // self.downscaling_factor[1]]  #
        # x = self.patch_merge(x).view(self.bs, c, p * self.downscaling_factor*2, new_h, new_w).permute(0, 2, 3, 4, 1)
        # x = x.reshape(self.bs, -1, new_h * new_w, c)
        # x = x.reshape(self.bs, p, -1, c)
        if 1 in self.downscaling_factor:
            for axis, half, factor in zip(axes, shape, self.downscaling_factor):
                if factor == 1: continue
                if axis == 2:
                    x = x.unfold(axis, half, half // 2).transpose(-1, -2)
                elif axis == 3:
                    x = x.unfold(axis, half, half // 2).transpose(3, 2)
                x = x.reshape(self.bs * factor * p, c, x.shape[-2], x.shape[-1])
            x = x.view(self.bs, c, -1, *shape).permute(0, 2, 3, 4, 1)
        else:
            # x = self.patch_merge(x).view(self.bs, c, -1, new_h, new_w).permute(0, 2, 3, 4, 1)
            x = x.unfold(axes[0], new_h, new_h // 2).unfold(axes[1], new_w, new_w // 2). \
                reshape(bs, c, -1, new_h, new_w).permute(0, 2, 3, 4, 1).contiguous()  # bs, p, h, w, c

        x = x.reshape(bs, x.shape[1], -1, c)
        x = self.linear(x)
        return x, (*shape, self.embed_dim)  # (b,p,n_h,n_w,c)


'''
QK得到Attention，原版QK是xW，而x是局部窗口的非局部信息压缩，存在信息损失
Q和K分离，先让Q意义不变，K：p*M, N_q, C; Q: 1, N_q, C, 类似SSD
'''


class MHSPatch(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_patch=None, stride_ratio=2, K=10):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = list(patch_size)
        self.stride_ratio = stride_ratio
        img_size = np.array(img_size)
        patch_size = np.array(patch_size)
        self.N = np.prod(patch_size)
        # new_h, new_w = patch_size
        # 128 - 32 / 16 + 1 = 7, 49
        # 64 - 32 / 16 + 1 = 3, 9
        # print(img_size, patch_size)
        # print(np.prod(np.array([((img_size - patch_size) // (patch_size // stride_ratio)) + 1]), axis=-1))
        if num_patch is None:
            self.num_patch = 1
        else:
            num_patch = np.prod(np.array([((img_size - patch_size) // (patch_size // stride_ratio)) + 1]), axis=-1)
            self.num_patch = num_patch = np.sum(num_patch)

        if isinstance(num_patch, int) and num_patch == 1:
            Warning("HPatchConvEmbed will get non-local info from local windows")
        else:
            Warning("HPatchConvEmbed will get more patch, equal to HirePatch, but share the weight among patchs")

        assert len(patch_size) == 2, print("per down_scale length should be 2, directions x,y")

        # self.patch_merge = nn.Unfold(kernel_size=int(self.downscaling_factor[0]), stride=int(self.downscaling_factor[0]), padding=0)
        # assert (embed_dim * K) % num_patch == 0, print("MHSPatch should be divided by num_patch")
        # 聚合多尺度patch到K个，每个embed_dim个通道
        self.linear = nn.Linear(in_chans * num_patch, embed_dim * K)
        self.num_patch = K


    def forward(self, tensor_list: List[Tensor], bs, axes=[2, 3], compatiable=True):  # h, w
        num_patch = self.num_patch
        stride_ratio = self.stride_ratio
        patch_size = self.patch_size  # 将不同层次的block按patch去生成，用于得到patch-level的比较
        new_h, new_w = patch_size
        N = self.N
        patch_list = []

        for x in tensor_list:

            b, c, h, w = x.shape  # b = B*p
            p = b // bs # p=1

            # h, w
            down_scale_y, down_scale_x = down_scale = [h // new_h, w // new_w]  #
            # 一种灵活的切patch方法, output:
            if 1 in down_scale:
                for axis, ps, factor in zip(axes, patch_size, down_scale):
                    if factor == 1: continue
                    if axis == 2:
                        x = x.unfold(axis, ps, ps // 2).transpose(-1, -2)
                    elif axis == 3:
                        x = x.unfold(axis, ps, ps // 2).transpose(3, 2)
                    x = x.reshape(bs * factor * p, c, x.shape[-2], x.shape[-1])
                x = x.view(bs, c, -1, *patch_size).permute(0, 2, 3, 4, 1)
            else:
                x = x.unfold(axes[0], new_h, new_h // 2).unfold(axes[1], new_w, new_w // 2). \
                    reshape(b, -1, N).transpose(2, 1)
            patch_list.append(x)
        #b,N,CP
        tensor_list = torch.cat(patch_list, dim=-1)#M*C*P

        # 得到多尺度的K
        tensor_list = self.linear(tensor_list)
        if compatiable:
            tensor_list = tensor_list.reshape(bs, N, num_patch, -1).permute(0, 2, 1, 3)
        return tensor_list, (*patch_size, self.embed_dim)  # (b,p,n_h,n_w,c)


class HireSharePatch(nn.Module):
    def __init__(self, img_size, down_scale, in_chans, embed_dim, num_patch=None, stride_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.down_scale = list(down_scale)
        self.num_patch = 1 if num_patch is None else np.prod(
            [((img_size - img_size // sr) // (img_size // (sr * stride_ratio))) + 1 for sr in down_scale])
        if self.num_patch == 1:
            Warning("HPatchConvEmbed will get non-local info from local windows")
        else:
            Warning("HPatchConvEmbed will get more patch, equal to HirePatch, but share the weight among patchs")

        assert len(down_scale) == 2, print("per down_scale length should be 2, directions x,y")
        # self.patch_merge = nn.Unfold(kernel_size=int(self.downscaling_factor[0]), stride=int(self.downscaling_factor[0]), padding=0)
        self.linear = nn.Linear(in_chans * self.num_patch, embed_dim * self.num_patch)

    def forward(self, x, bs, axes=[2, 3]):  # h, w
        b, c, h, w = x.shape
        # b, c, *hw = x.shape
        p = b // bs
        down_scale = self.down_scale
        # shape = [hw[axis] // self.downscaling_factor[axis] for axis in axes]
        new_h, new_w = shape = [h // down_scale[0], w // down_scale[1]]  #
        N = new_h * new_w
        # x = self.patch_merge(x).view(self.bs, c, p * self.downscaling_factor*2, new_h, new_w).permute(0, 2, 3, 4, 1)
        # x = x.reshape(self.bs, -1, new_h * new_w, c)
        # x = x.reshape(self.bs, p, -1, c)

        # 一种灵活的切patch方法
        if 1 in down_scale:
            for axis, half, factor in zip(axes, shape, down_scale):
                if factor == 1: continue
                if axis == 2:
                    x = x.unfold(axis, half, half // 2).transpose(-1, -2)
                elif axis == 3:
                    x = x.unfold(axis, half, half // 2).transpose(3, 2)
                x = x.reshape(bs * factor * p, c, x.shape[-2], x.shape[-1])
            x = x.view(bs, c, -1, *shape).permute(0, 2, 3, 4, 1)
        else:
            # x = self.patch_merge(x).view(self.bs, c, -1, new_h, new_w).permute(0, 2, 3, 4, 1)
            x = x.unfold(axes[0], new_h, new_h // 2).unfold(axes[1], new_w, new_w // 2). \
                reshape(b, -1, new_h * new_w).transpose(2, 1).contiguous()
        # 得到更大尺度的K
        x = self.linear(x)
        x = x.reshape(bs, N, self.num_patch, -1).permute(0, 2, 1, 3).reshape(bs, self.num_patch, N, -1).contiguous()
        return x, (*shape, self.embed_dim)  # (b,p,n_h,n_w,c)


class AggregateBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 逐层回放
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.conv = PrimalConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = PrimalConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, bs):
        # x1 = self.up(x1)
        B, C, H, W = x2.shape
        x2 = x2.reshape(bs, -1, C, H, W)
        x1 = x1.reshape(bs, -1, x1.shape[1], H, W)
        x = torch.cat([x2, x1], dim=2).reshape(B, -1, H, W)
        return self.conv(x)


class HPT(nn.Module):
    def __init__(self, args, image_size, down_scale, patch_stride, in_channels, out_channels, hidden_dim,
                 num_heads, mlp_ratios, depths, sr_ratio,
                 norm_pre=False, bilinear=True, compatiable=True):
        super(HPT, self).__init__()

        self.args = args
        # self.patch_num = patch_num = 2 * patch_size
        self.patch_size = image_size
        self.norm_pre = norm_pre
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        # self.inc = PrimalConvBlock(in_channels, hidden_dim[0])
        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim[0], kernel_size=3, padding=1),
                ResBlock(hidden_dim[0], kernel_size=5),
                ResBlock(hidden_dim[0], kernel_size=5)
            ) for _ in range(1)
        ])

        N = image_size * image_size
        num_patches_base1 = int(np.prod(down_scale[0]) ** 1)
        num_patches_base2 = num_patches_base1 ** 2
        num_patches_base3 = num_patches_base1 ** 3
        num_patches_base4 = num_patches_base1 ** 4

        num_patches1 = N // num_patches_base1
        num_patches2 = N // num_patches_base2
        num_patches3 = N // num_patches_base3
        num_patches4 = N // num_patches_base4

        base = 2
        patch_size1 = image_size // base
        patch_size2 = patch_size1 // base
        patch_size3 = patch_size2 // base
        patch_size4 = patch_size3 // base

        self.patch_embed1 = PatchConvEmbed(img_size=image_size, patch_stride=down_scale[0], in_chans=hidden_dim[0],
                                           embed_dim=hidden_dim[1])
        self.patch_embed2 = PatchConvEmbed(img_size=patch_size1, patch_stride=down_scale[1],
                                           in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2])
        self.patch_embed3 = PatchConvEmbed(img_size=patch_size2, patch_stride=down_scale[2],
                                           in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3])
        self.patch_embed4 = PatchConvEmbed(img_size=patch_size3, patch_stride=down_scale[3],
                                           in_chans=hidden_dim[3],
                                           embed_dim=hidden_dim[4])
        self.patch_hire1 = HireSharePatch(img_size=image_size, down_scale=down_scale[0],
                                          in_chans=hidden_dim[0], embed_dim=hidden_dim[1], num_patch=True)
        self.patch_hire2 = HireSharePatch(img_size=patch_size1, down_scale=down_scale[1],
                                          in_chans=hidden_dim[1], embed_dim=hidden_dim[2], num_patch=True)
        self.patch_hire3 = HireSharePatch(img_size=patch_size2, down_scale=down_scale[2],
                                          in_chans=hidden_dim[2], embed_dim=hidden_dim[3], num_patch=True)
        self.patch_hire4 = HireSharePatch(img_size=patch_size3, down_scale=down_scale[3],
                                          in_chans=hidden_dim[3], embed_dim=hidden_dim[4], num_patch=True)

        self.down1 = nn.ModuleList(
            [HPBlock(down_scale=1, norm_dim=[num_patches1, patch_size1, hidden_dim[1]], num_patch=num_patches1,
                     dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     sr_ratio=sr_ratio[0], compatiable=compatiable)
             for i in range(depths[0])])
        self.down2 = nn.ModuleList(
            [HPBlock(down_scale=1, norm_dim=[num_patches2, patch_size2, hidden_dim[2]], num_patch=num_patches2,
                     dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     sr_ratio=sr_ratio[1], compatiable=compatiable)
             for i in range(depths[1])])
        self.down3 = nn.ModuleList([HPBlock(down_scale=1,
                                            norm_dim=[num_patches3, patch_size3, hidden_dim[3]],
                                            num_patch=num_patches3,
                                            dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                                            qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            sr_ratio=sr_ratio[2], compatiable=compatiable)
                                    for i in range(depths[2])])
        factor = 2 if bilinear else 1
        self.down4 = nn.ModuleList([HPBlock(down_scale=1,
                                            num_patch=num_patches4,
                                            norm_dim=[num_patches4, patch_size4, hidden_dim[4]],
                                            # [k ** 4 for k in patch_num],
                                            dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                                            qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
                                            sr_ratio=sr_ratio[3], compatiable=compatiable)
                                    for i in range(depths[3])])

        # self.norm = nn.LayerNorm(64, eps=1e-6)
        self.norm = nn.LayerNorm([hidden_dim[1], image_size, image_size], eps=1e-6)
        # self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)

        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], 512 // factor,
                                 bilinear)  # AggregateBlock(batch_size, hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        self.up2 = PrimalUpBlock(512 // factor + hidden_dim[2], 256 // factor,
                                 bilinear)  # AggregateBlock(batch_size, 512 // factor + hidden_dim[2], 256 // factor, bilinear)#512
        self.up3 = PrimalUpBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        self.up4 = PrimalUpBlock(128 // factor + hidden_dim[0], 64, bilinear)
        self.up5 = PrimalUpBlock(128, 64, bilinear)
        # self.up3 = AggregateBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        # self.up4 = AggregateBlock(128 // factor + hidden_dim[0], 64, bilinear)
        # self.up5 = AggregateBlock(128, 64, bilinear)
        self.tail = OutConv(64, out_channels)

        # self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, hidden_dim[1]))
        # # self.pos_drop1 = nn.Dropout(p=drop_rate)
        # self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, hidden_dim[2]))
        # # self.pos_drop2 = nn.Dropout(p=drop_rate)
        # self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, hidden_dim[3]))
        # # self.pos_drop3 = nn.Dropout(p=drop_rate)
        # self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, hidden_dim[4]))
        # # self.pos_drop4 = nn.Dropout(p=drop_rate)
        # # init weights
        # nn.init.trunc_normal_(self.pos_embed1, std=.02)
        # nn.init.trunc_normal_(self.pos_embed2, std=.02)
        # nn.init.trunc_normal_(self.pos_embed3, std=.02)
        # nn.init.trunc_normal_(self.pos_embed4, std=.02)
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0.)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0.)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_pre(self, x):
        ...

    def forward_post(self, I):
        # x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        bs = I.shape[0]
        # x = self.pa(x)

        # N,C,H,W
        x0 = I
        for blk in self.head:
            x0 = blk(x0)
        # stage 1 H,W // 2, 4*C=128->64, x.shape: [B, P, N, C]
        x, (H, W, D) = self.patch_embed1(x0, bs)  # x
        x_h, (H, W, D) = self.patch_hire1(x0, bs)
        # x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, x_h, H, W)
        # [B, P, N, C]->[BP, H, W, C]，用于patch_embed
        x1 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # stage 2 -128
        x, (H, W, D) = self.patch_embed2(x1, bs)
        x_h, (H, W, D) = self.patch_hire2(x1, bs)
        # x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, x_h, H, W)
        x2 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # stage 3 -320
        x, (H, W, D) = self.patch_embed3(x2, bs)
        x_h, (H, W, D) = self.patch_hire3(x2, bs)
        # x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, x_h, H, W)
        x3 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # stage 4 -512
        x, (H, W, D) = self.patch_embed4(x3, bs)
        x_h, (H, W, D) = self.patch_hire4(x3, bs)
        # x = x + self.pos_embed4
        for blk in self.down4:
            x = blk(x, x_h, H, W)
        x4 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # x = self.up5(x, x0)
        x = self.up1(x4, x3, bs)
        x = self.up2(x, x2, bs)
        x = self.up3(x, x1, bs)
        x = self.up4(x, x0, bs)
        # B, H, W, _ = x.shape
        # x = x.flatten(2).transpose(2, 1)
        # x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        logits = self.tail(x) + I
        # logits = self.norm(logits)
        return logits

    def forward(self, x):

        if self.norm_pre:
            return self.forward_pre(x)
        return self.forward_post(x)


    def forward_chop(self, x, shave=12):
        args = self.args
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(self.patch_size)
        shave = int(self.patch_size / 2)
        # print(self.scale, self.idx_scale)
        scale = 1#self.scale[self.idx_scale]

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = F.unfold(x, padsize, stride=int(shave / 2)).transpose(0, 2).contiguous()

        ################################################
        # 最后一块patch单独计算
        ################################################

        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut = self.forward(x_hw_cut.cuda()).cpu()

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_cut = self.cut_w(x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 左上patch单独计算，不是平均而是覆盖
        ################################################

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(x_h_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)
        y_w_top = self.cut_w(x_w_top, h, w, h_cut, w_cut, padsize, shave, scale, batchsize)

        # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # img->patch，最大计算crop_s个patch，防止bs*p*p太大
        ################################################

        x_unfold = x_unfold.view(x_unfold.size(0), -1, padsize, padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append(self.forward(
            x_unfold[i * batchsize:(i + 1) * batchsize, ...]).cpu())
            # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())

        y_unfold = torch.cat(y_unfold, dim=0)

        y = F.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, (w - w_cut) * scale), padsize * scale,
                   stride=int(shave / 2 * scale))
        # 312， 480
        # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 第一块patch->y
        ################################################
        y[..., :padsize * scale, :] = y_h_top
        y[..., :, :padsize * scale] = y_w_top
        # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        y_unfold = y_unfold[...,
                   int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                   int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        #1，3，24，24
        y_inter = F.fold(y_unfold.view(y_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                         ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                         padsize * scale - shave * scale,
                         stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_inter.shape, dtype=y_inter.dtype)
        divisor = F.fold(F.unfold(y_ones, padsize * scale - shave * scale,
                                  stride=int(shave / 2 * scale)),
                         ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                         padsize * scale - shave * scale,
                         stride=int(shave / 2 * scale))

        y_inter = y_inter / divisor
        # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 第一个半patch
        ################################################
        y[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale),
        int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_inter
        # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2 * scale), :],
                       y_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
        #图分为前半和后半
        # x->y_w_cut
        # model->y_hw_cut
        y_w_cat = torch.cat([y_w_cut[..., :y_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
                             y_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
        y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2 * scale)],
                       y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
        # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        # plt.show()

        return y.cuda()

    def cut_h(self, x_h_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        for i in range(x_range):
            y_h_cut_unfold.append(self.forward(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                             ...]).cpu())  # P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_h_cut_unfold = torch.cat(y_h_cut_unfold, dim=0)

        y_h_cut = F.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_h_cut_unfold = y_h_cut_unfold[..., :,
                         int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
        y_h_cut_inter = F.fold(
            y_h_cut_unfold.view(y_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_h_cut_inter.shape, dtype=y_h_cut_inter.dtype)
        divisor = F.fold(
            F.unfold(y_ones, (padsize * scale, padsize * scale - shave * scale),
                                       stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
            (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
        y_h_cut_inter = y_h_cut_inter / divisor

        y_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = y_h_cut_inter
        return y_h_cut

    def cut_w(self, x_w_cut, h, w, h_cut, w_cut, padsize, shave, scale, batchsize):

        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).transpose(0,
                                                                                                       2).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()
        for i in range(x_range):
            y_w_cut_unfold.append(self.forward(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                             ...]).cpu())  # P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        y_w_cut_unfold = torch.cat(y_w_cut_unfold, dim=0)

        y_w_cut = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
        y_w_cut_unfold = y_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                         :].contiguous()
        y_w_cut_inter = torch.nn.functional.fold(
            y_w_cut_unfold.view(y_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
            ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
            stride=int(shave / 2 * scale))

        y_ones = torch.ones(y_w_cut_inter.shape, dtype=y_w_cut_inter.dtype)
        divisor = torch.nn.functional.fold(
            torch.nn.functional.unfold(y_ones, (padsize * scale - shave * scale, padsize * scale),
                                       stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
            (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
        y_w_cut_inter = y_w_cut_inter / divisor

        y_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = y_w_cut_inter
        return y_w_cut


def test_net():
    from torchstat import stat
    x = torch.randn(1, 3, 128, 128).cuda()
    # net = Pvt(patch_size=128, in_channels=3, out_channels=3, patch_stride=[2, 2, 2, 2],
    #         hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
    #         depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    scale = 2
    net = HPT(image_size=128, in_channels=3, out_channels=3,
              down_scale=[[scale, scale], [scale, scale], [scale, scale], [scale, scale]],
              patch_stride=[scale, scale, scale, scale],
              hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
              depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    out = net(x)
    print(out.shape)


def test_module():
    x = torch.randn(1, 3, 128, 128).cuda()
    x1 = torch.randn(1, 3, 64, 64).cuda()
    # module = HireSharePatch(img_size=128, down_scale=[2, 2], in_chans=3, embed_dim=32, num_patch=True,
    #                         stride_ratio=2).cuda()
    module = MHSPatch(img_size=[[128, 128], [64, 64]], patch_size=[32, 32], in_chans=3, embed_dim=32, num_patch=True, stride_ratio=2).cuda()
    print(module([x, x1], 1, compatiable=False)[0].shape)


if __name__ == "__main__":
    # model = test_other_module()
    # test_net()
    test_module()
    # ckpt = partial_load_checkpoint(torch.load("../PVT/pvt_tiny.pth"))
    # model.load_state_dict(ckpt, strict=False)
    # x = torch.randn(1, 3, 64, 64)
    # pred = model(x)

    # print(pred.shape)

    # x = torch.rand([2, 2, 3])
    # fc = torch.nn.Linear(3, 1)
    # out = fc(x)
    # print(fc.weight)
    # print(out.shape)

    # x = torch.rand(2, 3, 8,8)
    # print(x.unfold(3, 4, 4).shape)
