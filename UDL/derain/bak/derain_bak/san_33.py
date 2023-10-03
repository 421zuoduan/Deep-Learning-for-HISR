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
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.hidden_features = hidden_features

    #     self.apply(self._init_weights)
    #
    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class HyFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU(), drop=0.,
                 compatiable=True):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
    #         self.fc2 = nn.Linear(hidden_features, out_features)
    #         self.drop = nn.Dropout(drop)
    #         self.hidden_features = hidden_features
    #         self.conv = DWConv(hidden_features)
    #
    #     #     self.apply(self._init_weights)
    #     #
    #     # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         print(m)
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

    def forward(self, x, H, W):

        x = self.fc1(x)
        x = self.conv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



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
            #     ResBlock(out_channels, kernel_size=3)
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


        self.apply(self._reset_parameters)
        # self.sr_ratio = sr_ratio
        # if sr_ratio > 1:
        #     self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
        #     self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        out = torch.zeros_like(x)
        x = x.transpose(1, 2)
        B, p, N, C = x.shape

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

            out[:, :, i, ...] = sub_x

        return out

    def _reset_parameters(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0.)

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
        _, N_q, p_s, D = x_s.shape

        out = torch.zeros(B, N_q, 1, C).to(x_h.device)

        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(x_s).reshape(B, N_q, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            # q = q.reshape(B, self.num_heads, p_s, -1)  # B, heads, N_q*D

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

            out[:, :, i, ...] = x_s

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

        # self.apply(self._init_weights)

    def forward(self, x, H, W):
        # x: (b,P,N,c)
        x = x + self.attn(self.norm1(x), H, W)  # self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.ffn(self.norm2(x), H, W)  # self.drop_path(self.mlp(self.norm2(x)))

        return x

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()

class HPBlock_V2(nn.Module):

    def __init__(self, down_scale, norm_dim, dim, num_heads, num_patch,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=True):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim

        self.norm_embed = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_embed_h = norm_layer(dim if compatiable else norm_dim)  # dim

        # the effect of sr_ratio
        self.attn = PAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn_h = HAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        self.norm_h2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
            self.ffn_h = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                             compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.ffn_h = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, x_s, x_h, H, W):
        '''
        self-attn x@x^T
        attn: x_s@x_h@x_h的过程
        x_s只有1个聚合的patch
        '''
        # x: (b,P,N,c)
        x_s = self.norm1(x_s)
        p_x = x_s + self.attn(x_s, self.norm_h1(x_h), H, W)

        x = x_s + self.self_attn(self.norm_embed(p_x), H, W)
        x_h = x_h + self.self_attn_h(self.norm_embed_h(x_h), H, W)

        x = x + self.ffn(self.norm2(x), H, W)
        x_h = x_h + self.ffn_h(self.norm_h2(x_h), H, W)

        return x, x_h

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
                 drop_path=0., act_layer=nn.ReLU(inplace=True), norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True,
                 hybrid_ffn=False):  # compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)  # dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim
        # the effect of sr_ratio
        # self.attn = PAttention(dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #                        attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        self.self_attn = HAttention(dim=dim, num_heads=12, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # self.self_attn = nn.MultiheadAttention(dim, 12, dropout=attn_drop, bias=False)
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

        # self.apply(self._init_weights)

    def forward(self, x_s, x_h, H, W):
        # x: (b,P,N,c)
        x_s2 = self.norm1(x_s)
        # p_x = x_s + self.attn(x_s, self.norm_h1(x_h), H, W)
        x = x_s + self.self_attn(x_s2, x_s2, x_s2)[0]  #H, W)
        x = x + self.ffn(self.norm2(x), H, W)

        return x

    # def _init_weights(self, m):
    #     if isinstance(m, nn.Linear):
    #         nn.init.trunc_normal_(m.weight, std=.02)
    #         if isinstance(m, nn.Linear) and m.bias is not None:
    #             nn.init.constant_(m.bias, 0)
    #     elif isinstance(m, nn.LayerNorm):
    #         nn.init.constant_(m.bias, 0)
    #         nn.init.constant_(m.weight, 1.0)
    #     elif isinstance(m, nn.Conv2d):
    #         fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
    #         fan_out //= m.groups
    #         m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
    #         if m.bias is not None:
    #             m.bias.data.zero_()


class HirePatch(nn.Module):
    def __init__(self, down_scale, in_chans, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.downscaling_factor = list(down_scale)
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
    def __init__(self, img_size, patch_size, in_chans, embed_dim, num_patch=None, stride_ratio=2, K=16):
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
            self.num_patch = np.sum(num_patch)
            in_chans = num_patch.dot(in_chans)[0]

        if isinstance(num_patch, int) and num_patch == 1:
            Warning("HPatchConvEmbed will get non-local info from local windows")
        else:
            print(num_patch, in_chans)
            Warning("HPatchConvEmbed will get cross-scale non-local patch, equal to HirePatch, but share the weight among patchs")

        assert len(patch_size) == 2, print("per down_scale length should be 2, directions x,y")

        # self.patch_merge = nn.Unfold(kernel_size=int(self.downscaling_factor[0]), stride=int(self.downscaling_factor[0]), padding=0)
        # assert (embed_dim * K) % num_patch == 0, print("MHSPatch should be divided by num_patch")
        # 聚合多尺度patch到K个，每个embed_dim个通道
        self.linear = nn.Linear(in_chans, embed_dim * K)
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
                    if factor == 1:
                        continue
                    if axis == 2:
                        x = x.unfold(axis, ps, ps // 2).transpose(-1, -2)
                    elif axis == 3:
                        x = x.unfold(axis, ps, ps // 2).transpose(3, 2)
                    x = x.reshape(bs * factor * p, c, x.shape[-2], x.shape[-1])
                x = x.view(bs, -1, N).permute(0, 2, 1)#.permute(0, 2, 3, 1)
            else:
                x = x.unfold(axes[0], new_h, new_h // 2).unfold(axes[1], new_w, new_w // 2). \
                    reshape(b, -1, N).permute(0, 2, 1)#.permute(0, 2, 3, 1)
            patch_list.append(x)
        #b,n_h*n_w,CP
        patch_list = torch.cat(patch_list, dim=-1)#M*C*P

        # 得到多尺度的K
        patch_list = self.linear(patch_list)
        if compatiable:
            patch_list = patch_list.reshape(bs, N, num_patch, -1).permute(0, 2, 1, 3)

        return patch_list, (*patch_size, self.embed_dim)  # (b,p,n_h*n_w,c)


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
import copy
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos=None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm=False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

        # nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

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
                nn.Conv2d(in_channels, hidden_dim[1], kernel_size=3, padding=1),
                ResBlock(hidden_dim[1], kernel_size=5),
                ResBlock(hidden_dim[1], kernel_size=5)
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
        self.flatten_dim = embedding_dim = 9 * hidden_dim[1]
        self.out_dim = 9 * hidden_dim[1]
        n_dim = 36 * hidden_dim[1]
        self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
        self.linear_encoding_h = nn.Linear(n_dim, embedding_dim)

        self.mlp_head = nn.Sequential(
            nn.Linear(embedding_dim, n_dim),
            nn.ReLU(),
            nn.Linear(n_dim,  self.out_dim)
        )

        # self.patch_pool = MHSPatch(img_size=[[64, 64], [32, 32], [16, 16]], patch_size=[16, 16],
        #                            in_chans=[hidden_dim[1], hidden_dim[2], hidden_dim[3]], embed_dim=320, num_patch=True, stride_ratio=2)
        # self.pool_attn = nn.ModuleList([HPBlock(down_scale=1,
        #                                     norm_dim=[num_patches3, patch_size3, hidden_dim[3]],
        #                                     num_patch=num_patches3,
        #                                     dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
        #                                     qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                     sr_ratio=sr_ratio[2], compatiable=compatiable)
        #                             for i in range(depths[2])])



        self.down1 = nn.ModuleList(
            [HPBlock(down_scale=1, norm_dim=[num_patches1, patch_size1, hidden_dim[1]], num_patch=num_patches1,
                     dim=embedding_dim, num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None,
                     norm_layer=partial(nn.LayerNorm, eps=1e-6),
                     sr_ratio=sr_ratio[0], compatiable=compatiable)
             for i in range(12)])

        # encoder_layer = TransformerEncoderLayer(embedding_dim, 12, n_dim)
        # self.encoder = TransformerEncoder(encoder_layer, 12)

        # self.down2 = nn.ModuleList(
        #     [HPBlock(down_scale=1, norm_dim=[num_patches2, patch_size2, hidden_dim[2]], num_patch=num_patches2,
        #              dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None,
        #              norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #              sr_ratio=sr_ratio[1], compatiable=compatiable)
        #      for i in range(depths[1])])
        # self.down3 = nn.ModuleList([HPBlock(down_scale=1,
        #                                     norm_dim=[num_patches3, patch_size3, hidden_dim[3]],
        #                                     num_patch=num_patches3,
        #                                     dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
        #                                     qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                     sr_ratio=sr_ratio[2], compatiable=compatiable)
        #                             for i in range(depths[2])])
        # factor = 2 if bilinear else 1
        # self.down4 = nn.ModuleList([HPBlock(down_scale=1,
        #                                     num_patch=num_patches4,
        #                                     norm_dim=[num_patches4, patch_size4, hidden_dim[4]],
        #                                     # [k ** 4 for k in patch_num],
        #                                     dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
        #                                     qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                     sr_ratio=sr_ratio[3], compatiable=compatiable)
        #                             for i in range(depths[3])])
        #
        # # Decoder
        # self.norm = nn.LayerNorm([hidden_dim[1], image_size, image_size], eps=1e-6)

        # self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], 512 // factor,
        #                          bilinear)  # AggregateBlock(batch_size, hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        # self.up2 = PrimalUpBlock(512 // factor + hidden_dim[2], 256 // factor,
        #                          bilinear)  # AggregateBlock(batch_size, 512 // factor + hidden_dim[2], 256 // factor, bilinear)#512
        # self.up3 = PrimalUpBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        # self.up4 = PrimalUpBlock(128 // factor + hidden_dim[0], 64, bilinear)
        # self.up5 = PrimalUpBlock(128, 64, bilinear)


        # self.up3 = AggregateBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        # self.up4 = AggregateBlock(128 // factor + hidden_dim[0], 64, bilinear)
        # self.up5 = AggregateBlock(128, 64, bilinear)


        self.tail = OutConv(hidden_dim[1], out_channels)

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
            # print("1:", m.groups)
            # nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            # nn.init.constant_(m.bias, 0.)
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_pre(self, x):
        ...

    def forward_post(self, I):

        bs = I.shape[0]

        x0 = I
        # N,C,H,W
        for blk in self.head:
            x0 = blk(x0)
        # bs,128,128,3->bs,128,128,32->patch2win3x3: [(128-3)/3+1]=42个
        # 42*42,bs, 9*32 = 1764，bs，288，让1764个patch共享9 * n_feat个权重参数
        # 类比卷积，卷积是1个feat共享，也就是1764个win共享9个参数，有n_feat个
        # 总参数 9 * n_feat，但每个feat的patch只共享了9个
        # 他们都是全patch查询，克服ViT的patch聚合带来的信息损失
        #
        # 更好地结合了空间和谱，TODO：把这个用在卷积上
        x = torch.nn.functional.unfold(x0,
             3, stride=3).transpose(1, 2).transpose(0, 1).contiguous()
        # x_h = torch.nn.functional.unfold(x0,
        #      6, stride=6).transpose(1, 2).transpose(0, 1).contiguous()
        x = self.linear_encoding(x) + x
        # x_h = self.linear_encoding_h(x_h)
        #3x3下，N,c->N*c， b,1,P,Nc(D),近似(b,P,N,c)
        #P,Nc(D) x D,P x P,Nc -> P,Nc(D)
        x = x.transpose(1, 0).unsqueeze(2)
        # x_h = x_h.transpose(1, 0)#.unsqueeze(2)
        # x = x + self.pos_embed1
        for blk in self.down1:
            #8,256,1,576
            x = blk(x, x, 9, 9)
        # x = self.encoder(x)
        # x = self.mlp_head(x) + x
        #1，256，1，288
        x = x.transpose(0, 1).contiguous().view(x.size(1), -1, self.flatten_dim)
        #256，1，288
        x = torch.nn.functional.fold(x.permute(1, 2, 0).contiguous(), self.patch_size, 3,
                                     stride=3)
        # x = torch.nn.functional.fold(x.transpose(1, 2).contiguous(), self.patch_size, 3,
        #                              stride=3)
        logits = self.tail(x) + I

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
    x = torch.randn(1, 3, 48, 48).cuda()
    # net = Pvt(patch_size=128, in_channels=3, out_channels=3, patch_stride=[2, 2, 2, 2],
    #         hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
    #         depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    scale = 2
    net = HPT(None, image_size=48, in_channels=3, out_channels=3,
              down_scale=[[scale, scale], [scale, scale], [scale, scale], [scale, scale]],
              patch_stride=[scale, scale, scale, scale],
              hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
              depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    out = net(x)
    print(out.shape)


def test_module():
    x = torch.randn(1, 3, 128, 128).cuda()
    x1 = torch.randn(1, 32, 64, 64).cuda()
    x2 = torch.randn(1, 64, 32, 32).cuda()
    # module = HireSharePatch(img_size=128, down_scale=[2, 2], in_chans=3, embed_dim=32, num_patch=True,
    #                         stride_ratio=2).cuda()
    module = MHSPatch(img_size=[[128, 128], [64, 64], [32, 32]], patch_size=[32, 32],
                      in_chans=[3, 32, 64], embed_dim=320, num_patch=True, stride_ratio=2).cuda()
    print(module([x, x1, x2], 1, compatiable=True)[0].shape)


if __name__ == "__main__":
    # model = test_other_module()
    test_net()
    # test_module()