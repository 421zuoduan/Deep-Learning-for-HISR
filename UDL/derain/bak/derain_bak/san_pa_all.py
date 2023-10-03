import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from einops import rearrange
# from image_preprocessing_t import PatchifyAugment

######################################################
# 构造PatchesEmbedding，先得到patch再考虑如何处理patch
######################################################

class PatchMerging(nn.Module):
    def __init__(self, in_channels, out_channels, downscaling_factor, compatiable=True):
        super().__init__()
        # self.downscaling_factor = downscaling_factor
        self.downscaling_factor = downscaling_factor = int(downscaling_factor[0])#list(downscaling_factor.numpy())[0]
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

    def __init__(self, img_size=224, patch_sride=16, in_chans=3, embed_dim=768, compatiable=False):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patch_sride)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
        # self.pa = PatchifyAugment(False, self.H)
        self.comp = compatiable

    def forward(self, x, bs):
        B, C, H, W = x.shape
        p = B // bs

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]
        if len(x.shape) == 3:
            x = x.reshape(bs, p, H * W, -1)

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


class PatchNet(nn.Module):
    def __init__(self, img_size, windows_size, pad, in_channels, out_channels, identity=True, patch_func=None,
                 residual_func=None):
        super(PatchNet, self).__init__()

        # assert isinstance(module, IdentityModule)
        # self.identity = identity
        # if identity:
        #     if module is not None:
        #         self.module = module
        #     else:
        #         self.module = lambda x: x

        self.identity = identity
        if patch_func is not None:
            self.patch_func = patch_func
        else:
            self.patch_func = lambda x: x

        if residual_func is not None:
            self.residual_func = residual_func
        else:
            self.residual_func = lambda x: x

        self.num_patches = (img_size - windows_size + 2 * self.pad) // self.stride + 1
        self.num_patches = self.num_patches ** 2

    def pad2d(self, x):

        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride) - 1) * self.stride - w + self.windows_size
        extra_v = (math.ceil(h / self.stride) - 1) * self.stride - h + self.windows_size

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = F.pad(x, [left, right, top, bottom])

        return x

    def im2col(self, x):
        x = self.pad2d(x)
        x = x.unfold(2, self.windows_size, self.stride)
        x_0 = x.unfold(3, self.windows_size, self.stride)
        n, c, h, w, k, j = self.size = x_0.size()
        x = x_0.reshape(n, c, h * w, k, j)
        del x_0

        return x

    def shuffle_chnls(self, x, groups=2):
        """Channel Shuffle, Channel = inchannel * patch"""

        bs, chnls, h, w = x.data.size()
        if chnls % groups:
            return x
        chnls_per_group = chnls // groups
        x = x.view(bs, groups, chnls_per_group, h, w)
        x = torch.transpose(x, 1, 2).contiguous()
        x = x.view(bs, -1, h, w)
        return x

    def randperm_chnls(self, x, dim):
        return x[..., torch.randperm(x.size(dim))]

    def forward_nCPk_j(self, x):
        '''

        '''
        residual = x
        _, _, x_h, x_w = x.size()

        x = self.im2col(x)
        n, c, h, w, k, j = self.size
        p = h * w
        x_cp = x.reshape(n, c * p, k, j)

        out_ps = self.patch_func(x_cp)

        out_ps = out_ps.reshape(n, -1, h, w, k, j).contiguous()

        if self.identity:
            out_ps = rearrange(out_ps, "n c h w k j -> n c (h k) (w j) ")

            # out_ps = self.module(out_ps, residual)
            residual = self.residual_func(residual)

            out_ps = out_ps + residual

        return out_ps


###############################################################
# PatchExtracctor
###############################################################

# from swin_transformer.\
#     swin_transformer_pytorch.swin_transformer import SwinBlock, WindowAttention


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
        x = x.reshape(-1, H, W, C).permute(0, 3, 1, 2)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(b, p, -1, C)
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
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., compatiable=True):
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
        self.patch_embed2 = PatchConvEmbed(img_size=patch_size // 2, patch_sride=patch_stride[1], in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2])
        self.patch_embed3 = PatchConvEmbed(img_size=patch_size // 4, patch_sride=patch_stride[2], in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3])
        self.patch_embed4 = PatchConvEmbed(img_size=patch_size // 8, patch_sride=patch_stride[3], in_chans=hidden_dim[3],
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
            dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[0])
            for i in range(depths[0])])
        self.down2 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[1])
            for i in range(depths[1])])
        self.down3 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[2])
            for i in range(depths[2])])
        factor = 2 if bilinear else 1
        #TODO: 扩维更合适
        self.down4 = nn.ModuleList([PvtBlock(
            dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[3])
            for i in range(depths[3])])

        # self.norm = nn.LayerNorm(64, eps=1e-6)
        self.norm = nn.LayerNorm([hidden_dim[1], patch_size, patch_size], eps=1e-6)
        # self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)

        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        self.up2 = PrimalUpBlock(512 // factor + hidden_dim[2], 256 // factor, bilinear)#512
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

        x0 = self.inc(x)#32
        # stage 1 -64
        x, (H, W, C) = self.patch_embed1(x0, bs)  # x
        x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, H, W)
        x1 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2 -128
        x, (H, W, C)  = self.patch_embed2(x1, bs)
        x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, H, W)
        x2 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3 -320
        x, (H, W, C)  = self.patch_embed3(x2, bs)
        x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, H, W)
        x3 = x.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4 -512
        x, (H, W, C)  = self.patch_embed4(x3, bs)
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
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.inc = ConvBlock(48, 16, 16, 0, in_channels, 64)
        self.down1 = Down(image_size=24, windos_size=8, stride=8, pad=0, in_channels=64, out_channels=128)
        self.down2 = Down(12, 4, 4, 0, 128, 256)
        self.down3 = Down(6, 2, 2, 0, 256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)
        self.up1 = Up(6, 2, 2, 0, 1024, 512 // factor, bilinear)
        self.up2 = Up(12, 4, 4, 0, 512, 256 // factor, bilinear)
        self.up3 = Up(24, 8, 8, 0, 256, 128 // factor, bilinear)
        self.up4 = Up(48, 16, 16, 0, 128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


def test_net():
    from torchstat import stat
    x = torch.randn(1, 3, 128, 128).cuda()
    # net = Pvt(patch_size=128, in_channels=3, out_channels=3, patch_stride=[2, 2, 2, 2],
    #         hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
    #         depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    net = HPT(patch_size=[torch.tensor([2, 2]), torch.tensor([2, 2])], image_size=128, in_channels=3, out_channels=3,
        patch_stride=[2, 2, 2, 2],
        hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1]).cuda()
    out = net(x)
    print(out.shape)

    '''
    HPT
    =================================================================================================================================================================================
    Total params: 457,620,307
    ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 331.53MB
    Total MAdd: 421.65GMAdd
    Total Flops: 747.63GFlops
    Total MemR+W: 7.68GB
    
    
    PVT
    =================================================================================================================================================================
    Total params: 16,014,675
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 22.05MB
    Total MAdd: 1.23GMAdd
    Total Flops: 941.16MFlops
    Total MemR+W: 1.3GB
    
    '''


    '''
    ==========================================================================================================================================================================================
    Total params: 130,086,915
    ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 17.34MB
    Total MAdd: 5.54GMAdd
    Total Flops: 2.77GFlops
    Total MemR+W: 531.18MB
    
    ================================================================================================================================================================
    Total params: 6,814,595
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 24.91MB
    Total MAdd: 1.07GMAdd
    Total Flops: 789.17MFlops
    Total MemR+W: 1.01GB
    ======================================================================================================================================================================
    Total params: 76,268,995
    ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 67.67MB
    Total MAdd: 2.24GMAdd
    Total Flops: 3.9GFlops
    Total MemR+W: 10.75GB
    '''



# @register_model
# def test_other_module():
#     # x = torch.randn(1, 16 * 16, 64 * 16)  # 1 256 1024
#     # net = AttentionFromPvt(64 * 16, 8)
#     # out = net(x, 16, 16)
#     # print(out.shape)
#
#     # x = torch.randn(1, 16*16, 64*16)#1 256 1024
#     # net = Attention(64*16, 8)
#     # out = net(x)
#     # print(out.shape)
#
#     # x = torch.randn(1, 16 * 16, 64 * 16)  # 1 256 1024
#     # net = PvtBlock(64*16, 8)
#     # out = net(x, 16, 16)
#     # print(out.shape)
#
#     model = Pvt(
#         patch_size=64, in_channels=3, out_channels=3, patch_sride=[2, 2, 2, 2],
#         hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
#         depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1])
#
#     model.default_cfg = _cfg()
#
#
#     return model





def partial_load_checkpoint(state_dict):
    pretrained_dict = {}
    dismatch_list = ['pos_embed', 'norm']
    for module in state_dict.items():
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
        # 2. overwrite entries in the existing state dict
        k, v = module
        if all(m not in k for m in dismatch_list):
            # print(k)
            pretrained_dict.update({k: v})

    return pretrained_dict


class HAttention(nn.Module):
    def __init__(self, downscaling_factor, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(downscaling_factor)])
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

class PAttention(nn.Module):
    def __init__(self, downscaling_factor, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.H_module = nn.ModuleList([nn.Sequential(nn.Linear(dim, dim, bias=qkv_bias),
                                                     nn.Linear(dim, dim * 2, bias=qkv_bias),
                                                     nn.Dropout(attn_drop)) for i in range(downscaling_factor)])
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
        out = torch.zeros(B, 1, N_q, C).to(x_h.device)


        for i, blk in enumerate(self.H_module):
            q = blk[0]
            kv = blk[1]
            attn_drop = blk[2]

            q = q(x_s).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            q = q.reshape(B, self.num_heads, 1, -1)  # B, heads, N_q*D

            kv = kv(x_h).reshape(B, p, N, 2, self.num_heads, C // self.num_heads).permute(3, 0, 4, 1, 2, 5)#2, B, heads, p, N_q, C'= D
            kv = kv.reshape(2, B, self.num_heads, p, -1)
            k, v = kv[0], kv[1]

            attn = (q @ k.transpose(-2, -1)) * self.scale #B, heads, 1, p
            # attn = attn.softmax(dim=-1)
            attn = attn_drop(attn)
            x_h = (attn @ v).transpose(1, 2).reshape(B, N_q, D)

            x_h = self.proj(x_h)
            x_h = self.proj_drop(x_h)

            out[:, i, ...] = x_h

        return out

class HBlock(nn.Module):

    def __init__(self, downscaling_factor, norm_dim, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True, hybrid_ffn=True):#compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)#dim
        # the effect of sr_ratio
        self.attn = HAttention(downscaling_factor=downscaling_factor,
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, x, H, W):
        #x: (b,P,N,c)
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

    def __init__(self, downscaling_factor, norm_dim, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, compatiable=True, hybrid_ffn=True):#compatiable=True即p=1，BPNC，而不是BNC
        super().__init__()
        self.norm1 = norm_layer(dim if compatiable else norm_dim)#dim
        self.norm_h1 = norm_layer(dim if compatiable else norm_dim)  # dim
        # the effect of sr_ratio
        self.attn = PAttention(downscaling_factor=downscaling_factor,
            dim=dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim if compatiable else norm_dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        if hybrid_ffn:
            self.ffn = HyFFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop, compatiable=compatiable)
        else:
            self.ffn = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def forward(self, x_s, x_h, H, W):
        #x: (b,P,N,c)
        x = x_s + self.attn(self.norm1(x_s), self.norm_h1(x_h), H, W)  # self.drop_path(self.attn(self.norm1(x), H, W))
        x = x_s + self.ffn(self.norm2(x), H, W)  # self.drop_path(self.mlp(self.norm2(x)))

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
    def __init__(self, downscaling_factor, patch_stride, in_chans, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.downscaling_factor = list(downscaling_factor.numpy())
        # self.patch_merge = nn.Unfold(kernel_size=int(self.downscaling_factor[0]), stride=int(self.downscaling_factor[0]), padding=0)
        self.linear = nn.Linear(in_chans, embed_dim)

    def forward(self, x, bs, axes=[2, 3]): # h, w
        b, c, h, w = x.shape
        # b, c, *hw = x.shape
        p = b // bs
        # shape = [hw[axis] // self.downscaling_factor[axis] for axis in axes]
        new_h, new_w = shape = [h // self.downscaling_factor[0], w // self.downscaling_factor[1]]#
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
            x = x.unfold(axes[0], new_h, new_h // 2).unfold(axes[1], new_w, new_w // 2).\
                reshape(bs, c, -1, new_h, new_w).permute(0, 2, 3, 4, 1).contiguous()

        x = x.reshape(bs, x.shape[1], -1, c)
        x = self.linear(x)
        return x, (*shape, self.embed_dim) #(b,p,n_h,n_w,c)

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
    def __init__(self, image_size, patch_size, patch_stride, in_channels, out_channels, hidden_dim,
                 num_heads, mlp_ratios, depths, sr_ratio,
                 norm_pre=False, bilinear=True, compatiable=True):
        super(HPT, self).__init__()

        self.patch_num = patch_num = 2 * patch_size
        self.norm_pre = norm_pre
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.inc = PrimalConvBlock(in_channels, hidden_dim[0])

        # self.patch_embed1 = HirePatch(downscaling_factor=patch_size[0], patch_stride=patch_stride[0], in_chans=hidden_dim[0],
        #                                    embed_dim=hidden_dim[1])
        # self.patch_embed2 = HirePatch(downscaling_factor=patch_size[1], patch_stride=patch_stride[1], in_chans=hidden_dim[1],
        #                                    embed_dim=hidden_dim[2])
        # self.patch_embed3 = HirePatch(downscaling_factor=patch_size, patch_stride=patch_stride[2], in_chans=hidden_dim[2],
        #                                    embed_dim=hidden_dim[3], batch_size=batch_size)
        # self.patch_embed4 = HirePatch(downscaling_factor=patch_size, patch_stride=patch_stride[3], in_chans=hidden_dim[3],
        #                                    embed_dim=hidden_dim[4], batch_size=batch_size)
        self.patch_embed1 = PatchMerging(in_channels=hidden_dim[0], out_channels=hidden_dim[1], downscaling_factor=patch_num[0],
                                         compatiable=True)
        self.patch_hire1 = HirePatch(downscaling_factor=patch_size[0], patch_stride=patch_stride[0], in_chans=hidden_dim[0],
                                           embed_dim=hidden_dim[1])
        self.patch_embed2 = PatchMerging(in_channels=hidden_dim[1], out_channels=hidden_dim[2], downscaling_factor=patch_num[0],
                                         compatiable=True)
        self.patch_hire2 = HirePatch(downscaling_factor=patch_size[1], patch_stride=patch_stride[1], in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2])

        self.patch_embed3 = PatchMerging(in_channels=hidden_dim[2], out_channels=hidden_dim[3], downscaling_factor=patch_num[0],
                                         compatiable=True)
        self.patch_hire3 = HirePatch(downscaling_factor=patch_size[1], patch_stride=patch_stride[2], in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3])
        self.patch_embed4 = PatchMerging(in_channels=hidden_dim[3], out_channels=hidden_dim[4], downscaling_factor=patch_num[0],
                                         compatiable=True)
        self.patch_hire4 = HirePatch(downscaling_factor=patch_size[1], patch_stride=patch_stride[3], in_chans=hidden_dim[3],
                                           embed_dim=hidden_dim[4])
        # self.patch_embed3 = PatchConvEmbed(img_size=image_size // 4, patch_sride=patch_stride[2], in_chans=hidden_dim[2],
        #                                    embed_dim=hidden_dim[3], compatiable=compatiable)
        # self.patch_embed4 = PatchConvEmbed(img_size=image_size // 8, patch_sride=patch_stride[3], in_chans=hidden_dim[3],
        #                                    embed_dim=hidden_dim[4], compatiable=compatiable)
        N = image_size * image_size
        num_patches = 1 if compatiable else torch.prod(patch_num[0]) ** 1
        self.down1 = nn.ModuleList([HPBlock(downscaling_factor= num_patches, norm_dim=[num_patches, N // num_patches, hidden_dim[1]],
            dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[0], compatiable=compatiable)
            for i in range(depths[0])])
        num_patches_base = torch.prod(patch_num[1])
        num_patches1 = num_patches_base ** 2
        self.down2 = nn.ModuleList([HPBlock(downscaling_factor=num_patches, norm_dim=[num_patches1, N // num_patches1, hidden_dim[2]],
            dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[1], compatiable=compatiable)
            for i in range(depths[1])])
        num_patches = num_patches_base * num_patches1
        self.down3 = nn.ModuleList([HPBlock(downscaling_factor=1, norm_dim=[num_patches1, N // num_patches, hidden_dim[3]],#[k ** 3 for k in patch_num],
            dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[2], compatiable=compatiable)
            for i in range(depths[2])])
        factor = 2 if bilinear else 1
        #TODO: 扩维更合适
        num_patches = num_patches_base * num_patches
        self.down4 = nn.ModuleList([HPBlock(downscaling_factor=1, norm_dim=[num_patches1, N // num_patches, hidden_dim[4]],#[k ** 4 for k in patch_num],
            dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[3], compatiable=compatiable)
            for i in range(depths[3])])

        # self.norm = nn.LayerNorm(64, eps=1e-6)
        self.norm = nn.LayerNorm([hidden_dim[1], image_size, image_size], eps=1e-6)
        # self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)

        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)#AggregateBlock(batch_size, hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        self.up2 = PrimalUpBlock(512 // factor + hidden_dim[2], 256 // factor, bilinear)#AggregateBlock(batch_size, 512 // factor + hidden_dim[2], 256 // factor, bilinear)#512
        self.up3 = PrimalUpBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        self.up4 = PrimalUpBlock(128 // factor + hidden_dim[0], 64, bilinear)
        self.up5 = PrimalUpBlock(128, 64, bilinear)
        # self.up3 = AggregateBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        # self.up4 = AggregateBlock(128 // factor + hidden_dim[0], 64, bilinear)
        # self.up5 = AggregateBlock(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

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
        x0 = self.inc(I)
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
        x = self.norm(x)
        # x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        logits = self.outc(x) + I
        # logits = self.norm(logits)
        return logits

    def forward(self, x):

        if self.norm_pre:
            return self.forward_pre(x)
        return self.forward_post(x)


if __name__ == "__main__":
    # model = test_other_module()
    test_net()
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

