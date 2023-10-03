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
    def __init__(self, in_channels, out_channels, downscaling_factor):
        super().__init__()
        self.downscaling_factor = downscaling_factor
        self.patch_merge = nn.Unfold(kernel_size=downscaling_factor, stride=downscaling_factor, padding=0)
        self.linear = nn.Linear(in_channels * downscaling_factor ** 2, out_channels)

    def forward(self, x):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h, new_w).permute(0, 2, 3, 1)
        x = self.linear(x)
        return x


class PatchConvEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_sride=16, in_chans=3, embed_dim=768):
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

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


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

        x = NF.pad(x, [left, right, top, bottom])

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
        x = x + self.mlp(self.norm2(x))  # self.drop_path(self.mlp(self.norm2(x)))

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
                BIMBlockND(in_channels, mid_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True,
                           False, False),
                # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                BIMBlockND(mid_channels, out_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True,
                           False, False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                # BIMBlockND(in_channels, mid_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True, False, False),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                # BIMBlockND(mid_channels, out_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True, False, False),
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


class APUNet(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(APUNet, self).__init__()
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        patch_size = 64
        hidden_dim = 32
        # self.pa = PatchifyAugment(False, grid_size=patch_size)
        self.inc = ConvBlock(48, 16, 16, 0, in_channels, 32, split_patch=False)
        # self.down1 = DownBlock(image_size=24, windos_size=8, stride=8, pad=0, in_channels=64, out_channels=128)
        # self.down2 = DownBlock(12, 4, 4, 0, 128, 256)
        # self.down3 = DownBlock(6, 2, 2, 0, 256, 512)

        self.patch_embed1 = PatchConvEmbed(img_size=patch_size, patch_size=2, in_chans=hidden_dim,
                                           embed_dim=hidden_dim * 2)
        self.patch_embed2 = PatchConvEmbed(img_size=patch_size // 2, patch_size=2, in_chans=hidden_dim * 2,
                                           embed_dim=hidden_dim * 4)
        self.patch_embed3 = PatchConvEmbed(img_size=patch_size // 4, patch_size=2, in_chans=hidden_dim * 4,
                                           embed_dim=hidden_dim * 8)
        self.patch_embed4 = PatchConvEmbed(img_size=patch_size // 8, patch_size=2, in_chans=hidden_dim * 8,
                                           embed_dim=hidden_dim * 8)
        # self.patch_embed5 = PatchConvEmbed(img_size=3, patch_size=2, in_chans=256,
        #                                     embed_dim=256)

        self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, 64))
        # self.pos_drop1 = nn.Dropout(p=drop_rate)
        self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, 128))
        # self.pos_drop2 = nn.Dropout(p=drop_rate)
        self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, 256))
        # self.pos_drop3 = nn.Dropout(p=drop_rate)
        self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, 256))
        # self.pos_drop4 = nn.Dropout(p=drop_rate)

        self.down1 = nn.ModuleList([PvtBlock(
            dim=64, num_heads=4, mlp_ratio=8, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=1)
            for i in range(2)])
        self.down2 = nn.ModuleList([PvtBlock(
            dim=128, num_heads=8, mlp_ratio=8, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=1)
            for i in range(2)])
        self.down3 = nn.ModuleList([PvtBlock(
            dim=256, num_heads=16, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=1)
            for i in range(18)])
        factor = 2 if bilinear else 1
        #TODO: 扩维更合适
        self.down4 = nn.ModuleList([PvtBlock(
            dim=256, num_heads=32, mlp_ratio=4, qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=1)
            for i in range(2)])

        self.norm = nn.LayerNorm([64, patch_size, patch_size], eps=1e-6)
        # self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)
        self.up1 = UpBlock(6, 2, 2, 0, 512, 512 // factor, bilinear)
        self.up2 = UpBlock(12, 4, 4, 0, 384, 256 // factor, bilinear)
        self.up3 = UpBlock(24, 8, 8, 0, 192, 128 // factor, bilinear)
        self.up4 = UpBlock(48, 16, 16, 0, 96, 64, bilinear)
        self.up5 = UpBlock(6, 2, 2, 0, 64, 64, bilinear)
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

    def forward(self, x):

        x = x.view(x.shape[0] * x.shape[1], x.shape[2], x.shape[3], x.shape[4])

        B = x.shape[0]
        # x = self.pa(x)


        x0 = self.inc(x)
        # stage 1
        x, (H, W) = self.patch_embed1(x0)#x
        x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, H, W)
        x1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        x, (H, W) = self.patch_embed2(x1)
        x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, H, W)
        x2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        x, (H, W) = self.patch_embed3(x2)
        x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, H, W)
        x3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        x, (H, W) = self.patch_embed4(x3)
        x = x + self.pos_embed4
        for blk in self.down4:
            x = blk(x, H, W)
        x4 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # x = self.up5(x, x0)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.norm(x)

        logits = self.outc(x)

        return logits



'''
改造PVT以对应DETR中的transformer
这是一个U-shaped结构的Transformer
DETR: backbone + top-down Transformer + ConvNetSegmentation 
s.t. backbone + Pvt + ConvNetSegmentation
'''
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg

class Pvt(nn.Module):
    def __init__(self, patch_size, patch_sride, in_channels, out_channels, hidden_dim,
                 num_heads, mlp_ratios, depths, sr_ratio,
                 norm_pre=False, bilinear=True):
        super(Pvt, self).__init__()

        self.norm_pre = norm_pre
        self.n_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        self.inc = PrimalConvBlock(in_channels, hidden_dim[0])

        self.patch_embed1 = PatchConvEmbed(img_size=patch_size, patch_sride=patch_sride[0], in_chans=hidden_dim[0],
                                           embed_dim=hidden_dim[1])
        self.patch_embed2 = PatchConvEmbed(img_size=patch_size // 2, patch_sride=patch_sride[1], in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2])
        self.patch_embed3 = PatchConvEmbed(img_size=patch_size // 4, patch_sride=patch_sride[2], in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3])
        self.patch_embed4 = PatchConvEmbed(img_size=patch_size // 8, patch_sride=patch_sride[3], in_chans=hidden_dim[3],
                                           embed_dim=hidden_dim[4])
        # self.patch_embed5 = PatchConvEmbed(img_size=3, patch_size=2, in_chans=256,
        #                                     embed_dim=256)

        # self.pos_embed1 = nn.Parameter(torch.zeros(1, self.patch_embed1.num_patches, hidden_dim[1]))
        # # self.pos_drop1 = nn.Dropout(p=drop_rate)
        # self.pos_embed2 = nn.Parameter(torch.zeros(1, self.patch_embed2.num_patches, hidden_dim[2]))
        # # self.pos_drop2 = nn.Dropout(p=drop_rate)
        # self.pos_embed3 = nn.Parameter(torch.zeros(1, self.patch_embed3.num_patches, hidden_dim[3]))
        # # self.pos_drop3 = nn.Dropout(p=drop_rate)
        # self.pos_embed4 = nn.Parameter(torch.zeros(1, self.patch_embed4.num_patches, hidden_dim[4]))
        # # self.pos_drop4 = nn.Dropout(p=drop_rate)

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

        self.norm = nn.LayerNorm([hidden_dim[1], patch_size, patch_size], eps=1e-6)
        # self.down4 = Down(3, None, None, None, 512, 1024 // factor, split_patch=False)

        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        self.up2 = PrimalUpBlock(512 // factor + hidden_dim[2], 256 // factor, bilinear)#512
        self.up3 = PrimalUpBlock(256 // factor + hidden_dim[1], 128 // factor, bilinear)
        self.up4 = PrimalUpBlock(128 // factor + hidden_dim[0], 64, bilinear)
        self.up5 = PrimalUpBlock(128, 64, bilinear)
        self.outc = OutConv(64, out_channels)

        # # init weights
        # nn.init.trunc_normal_(self.pos_embed1, std=.02)
        # nn.init.trunc_normal_(self.pos_embed2, std=.02)
        # nn.init.trunc_normal_(self.pos_embed3, std=.02)
        # nn.init.trunc_normal_(self.pos_embed4, std=.02)
        # # self.apply(self._init_weights)

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

        B = x.shape[0]
        # x = self.pa(x)

        x0 = self.inc(x)#32
        # stage 1 -64
        x, (H, W) = self.patch_embed1(x0)  # x
        # x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, H, W)
        x1 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2 -128
        x, (H, W) = self.patch_embed2(x1)
        # x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, H, W)
        x2 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3 -320
        x, (H, W) = self.patch_embed3(x2)
        # x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, H, W)
        x3 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4 -512
        x, (H, W) = self.patch_embed4(x3)
        # x = x + self.pos_embed4
        for blk in self.down4:
            x = blk(x, H, W)
        x4 = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # x = self.up2(x4, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        # x = self.up5(x, x0)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.up4(x, x0)
        x = self.norm(x)

        logits = self.outc(x)

        return logits

    def forward(self, x):

        if self.norm_pre:
            return self.forward_pre(x)
        return self.forward_post(x)
    def init_eval_obj(self, args):
        self.args = args

    def forward_chop(self, x, shave=12):
        args = self.args
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(args.patch_size)
        shave = int(args.patch_size / 2)
        # print(self.scale, self.idx_scale)
        scale = 1  # self.scale[self.idx_scale]

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
        # 1，3，24，24
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
        # 图分为前半和后半
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


class BIMBlockND(nn.Module):
    def __init__(self, in_channels, out_channels, image_size, windows_size, stride, pad, mode, loc_conn_point,
                 shuffle_patches, is_conv, is_bn, bias=False):
        super(BIMBlockND, self).__init__()

        # self.shuffle_patches = shuffle_patches
        self.out_channels = out_channels
        self.windows_size = windows_size
        self.stride = stride
        self.pad = pad
        self.is_conv = is_conv
        self.is_bn = is_bn
        # self.image_size_list = [128, 64, 32, 16, 8, 4]
        self.mode = mode

        self.num_patches = (image_size - self.windows_size + 2 * self.pad) // self.stride + 1
        self.num_patches = self.num_patches ** 2

        transposed = False
        kernel_size = _pair(self.windows_size)
        if transposed:
            self.weight = nn.Parameter(torch.Tensor(
                in_channels, out_channels, *kernel_size))
        else:
            self.weight = nn.Parameter(torch.Tensor(
                out_channels, in_channels, *kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.softmax = nn.Softmax(dim=-1)
        # 能分成几块，需要知道原图大小、卷积核大小、步长、pad
        # mode = ["NPk_j", "nckjP", "nCPk_j", "N1kjP"]
        if self.mode == "NPk_j":
            self.g = nn.Conv2d(self.num_patches, self.num_patches, kernel_size=1, stride=1, bias=False)
        elif self.mode == "nckjP":
            # hw = self.windows_size * self.windows_size
            self.g = nn.Conv2d(in_channels, out_channels, kernel_size=loc_conn_point, stride=1, bias=False)
        elif self.mode == "nCPk_j":
            self.g = nn.Conv2d(self.num_patches * in_channels,
                               self.num_patches * out_channels, kernel_size=1, stride=1, bias=False)
        elif self.mode == "N1kjP":
            self.g = nn.Conv2d(1, 1, kernel_size=1, stride=1, bias=False)
        else:
            self.g = lambda x: x
        # is_conv则进行同尺寸不同层之间的核级别注意力
        if self.is_conv:
            self.theta = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.phi = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        else:
            self.theta = lambda x: x
            self.phi = lambda x: x

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def pad2d(self, x):
        # n, c, h_in, w_in = x.shape
        # # d, c, k, j = weight.shape
        # x_pad = torch.zeros(n, c, h_in + 2 * self.pad, w_in + 2 * self.pad)  # 对输入进行补零操作
        # if self.pad > 0:
        #     x_pad[:, :, self.pad:-self.pad, self.pad:-self.pad] = x
        # else:
        #     x_pad = x
        #
        # return x_pad
        h, w = x.shape[-2:]

        extra_h = (math.ceil(w / self.stride) - 1) * self.stride - w + self.windows_size
        extra_v = (math.ceil(h / self.stride) - 1) * self.stride - h + self.windows_size

        left = extra_h // 2
        right = extra_h - left
        top = extra_v // 2
        bottom = extra_v - top

        x = NF.pad(x, [left, right, top, bottom])

        return x

    # def conv2d(self, x_pad, k, j,stride=1):
    #     # x_pad = x_pad.fold(2, k, stride)
    #     # out = x_pad.fold(3, j, stride)  # 按照滑动窗展开
    #     # 1 8 2 2 16 16 | 8 2 16 16
    #     # out = torch.einsum(  # 按照滑动窗相乘，
    #     #     'nchwkj,dckj->ndhw',  # 并将所有输入通道卷积结果累加
    #     #     x_pad, self.weight)
    #     # if self.bias is not None:
    #     #     out = out + self.bias.view(1, -1, 1, 1)  # 添加偏置值
    #     # else:
    #     #     return out
    #     return out

    def im2col(self, x):
        x = self.pad2d(x)
        # 8, 4, 4
        x = x.unfold(2, self.windows_size, self.stride)
        x_0 = x.unfold(3, self.windows_size, self.stride)
        n, c, h, w, k, j = self.size = x_0.size()
        # print("init unfold", x_0.size())
        # x = x.reshape(n, c, h*w, -1) #把窗口摊平了，计算相似吧
        x = x_0.reshape(n, c, h * w, k, j)  # 不把窗口摊平,适合torch.mean和var，计算相似吧 = n, c, pathes, k, j
        # print(x)
        # print("hw unfold", x.size())  # 8, 2, 2, 3, 3 N,h,w,k,j h,w是输出大小,kj是kernel大小，计算
        # p = h * w
        # x_nc = x.reshape(n * c, p, k, j)
        # print(x_nc.size())
        # x_f = x.reshape(n, c, p, k * j)
        # print("flatten", x_f.size())
        del x_0

        return x  # x_f, x_nc

    def shuffle_chnls(self, x, groups=2):
        """Channel Shuffle"""

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

    def forward_NPk_j(self, x):
        # NPh_w,对通道方向等价于做深度可分离卷积
        # 或者排序，图更大了，3x3的获得的细节更大，按相似性筛选一下，扣除这些patches再最后用全黑的补在x上，作为x的残差，降低影响
        # 或者repeat通道,但是感受野不一样，等于没有
        # 粗暴点直接插值，但会极大降低大图上的效果
        residual = x
        _, _, x_h, x_w = x.size()
        p_in, p_out, _, _ = self.g.weight.data.size()
        g_h = math.sqrt(p_in)
        if g_h != x_h:
            # torch.Size([1, 64, 128, 128]) 1024 32.0
            # print(x.size(), p_in, g_h)
            x = NF.interpolate(x, size=_pair(int(g_h)), mode='bilinear', align_corners=True)

        # x = self.pad2d(x)
        # # 8, 4, 4
        # x = x.unfold(2, self.windows_size, self.stride)
        # x_0 = x.unfold(3, self.windows_size, self.stride)
        # n, c, h, w, k, j = x_0.size()
        # print("init unfold", x_0.size())
        # # x = x.reshape(n, c, h*w, -1) #把窗口摊平了，计算相似吧
        # x = x_0.reshape(n, c, h * w, k, j)  # 不把窗口摊平,适合torch.mean和var，计算相似吧 = n, c, pathes, k, j
        # # print(x)
        # print("hw unfold", x.size())  # 8, 2, 2, 3, 3 N,h,w,k,j h,w是输出大小,kj是kernel大小，计算
        # p = h * w
        # x_nc = x.reshape(n * c, p, k, j)
        # print(x_nc.size())
        # x_f = x.reshape(n, c, p, k * j)
        # print("flatten", x_f.size())

        x_p = self.im2col(x)
        n, c, h, w, k, j = self.size
        p = h * w
        x_nc = x_p.reshape(n * c, p, k, j)
        # N*C P P: 2 4 4 x_f * weight1 , weight2
        # out_ps = torch.einsum('ncpl, ncdl -> ncpd', x_f, x_f).view(n * c, p, p)
        # out_ps = self.softmax(out_ps)

        # N*C P‘ H*W 2 4 9
        out_ps = self.g(x_nc)
        # 与g特征进行矩阵乘运算，[N*C, P, H * W]
        # print(x_g.size())
        # print("out_ps", out_ps.size())
        # out_ps = torch.matmul(out_ps, x_g).view(self.size).contiguous()
        # print(out_ps.size())
        out_ps = out_ps.view(self.size).contiguous()
        out_ps = self.conv2d(out_ps)
        # print(out_ps.size())
        if g_h != x_h:
            # torch.Size([1, 64, 128, 128]) 1024 32.0
            # print(x.size(), p_in, g_h)
            out_ps = NF.interpolate(out_ps, size=(x_h, x_w), mode='bilinear', align_corners=True)

        out_ps = out_ps + residual

        return out_ps

    def forward_nckjP(self, x):
        residual = x

        x = self.phi(x)

        x = self.im2col(x)
        n, c, h, w, k, j = self.size
        x_f = x.reshape(n, c, h * w, k * j)
        x_f_t = x_f.permute(0, 1, 3, 2).contiguous()

        if self.shuffle_patches:
            x_f_t = self.randperm_chnls(x_f_t, -1)

        # N*C P‘ H*W 2 4 9
        out_ps = self.g(x_f_t)  # .permute(0, 2, 1).contiguous()
        # 与g特征进行矩阵乘运算，[N*C, P, H * W]
        # print(x_g.size())
        # print("out_ps", out_ps.size())
        # out_ps = torch.matmul(out_ps, x_g).view(self.size).contiguous()
        # print(out_ps.size())
        out_ps = out_ps.view(self.size).contiguous()
        out_ps = self.conv2d(out_ps)

        out_ps = out_ps + residual

        return out_ps

    def forward_N1kjP(self, x):
        residual = x

        x = self.phi(x)

        x = self.im2col(x)
        n, c, h, w, k, j = self.size
        x_f = x.reshape(n * c, 1, h * w, k * j)
        x_f_t = x_f.permute(0, 1, 3, 2).contiguous()

        if self.shuffle_patches:
            x_f_t = self.randperm_chnls(x_f_t, -1)

        # N*C P‘ H*W 2 4 9
        out_ps = self.g(x_f_t)  # .permute(0, 2, 1).contiguous()
        # 与g特征进行矩阵乘运算，[N*C, P, H * W]
        # print(x_g.size())
        # print("out_ps", out_ps.size())
        # out_ps = torch.matmul(out_ps, x_g).view(self.size).contiguous()
        # print(out_ps.size())
        out_ps = out_ps.view(self.size).contiguous()
        out_ps = self.conv2d(out_ps)

        out_ps = out_ps + residual

        return out_ps

    def forward_nCPk_j(self, x):
        # NPh_w,对通道方向等价于做深度可分离卷积
        # 或者排序，图更大了，3x3的获得的细节更大，按相似性筛选一下，扣除这些patches再最后用全黑的补在x上，作为x的残差，降低影响
        # 或者repeat通道,但是感受野不一样，等于没有
        # 粗暴点直接插值，但会极大降低大图上的效果
        residual = x
        _, _, x_h, x_w = x.size()
        # p_in, p_out, _, _ = self.g.weight.data.size()
        # g_h = math.sqrt(p_in)
        # if g_h != x_h:
        # torch.Size([1, 64, 128, 128]) 1024 32.0
        # print(x.size(), p_in, g_h)
        # x = NF.interpolate(x, size=_pair(int(g_h)), mode='bilinear', align_corners=True)

        # x = self.pad2d(x)
        # # 8, 4, 4
        # x = x.unfold(2, self.windows_size, self.stride)
        # x_0 = x.unfold(3, self.windows_size, self.stride)
        # n, c, h, w, k, j = x_0.size()
        # print("init unfold", x_0.size())
        # # x = x.reshape(n, c, h*w, -1) #把窗口摊平了，计算相似吧
        # x = x_0.reshape(n, c, h * w, k, j)  # 不把窗口摊平,适合torch.mean和var，计算相似吧 = n, c, pathes, k, j
        # # print(x)
        # print("hw unfold", x.size())  # 8, 2, 2, 3, 3 N,h,w,k,j h,w是输出大小,kj是kernel大小，计算
        # p = h * w
        # x_nc = x.reshape(n * c, p, k, j)
        # print(x_nc.size())
        # x_f = x.reshape(n, c, p, k * j)
        # print("flatten", x_f.size())

        x = self.im2col(x)
        n, c, h, w, k, j = self.size
        p = h * w
        x_cp = x.reshape(n, c * p, k, j)
        # N*C P P: 2 4 4 x_f * weight1 , weight2
        # out_ps = torch.einsum('ncpl, ncdl -> ncpd', x_f, x_f).view(n * c, p, p)
        # out_ps = self.softmax(out_ps)

        # N*C P‘ H*W 2 4 9
        out_ps = self.g(x_cp)

        out_ps = out_ps.reshape(n, -1, h, w, k, j).contiguous()

        out_ps = rearrange(out_ps, "n c h w k j -> n c (h k) (w j) ")

        residual = self.theta(residual)

        out_ps = out_ps + residual

        return out_ps

    # mode = ["NPk_j", "nckjP", "nCPk_j", "N1kjP"]
    def forward(self, x):
        if self.mode == "NPk_j":
            return self.forward_nCPk_j(x)
        if self.mode == "nckjP":
            return self.forward_nckjP(x)
        if self.mode == "nCPk_j":
            return self.forward_nCPk_j(x)
        if self.mode == "N1kjP":
            return self.forward_N1kjP(x)
        else:
            assert False, print("BA mode not exist")


class ConvBlock(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, image_size, windows_size, stride, pad, in_channels, out_channels, mid_channels=None,
                 split_patch=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if split_patch:
            self.double_conv = nn.Sequential(
                BIMBlockND(in_channels, mid_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True,
                           False, False),
                # nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                # nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                BIMBlockND(mid_channels, out_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True,
                           False, False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                # BIMBlockND(in_channels, mid_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True, False, False),
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                # BIMBlockND(mid_channels, out_channels, image_size, windows_size, stride, pad, "nCPk_j", 1, False, True, False, False),
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

    def forward(self, x1, x2):
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
    x = torch.randn(1, 3, 48, 48).cuda()
    net = APUNet(3, 3)
    stat(net.cuda(), [(3, 48, 48)])
    out = net(x)
    print(out.shape)
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


def test_module():
    # 根据kernel_size预先计算x的im2col后的维度
    # "nckjP", "nCPk_j" 1x1一个意思，不做第二个
    mode = ["NPk_j", "nckjP", "nCPk_j", "N1kjP"]
    patches_size = 16
    stride = 1
    pad = 0
    in_channel = 8
    image_size = 32
    new_inputs = torch.randn([1, in_channel, image_size, image_size]).cuda()
    # bim = CBIMBlockND(16, 16, 3, 1, 1, True, False)
    # bim = CBIMBlockND(in_channel, 64, 3, patches_size, stride, pad, True, False).cuda()
    bim = BIMBlockND(in_channel, 16, image_size=image_size, windows_size=patches_size, stride=patches_size, pad=pad,
                     mode=mode[2],
                     loc_conn_point=1, shuffle_patches=True, is_conv=True, is_bn=False).cuda()
    out = bim(new_inputs)
    print(out.shape)

@register_model
def test_other_module():
    # x = torch.randn(1, 16 * 16, 64 * 16)  # 1 256 1024
    # net = AttentionFromPvt(64 * 16, 8)
    # out = net(x, 16, 16)
    # print(out.shape)

    # x = torch.randn(1, 16*16, 64*16)#1 256 1024
    # net = Attention(64*16, 8)
    # out = net(x)
    # print(out.shape)

    # x = torch.randn(1, 16 * 16, 64 * 16)  # 1 256 1024
    # net = PvtBlock(64*16, 8)
    # out = net(x, 16, 16)
    # print(out.shape)

    model = Pvt(
        patch_size=64, in_channels=3, out_channels=3, patch_sride=[2, 2, 2, 2],
        hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
        depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1])

    model.default_cfg = _cfg()


    return model


def partial_load_checkpoint(state_dict):
    pretrained_dict = {}
    dismatch_list = ['pos_embed', 'norm', 'patch_embed1.proj.weight']
    for module in state_dict.items():
        # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
        # 2. overwrite entries in the existing state dict
        k, v = module
        if all(m not in k for m in dismatch_list):
            # print(k)
            pretrained_dict.update({k: v})

    return pretrained_dict

if __name__ == "__main__":
    model = test_other_module()
    # test_net()
    # ckpt = partial_load_checkpoint(torch.load("../PVT/pvt_tiny.pth"))
    # model.load_state_dict(ckpt, strict=False)
    x = torch.randn(1, 3, 64, 64)
    pred = model(x)

    print(pred.shape)