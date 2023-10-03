import math
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import copy
from module import *

def computePatchNumAndSize(image_size, down_scale):
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

    return N, num_patches1, num_patches2, num_patches3, num_patches4, \
           patch_size1, patch_size2, patch_size3, patch_size4

class DFTL(nn.Module):
    def __init__(self, args, image_size, down_scale, patch_stride, in_channels, out_channels,
                 hidden_dim, num_heads, mlp_ratios, depths, sr_ratio,
                 norm_pre=False, bilinear=True, compatiable=True, mode="one"):
        super(DFTL, self).__init__()

        self.args = args
        self.mode = mode

        N, num_patches1, num_patches2, num_patches3, \
        num_patches4, patch_size1, patch_size2, \
        patch_size3, patch_size4 = computePatchNumAndSize(image_size, down_scale)

        self.head = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim[0], kernel_size=3, padding=1),
                ResBlock(hidden_dim[0], kernel_size=5),
                ResBlock(hidden_dim[0], kernel_size=5)
            ) for _ in range(1)
        ])

        # self.patch_embed1 = PatchSplitting(down_scale=down_scale[0], in_chans=hidden_dim[0], embed_dim=hidden_dim[1])
        # self.patch_embed1 = PatchSplitting(down_scale=down_scale[1], in_chans=hidden_dim[1], embed_dim=hidden_dim[2])
        # self.patch_embed1 = PatchSplitting(down_scale=down_scale[2], in_chans=hidden_dim[2], embed_dim=hidden_dim[3])
        # self.patch_embed1 = PatchSplitting(down_scale=down_scale[3], in_chans=hidden_dim[3], embed_dim=hidden_dim[4])

        self.patch_embed1 = PatchConvEmbed(img_size=image_size, patch_stride=patch_stride[0], in_chans=hidden_dim[0],
                                           embed_dim=hidden_dim[1], compatiable=True)
        self.patch_embed2 = PatchConvEmbed(img_size=image_size // 2, patch_stride=patch_stride[1],
                                           in_chans=hidden_dim[1],
                                           embed_dim=hidden_dim[2], compatiable=True)
        self.patch_embed3 = PatchConvEmbed(img_size=image_size // 4, patch_stride=patch_stride[2],
                                           in_chans=hidden_dim[2],
                                           embed_dim=hidden_dim[3], compatiable=True)
        self.patch_embed4 = PatchConvEmbed(img_size=image_size // 8, patch_stride=patch_stride[3],
                                           in_chans=hidden_dim[3],
                                           embed_dim=hidden_dim[4], compatiable=True)

        # self.patch_hire1 = HirePatch(down_scale=down_scale[0],
        #                              in_chans=hidden_dim[0], embed_dim=hidden_dim[1])
        # self.patch_hire2 = HirePatch(down_scale=down_scale[1],
        #                              in_chans=hidden_dim[1], embed_dim=hidden_dim[2])
        # self.patch_hire3 = HirePatch(down_scale=down_scale[2],
        #                              in_chans=hidden_dim[2], embed_dim=hidden_dim[3])
        # self.patch_hire4 = HirePatch(down_scale=down_scale[3],
        #                              in_chans=hidden_dim[3], embed_dim=hidden_dim[4])

        # self.patch_pool = MHSPatch(img_size=[[patch_size1, patch_size1], [patch_size2, patch_size2],
        #                                      [patch_size3, patch_size3]], patch_size=[patch_size3, patch_size3],
        #                            in_chans=[hidden_dim[1], hidden_dim[2], hidden_dim[3]],
        #                            embed_dim=320, num_patch=True, stride_ratio=2)

        # 1500MB
        # self.down0 = nn.ModuleList([HBlock(down_scale=1, norm_dim=[1, num_patches1, hidden_dim[0]],
        #     dim=hidden_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     sr_ratio=sr_ratio[0])
        #     for i in range(depths[0])])

        self.down1 = nn.ModuleList([HBlock(down_scale=1, norm_dim=[1, num_patches1, hidden_dim[1]],
            dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[0])
            for i in range(depths[0])])

        self.down2 = nn.ModuleList([HBlock(down_scale=1, norm_dim=[1, num_patches2, hidden_dim[2]],
            dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[1])
            for i in range(depths[1])])

        self.down3 = nn.ModuleList([HBlock(down_scale=1, norm_dim=[1, num_patches3, hidden_dim[3]],#[k ** 3 for k in patch_num],
            dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[2])
            for i in range(depths[2])])

        self.down4 = nn.ModuleList([HBlock(down_scale=1, norm_dim=[1, num_patches4, hidden_dim[4]],#[k ** 4 for k in patch_num],
            dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
            sr_ratio=sr_ratio[3])
            for i in range(depths[3])])

        # self.down1 = nn.ModuleList([TwoHPBlock(down_scale=1, norm_dim=[1, num_patches1, hidden_dim[1]],
        #     dim=hidden_dim[1], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     sr_ratio=sr_ratio[0])
        #     for i in range(depths[0])])
        #
        # self.down2 = nn.ModuleList([TwoHPBlock(down_scale=1, norm_dim=[1, num_patches2, hidden_dim[2]],
        #     dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     sr_ratio=sr_ratio[1])
        #     for i in range(depths[1])])
        #
        # self.down3 = nn.ModuleList([TwoHPBlock(down_scale=1, norm_dim=[1, num_patches3, hidden_dim[3]],#[k ** 3 for k in patch_num],
        #     dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     sr_ratio=sr_ratio[2])
        #     for i in range(depths[2])])
        #
        # self.down4 = nn.ModuleList([TwoHPBlock(down_scale=1, norm_dim=[1, num_patches4, hidden_dim[4]],#[k ** 4 for k in patch_num],
        #     dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #     sr_ratio=sr_ratio[3])
        #     for i in range(depths[3])])



        # PSA 3x3 IPT

        # self.down1 = nn.ModuleList(
        #     [PSABlock(dim=hidden_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None,
        #              norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #              sr_ratio=sr_ratio[1])
        #      for i in range(depths[1])])
        #
        # self.down2 = nn.ModuleList(
        #     [PSABlock(dim=hidden_dim[2], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True, qk_scale=None,
        #              norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #              sr_ratio=sr_ratio[1])
        #      for i in range(depths[1])])
        # self.down3 = nn.ModuleList([PSABlock(dim=hidden_dim[3], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
        #                                     qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                     sr_ratio=sr_ratio[2])
        #                             for i in range(depths[2])])
        # factor = 2 if bilinear else 1
        # self.down4 = nn.ModuleList([PSABlock(dim=hidden_dim[4], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
        #                                     qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                                     sr_ratio=sr_ratio[3])
        #                             for i in range(depths[3])])

        self.norm = nn.LayerNorm([hidden_dim[1], image_size, image_size], eps=1e-6)

        factor = 2 if bilinear else 1
        self.up1 = PrimalUpBlock(hidden_dim[3] + hidden_dim[4], hidden_dim[3] // factor,
                                 bilinear)  # AggregateBlock(batch_size, hidden_dim[3] + hidden_dim[4], 512 // factor, bilinear)
        self.up2 = PrimalUpBlock(hidden_dim[3] // factor + hidden_dim[2], hidden_dim[2] // factor,
                                 bilinear)  # AggregateBlock(batch_size, 512 // factor + hidden_dim[2], 256 // factor, bilinear)#512
        self.up3 = PrimalUpBlock(hidden_dim[2] // factor + hidden_dim[1], hidden_dim[1] // factor, bilinear)
        self.up4 = PrimalUpBlock(hidden_dim[1] // factor + hidden_dim[0], hidden_dim[0], bilinear)

        # decode
        # self.up1 = patchMergingWithAttn(dim=hidden_dim[4] + hidden_dim[3], norm_dim=[1, num_patches3, hidden_dim[3]], hidden_dim=hidden_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=sr_ratio[3])
        # self.up2 = patchMergingWithAttn(dim=hidden_dim[3] + hidden_dim[2], norm_dim=[1, num_patches2, hidden_dim[2]], hidden_dim=hidden_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=sr_ratio[3])
        # self.up3 = patchMergingWithAttn(dim=hidden_dim[2] + hidden_dim[1], norm_dim=[1, num_patches1, hidden_dim[1]], hidden_dim=hidden_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=sr_ratio[3])
        # self.up4 = patchMergingWithAttn(dim=hidden_dim[1] + hidden_dim[0], norm_dim=[1, N, hidden_dim[0]], hidden_dim=hidden_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=sr_ratio[3])

        #PSA
        # self.up1 = IPTDecoderLayer(dim=hidden_dim[4] + hidden_dim[3], hidden_dim=hidden_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios, qkv_bias=True,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6), sr_ratio=sr_ratio[3])
        # self.up2 = IPTDecoderLayer(dim=hidden_dim[3] + hidden_dim[2], hidden_dim=hidden_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
        #                         qkv_bias=True, qk_scale=None, norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                         sr_ratio=sr_ratio[2])
        # self.up3 = IPTDecoderLayer(dim=hidden_dim[2] + hidden_dim[1], hidden_dim=hidden_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=True,
        #                         qk_scale=None,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                         sr_ratio=sr_ratio[1])
        # self.up4 = IPTDecoderLayer(dim=hidden_dim[1] + hidden_dim[0], hidden_dim=hidden_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=True,
        #                         qk_scale=None,
        #                         norm_layer=partial(nn.LayerNorm, eps=1e-6),
        #                         sr_ratio=sr_ratio[0])

        self.tail = OutConv(hidden_dim[0], out_channels)



    def _init_weights(self, m):
        # if isinstance(m, nn.Linear):
        #     nn.init.trunc_normal_(m.weight, std=.02)
        #     if isinstance(m, nn.Linear) and m.bias is not None:
        #         nn.init.constant_(m.bias, 0.)
        # elif isinstance(m, nn.LayerNorm):
        #     nn.init.constant_(m.bias, 0.)
        #     nn.init.constant_(m.weight, 1.0)
        if isinstance(m, nn.Conv2d):
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

    def forward_post(self, I):
        # bs,128,128,3->bs,128,128,32->patch2win3x3: [(128-3)/3+1]=42个
        # 42*42,bs, 9*32 = 1764，bs，288，让1764个patch共享9 * n_feat个权重参数
        # 类比卷积，卷积是1个feat共享，也就是1764个win共享9个参数，有n_feat个
        # 总参数 9 * n_feat，但每个feat的patch只共享了9个
        # 他们都是全patch查询，克服ViT的patch聚合带来的信息损失
        #
        # 更好地结合了空间和谱，TODO：把这个用在卷积上

        bs, C, H, W = I.shape
        # x = self.pa(x)

        # N,C,H,W
        x0 = I
        for blk in self.head:
            x0 = blk(x0)
        # print(x0.shape)
        # x0 = x0.reshape(bs, 1, -1, H*W).permute(0, 1, 3, 2).contiguous()
        # for blk in self.down0:
        #     x0 = blk(x0, H, W)
        # x0 = x0.reshape(bs, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # stage 1 H,W // 2, 4*C=128->64, x.shape: [B, P, N, C]
        x, (H, W, D) = self.patch_embed1(x0, bs)  # x
        # x, (H, W, D) = self.patch_hire1(x0, bs)
        # x = x + self.pos_embed1
        for blk in self.down1:
            x = blk(x, H, W)
        # [B, P, N, C]->[BP, H, W, C]，用于patch_embed
        x1 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # stage 2 -128
        x, (H, W, D) = self.patch_embed2(x1, bs)
        # x, (H, W, D) = self.patch_hire2(x1, bs)
        # x = x + self.pos_embed2
        for blk in self.down2:
            x = blk(x, H, W)
        x2 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # stage 3 -320
        x, (H, W, D) = self.patch_embed3(x2, bs)
        # x, (H, W, D) = self.patch_hire3(x2, bs)
        # x = x + self.pos_embed3
        for blk in self.down3:
            x = blk(x, H, W)
        x3 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        # stage 4 -512
        x, (H, W, D) = self.patch_embed4(x3, bs)
        # x, (H, W, D) = self.patch_hire4(x3, bs)
        # x = x + self.pos_embed4
        for blk in self.down4:
            x = blk(x, H, W)
        x4 = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        x = self.up1(x4, x3, bs)
        x = self.up2(x, x2, bs)
        x = self.up3(x, x1, bs)
        x = self.up4(x, x0, bs)

        logits = self.tail(x) + I

        return logits

    def forward_post_twostream(self, I):
        # bs,128,128,3->bs,128,128,32->patch2win3x3: [(128-3)/3+1]=42个
        # 42*42,bs, 9*32 = 1764，bs，288，让1764个patch共享9 * n_feat个权重参数
        # 类比卷积，卷积是1个feat共享，也就是1764个win共享9个参数，有n_feat个
        # 总参数 9 * n_feat，但每个feat的patch只共享了9个
        # 他们都是全patch查询，克服ViT的patch聚合带来的信息损失
        #
        # 更好地结合了空间和谱，TODO：把这个用在卷积上

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

        x = self.up1(x4, x3, bs)
        x = self.up2(x, x2, bs)
        x = self.up3(x, x1, bs)
        x = self.up4(x, x0, bs)

        logits = self.tail(x) + I

        return logits

    def forward(self, x):
        if self.mode == "one":
            return self.forward_post(x)
        elif self.mode == "two":
            return self.forward_post_twostream(x)

    def forward_chop(self, x, shave=12):
        args = self.args
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        x.cpu()
        batchsize = self.args.crop_batch_size
        h, w = x.size()[-2:]
        padsize = int(args.patch_size)
        shave = int(args.patch_size / 2)
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





def test_net(mode="one"):
    from torchstat import stat
    x = torch.randn(1, 3, 48, 48).cuda()
    scale = 2
    net = DFTL(None, image_size=48, in_channels=3, out_channels=3,
              down_scale=[[scale, scale], [scale, scale], [scale, scale], [scale, scale]],
              patch_stride=[scale, scale, scale, scale],
              hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
              depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1], mode=mode).cuda()#14,952,211

    out = net(x)
    stat(net, [x.size()[1:]])
    print(out.shape)



def test_module():
    x = torch.randn(1, 3, 128, 128).cuda()
    x1 = torch.randn(1, 32, 64, 64).cuda()
    x2 = torch.randn(1, 64, 32, 32).cuda()
    # module = HireSharePatch(img_size=128, down_scale=[2, 2], in_chans=3, embed_dim=32, num_patch=True,
    #                         stride_ratio=2).cuda()
    # module = MHSPatch(img_size=[[128, 128], [64, 64], [32, 32]], patch_size=[32, 32],
    #                   in_chans=[3, 32, 64], embed_dim=320, num_patch=True, stride_ratio=2).cuda()
    # print(module([x, x1, x2], 1, compatiable=True)[0].shape)

    module = BSABlock(dim=32, patch_dim=3, num_heads=8, qkv_bias=True).cuda()
    print(module(torch.randn(2, 1, 48 * 48, 32).cuda(), 48, 48).shape)
    #
    # module = PSABlock(dim=32, patch_dim=3, num_heads=8, qkv_bias=True).cuda()
    # print(module(torch.randn(2, 1, 48 * 48, 32).cuda(), 48, 48).shape)

if __name__ == "__main__":
    # test_module()
    test_net()