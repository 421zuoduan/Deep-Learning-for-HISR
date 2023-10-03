import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from typing import Optional, List
from torch import Tensor
import numpy as np
from .self_attn_module import *

# implement tf.gather_nd() in pytorch
def gather_nd(tensor, indexes, ndim):
    '''
    inputs = torch.randn(1, 3, 5)
    base = torch.arange(3)
    X_row = base.reshape(-1, 1).repeat(1, 5)
    lookup_sorted, indexes = torch.sort(inputs, dim=2, descending=True)
    print(inputs)
    print(indexes, indexes.shape)
    # print(gathered)
    print(gather_nd(inputs, indexes, [1, 2]))
    '''
    if len(ndim) == 2:
        base = torch.arange(indexes.shape[ndim[0]])
        row_index = base.reshape(-1, 1).repeat(1, indexes.shape[ndim[1]])
        gathered = tensor[..., row_index, indexes]
    elif len(ndim) == 1:
        base = torch.arange(indexes.shape[ndim[0]])
        gathered = tensor[..., base, indexes]
    else:
        raise NotImplementedError
    return gathered


class PatchMergeModule(nn.Module):

    def __init__(self, bs_axis_merge=False):
        super().__init__()
        if bs_axis_merge:
            self.split_func = torch.split
        else:
            self.split_func = lambda x, _, dim: [x]

    def forward_chop(self, is_training, *x, shave=12, **kwargs):
        # 不存在输入张量不一样的情况, 如不一样请先处理成一样的再输入
        # 但输出存在维度不一样的情况, 因为网络有多层并且每层尺度不同, 进行分开处理: final_output, intermediate_output
        x = torch.cat(x, dim=0)
        split_func = self.split_func
        args = self.args
        # self.axes[1][0].imshow(x[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        x.cpu()
        batchsize = args.crop_batch_size
        b, c, h, w = x.size()
        padsize = int(args.patch_size)
        shave = int(args.patch_size / 2)
        # print(self.scale, self.idx_scale)
        scale = args.scale  # self.scale[self.idx_scale]

        h_cut = (h - padsize) % (int(shave / 2))
        w_cut = (w - padsize) % (int(shave / 2))

        x_unfold = F.unfold(x, padsize, stride=int(shave / 2)).permute(2, 0, 1).contiguous()

        ################################################
        # 最后一块patch单独计算
        ################################################

        x_hw_cut = x[..., (h - padsize):, (w - padsize):]
        y_hw_cut = self.forward(is_training, *[s.cuda() for s in split_func(x_hw_cut, [1, 1], dim=0)], **kwargs)
        y_hw_cut = [s.cpu() for s in y_hw_cut]

        x_h_cut = x[..., (h - padsize):, :]
        x_w_cut = x[..., :, (w - padsize):]
        y_h_cut = self.cut_h(x_h_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs)
        y_w_cut = self.cut_w(x_w_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs)
        # self.axes[0][0].imshow(y_h_cut[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # 左上patch单独计算，不是平均而是覆盖
        ################################################

        x_h_top = x[..., :padsize, :]
        x_w_top = x[..., :, :padsize]
        y_h_top = self.cut_h(x_h_top, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs)
        y_w_top = self.cut_w(x_w_top, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs)

        # self.axes[0][1].imshow(y_h_top[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
        ################################################
        # img->patch，最大计算crop_s个patch，防止bs*p*p太大
        ################################################

        x_unfold = x_unfold.view(x_unfold.size(0), -1, c, padsize, padsize)
        y_unfold = []

        x_range = x_unfold.size(0) // batchsize + (x_unfold.size(0) % batchsize != 0)
        x_unfold.cuda()
        for i in range(x_range):
            y_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...] for s in split_func(x_unfold[i * batchsize:(i + 1) * batchsize, ...], [1, 1], dim=1)]
                , **kwargs)])
            # P.data_parallel(self.model, x_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())

        for i, s in enumerate(zip(*y_unfold)):
            if i < len(y_unfold):
                y_unfold[i] = s
            else:
                y_unfold.append(s)

        out = []
        for s_unfold, s_h_top, s_w_top, s_h_cut, s_w_cut, s_hw_cut in zip(y_unfold, y_h_top, y_w_top, y_h_cut, y_w_cut, y_hw_cut):

            s_unfold = torch.cat(s_unfold, dim=0)

            y = F.fold(s_unfold.view(s_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                       ((h - h_cut) * scale, (w - w_cut) * scale), padsize * scale,
                       stride=int(shave / 2 * scale))
            # 312， 480
            # self.axes[0][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            ################################################
            # 第一块patch->y
            ################################################
            y[..., :padsize * scale, :] = s_h_top
            y[..., :, :padsize * scale] = s_w_top
            # self.axes[0][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            s_unfold = s_unfold[...,
                       int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                       int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
            # 1，3，24，24
            s_inter = F.fold(s_unfold.view(s_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                             ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                             padsize * scale - shave * scale,
                             stride=int(shave / 2 * scale))

            s_ones = torch.ones(s_inter.shape, dtype=s_inter.dtype)
            divisor = F.fold(F.unfold(s_ones, padsize * scale - shave * scale,
                                      stride=int(shave / 2 * scale)),
                             ((h - h_cut - shave) * scale, (w - w_cut - shave) * scale),
                             padsize * scale - shave * scale,
                             stride=int(shave / 2 * scale))

            s_inter = s_inter / divisor
            # self.axes[1][1].imshow(y_inter[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            ################################################
            # 第一个半patch
            ################################################
            y[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale),
            int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = s_inter
            # self.axes[1][2].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            y = torch.cat([y[..., :y.size(2) - int((padsize - h_cut) / 2 * scale), :],
                           s_h_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
            # 图分为前半和后半
            # x->y_w_cut
            # model->y_hw_cut
            y_w_cat = torch.cat([s_w_cut[..., :s_w_cut.size(2) - int((padsize - h_cut) / 2 * scale), :],
                                 s_hw_cut[..., int((padsize - h_cut) / 2 * scale + 0.5):, :]], dim=2)
            y = torch.cat([y[..., :, :y.size(3) - int((padsize - w_cut) / 2 * scale)],
                           y_w_cat[..., :, int((padsize - w_cut) / 2 * scale + 0.5):]], dim=3)
            out.append(y.cuda())
            # self.axes[1][3].imshow(y[0, ...].permute(1, 2, 0).cpu().numpy() / 255)
            # plt.show()

        return out#y.cuda()

    def cut_h(self, x_h_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs):
        split_func = self.split_func
        # b,c*k*k,n_H*n_W: 1, 30000, 3->2, 30000, 3
        x_h_cut_unfold = F.unfold(x_h_cut, padsize, stride=int(shave / 2)).permute(2, 0, 1).contiguous() # transpose(0, 2)

        x_h_cut_unfold = x_h_cut_unfold.view(x_h_cut_unfold.size(0), -1, c, padsize, padsize) # x_h_cut_unfold.size(0), -1, padsize, padsize
        x_range = x_h_cut_unfold.size(0) // batchsize + (x_h_cut_unfold.size(0) % batchsize != 0)
        y_h_cut_unfold = []
        x_h_cut_unfold.cuda()
        # TODO: [[a0, b0, c0], [a1, b1, c1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
        for i in range(x_range):
            y_h_cut_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...] for s in split_func(x_h_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                               ...], [1, 1], dim=1)], **kwargs)])  # P.data_parallel(self.model, x_h_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())

        for i, s in enumerate(zip(*y_h_cut_unfold)):
            if i < len(y_h_cut_unfold):
                y_h_cut_unfold[i] = s
            else:
                y_h_cut_unfold.append(s)

        y_h_cut = []
        for s_h_cut_unfold  in y_h_cut_unfold:

            s_h_cut_unfold = torch.cat(s_h_cut_unfold, dim=0)
            # nH*nW, c, k, k: 3, 3, 100, 100
            s_h_cut = F.fold(
                s_h_cut_unfold.view(s_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                (padsize * scale, (w - w_cut) * scale), padsize * scale, stride=int(shave / 2 * scale))
            s_h_cut_unfold = s_h_cut_unfold[..., :,
                             int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale)].contiguous()
            s_h_cut_inter = F.fold(
                s_h_cut_unfold.view(s_h_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                (padsize * scale, (w - w_cut - shave) * scale), (padsize * scale, padsize * scale - shave * scale),
                stride=int(shave / 2 * scale))

            s_ones = torch.ones(s_h_cut_inter.shape, dtype=s_h_cut_inter.dtype)
            divisor = F.fold(
                F.unfold(s_ones, (padsize * scale, padsize * scale - shave * scale),
                         stride=int(shave / 2 * scale)), (padsize * scale, (w - w_cut - shave) * scale),
                (padsize * scale, padsize * scale - shave * scale), stride=int(shave / 2 * scale))
            s_h_cut_inter = s_h_cut_inter / divisor

            s_h_cut[..., :, int(shave / 2 * scale):(w - w_cut) * scale - int(shave / 2 * scale)] = s_h_cut_inter
            y_h_cut.append(s_h_cut)

        return y_h_cut

    def cut_w(self, x_w_cut, h, w, c, h_cut, w_cut, padsize, shave, scale, batchsize, **kwargs):

        split_func = self.split_func
        x_w_cut_unfold = torch.nn.functional.unfold(x_w_cut, padsize, stride=int(shave / 2)).permute(2, 0, 1).contiguous()

        x_w_cut_unfold = x_w_cut_unfold.view(x_w_cut_unfold.size(0), -1, c, padsize, padsize)
        x_range = x_w_cut_unfold.size(0) // batchsize + (x_w_cut_unfold.size(0) % batchsize != 0)
        y_w_cut_unfold = []
        x_w_cut_unfold.cuda()

        #TODO: [[a0, b0], [a1, b1], ...] -> [[a0, a1], [b0, b1], ...] -> [cat() for s in [[a0, a1], [b0, b1], ...]]
        for i in range(x_range):
            y_w_cut_unfold.append([s.cpu() for s in self.forward(*[s[:, 0, ...]for s in split_func(x_w_cut_unfold[i * batchsize:(i + 1) * batchsize,
                                               ...], [1, 1], dim=1)], **kwargs)])  # P.data_parallel(self.model, x_w_cut_unfold[i*batchsize:(i+1)*batchsize,...], range(self.n_GPUs)).cpu())
        for i, s in enumerate(zip(*y_w_cut_unfold)):
            if i < len(y_w_cut_unfold):
                y_w_cut_unfold[i] = s
            else:
                y_w_cut_unfold.append(s)
        y_w_cut = []
        for s_w_cut_unfold in y_w_cut_unfold:

            s_w_cut_unfold = torch.cat(s_w_cut_unfold, dim=0)

            s_w_cut = torch.nn.functional.fold(
                s_w_cut_unfold.view(s_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                ((h - h_cut) * scale, padsize * scale), padsize * scale, stride=int(shave / 2 * scale))
            s_w_cut_unfold = s_w_cut_unfold[..., int(shave / 2 * scale):padsize * scale - int(shave / 2 * scale),
                             :].contiguous()
            s_w_cut_inter = torch.nn.functional.fold(
                s_w_cut_unfold.view(s_w_cut_unfold.size(0), -1, 1).transpose(0, 2).contiguous(),
                ((h - h_cut - shave) * scale, padsize * scale), (padsize * scale - shave * scale, padsize * scale),
                stride=int(shave / 2 * scale))

            s_ones = torch.ones(s_w_cut_inter.shape, dtype=s_w_cut_inter.dtype)
            divisor = torch.nn.functional.fold(
                torch.nn.functional.unfold(s_ones, (padsize * scale - shave * scale, padsize * scale),
                                           stride=int(shave / 2 * scale)), ((h - h_cut - shave) * scale, padsize * scale),
                (padsize * scale - shave * scale, padsize * scale), stride=int(shave / 2 * scale))
            s_w_cut_inter = s_w_cut_inter / divisor

            s_w_cut[..., int(shave / 2 * scale):(h - h_cut) * scale - int(shave / 2 * scale), :] = s_w_cut_inter
            y_w_cut.append(s_w_cut)

        return y_w_cut




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
                               padding=(kernel_size // 2), bias=bias))
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
            x = x.reshape(bs, p, H * W, num_patch, -1).permute(0, 1, 3, 2, 4). \
                reshape(bs, p * num_patch, H * W, -1).contiguous()

        return x, (H, W, x.shape[-1])


class PatchEmbed(nn.Module):
    """ Image to Visual Word Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, outer_dim=768, inner_dim=24,
                 inner_stride=1, num_patch=1, stride_ratio=1):
        super().__init__()
        # self.down_scale = list(down_scale)
        patch_size = _pair(patch_size)
        img_size = _pair(img_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.num_patch = 1 if num_patch is None else np.prod(
        #     [((img_size - img_size // sr) // (img_size // (sr * stride_ratio))) + 1 for sr in down_scale])
        # self.img_size = _pair(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        # self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size) # patch_size
        # self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=3, padding=1, stride=inner_stride)

    def forward(self, x):
        B, C, H, W = x.shape
        # new_h, new_w = [H // self.down_scale[0], W // self.down_scale[1]]  #
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x)  # B, Ck2, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size)  # B*N, C, 16, 16
        x = self.proj(x)  # B*N, C, 8, 8, 不是C*P^2而已，其余是一样的

        # outer_tokens: B, self.num_patches
        # inner_tokens: B*self.num_patches
        '''
        inner_tokens = self.patch_embed(x) + self.inner_pos # B*P, 8*8, C
        
        outer_tokens = self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, self.num_patches, -1))))  # B*P, 8*8*C      
        outer_tokens = torch.cat((self.cls_token.expand(B, -1, -1), outer_tokens), dim=1)
        
        if self.has_inner:
            inner_tokens = inner_tokens + self.drop_path(self.inner_attn(self.inner_norm1(inner_tokens))) # B*P, k*k, c, C->D
            inner_tokens = inner_tokens + self.drop_path(self.inner_mlp(self.inner_norm2(inner_tokens))) # B*P, k*k, c
            B, P, C = outer_tokens.size() #B, P, 9C(8*8*C)
            outer_tokens[:,1:] = outer_tokens[:,1:] + self.proj_norm2(self.proj(self.proj_norm1(inner_tokens.reshape(B, P-1, -1)))) # B, P, C
        if self.se > 0:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            tmp_ = self.outer_mlp(self.outer_norm2(outer_tokens))
            outer_tokens = outer_tokens + self.drop_path(tmp_ + self.se_layer(tmp_))
        else:
            outer_tokens = outer_tokens + self.drop_path(self.outer_attn(self.outer_norm1(outer_tokens)))
            outer_tokens = outer_tokens + self.drop_path(self.outer_mlp(self.outer_norm2(outer_tokens)))
        '''
        x = x.reshape(B, self.num_patches, self.inner_dim, *self.patch_size).permute(0, 1, 3, 4, 2)  # B, N, 8, 8, C
        return x, (*self.patch_size, self.inner_dim)

class PatchEmbedUnfified(nn.Module):
    """ Image to Visual Word Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, outer_dim=768, inner_dim=24,
                 inner_stride=1, num_patch=1, stride_ratio=1):
        super().__init__()
        # self.down_scale = list(down_scale)
        patch_size = _pair(patch_size)
        img_size = _pair(img_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        # self.num_patch = 1 if num_patch is None else np.prod(
        #     [((img_size - img_size // sr) // (img_size // (sr * stride_ratio))) + 1 for sr in down_scale])
        # self.img_size = _pair(img_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.inner_dim = inner_dim
        # self.num_words = math.ceil(patch_size[0] / inner_stride) * math.ceil(patch_size[1] / inner_stride)

        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)  # patch_size
        # self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=7, padding=3, stride=inner_stride)
        self.proj = nn.Conv2d(in_chans, inner_dim, kernel_size=3, padding=1, stride=inner_stride)
        self.linear_proj()

    def linear_proj(self):
        outer_dim = self.patch_size[0] * self.patch_size[1] * self.inner_dim
        self._proj_norm1 = nn.LayerNorm(outer_dim)
        self._proj = nn.Linear(outer_dim,  outer_dim)
        self._proj_norm2 = nn.LayerNorm(outer_dim)

    def conv_forward_features(self, x, bs, p):
        x = self.proj(x)  # B*N, C, 8, 8, 不是C*P^2而已，其余是一样的
        x = x.reshape(bs, p * self.num_patches, self.inner_dim, -1).permute(0, 1, 3, 2)  # B, N, 8, 8, C
        return x

    def linear_forward_features(self, x, bs, p):
        #B*N, k*k, D -> B, N, 8*8*C, 8*8*C-> IPT
        x = self._proj_norm2(
            self._proj(self._proj_norm1(x.reshape(bs, p * self.num_patches, -1))))
        x = x.reshape(bs, p * self.num_patches, self.inner_dim, -1).permute(0, 1, 3, 2)  # B, N, 8, 8, C
        return x


    def forward(self, x, bs):
        B, C, H, W = x.shape
        p = B // bs
        # new_h, new_w = [H // self.down_scale[0], W // self.down_scale[1]]  #
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.unfold(x)  # B, Ck2, N
        x = x.transpose(1, 2).reshape(B * self.num_patches, C, *self.patch_size)  # B*N, C, 16, 16
        inner_tokens = self.conv_forward_features(x, bs, p)
        outer_tokens = self.linear_forward_features(inner_tokens, bs, p)


        return inner_tokens, outer_tokens, (*self.patch_size, self.inner_dim)

# 与HirePatch的区别是Linear，P^2C -> D
class PatchSplitting(nn.Module):
    def __init__(self, down_scale, in_chans, embed_dim):
        super().__init__()
        # self.downscaling_factor = downscaling_factor
        self.down_scale = list(down_scale)
        self.patch_merge = nn.Unfold(kernel_size=down_scale[0], stride=down_scale[0], padding=0)
        self.linear = nn.Linear(in_chans * down_scale[0] ** 2, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, x, bs):
        b, c, h, w = x.shape
        new_h, new_w = h // self.downscaling_factor, w // self.downscaling_factor
        x = self.patch_merge(x).view(b, -1, new_h * new_w).transpose(2, 1)
        # unsqueeze 与HirePatch的维度对齐，传统的需要压缩patch维度，所以这里p=1
        x = self.linear(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(1)
        return x, (new_h, new_w, self.C)

class HirePatch(nn.Module):
    def __init__(self, down_scale, in_chans, embed_dim, patch_stride=1):
        super().__init__()
        self.embed_dim = embed_dim
        self.down_scale = list(down_scale)
        # self.patch_merge = nn.Unfold(kernel_size=int(self.downscaling_factor[0]), stride=int(self.downscaling_factor[0]), padding=0)
        self.linear = nn.Linear(in_chans, embed_dim)


    def forward(self, x, bs, axes=[2, 3]):  # h, w
        b, c, h, w = x.shape
        # b, c, *hw = x.shape
        p = b // bs
        # shape = [hw[axis] // self.downscaling_factor[axis] for axis in axes]
        new_h, new_w = shape = [h // self.down_scale[0], w // self.down_scale[1]]  #
        # x = self.patch_merge(x).view(self.bs, c, p * self.downscaling_factor*2, new_h, new_w).permute(0, 2, 3, 4, 1)
        # x = x.reshape(self.bs, -1, new_h * new_w, c)
        # x = x.reshape(self.bs, p, -1, c)
        if 1 in self.down_scale:
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
        # reshape不把P^2合并到C中
        x = x.reshape(bs, x.shape[1], -1, c)
        x = self.linear(x)
        return x, (*shape, self.embed_dim)  # (b,p,n_h,n_w,c)

class MHSPatch(nn.Module):
    '''
    HireSharePatch的堵patch版本
    Step1:
    不进行patch的压缩
    patch_size = 32
    128-32 / 16 + 1 = 7 * 7 = 49
    1,128,128,3-> 1, 16, 32, 32, C
    Step2:
    输入有128, 64, 32尺度时候
    patch_num = [49, 9, 1]
    outout: [1,[49, 9, 1], 32, 32, C] -> [1, 32, 32, 49*C1+9*C2+1*C3] -> [1, 32, 32, C * K] #这只是通道方向对空间的保留
    example: san_cs.py
    '''
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
            Warning(
                "HPatchConvEmbed will get cross-scale non-local patch, equal to HirePatch, but share the weight among patchs")

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
            p = b // bs  # p=1

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
                x = x.view(bs, -1, N).permute(0, 2, 1)  # .permute(0, 2, 3, 1)
            else:
                x = x.unfold(axes[0], new_h, new_h // 2).unfold(axes[1], new_w, new_w // 2). \
                    reshape(b, -1, N).permute(0, 2, 1)  # .permute(0, 2, 3, 1)
            patch_list.append(x)
        # b,n_h*n_w,CP
        patch_list = torch.cat(patch_list, dim=-1)  # M*C*P

        # 得到多尺度的K, 1,1024,449 x 449*5120
        patch_list = self.linear(patch_list)
        if compatiable:
            patch_list = patch_list.reshape(bs, N, num_patch, -1).permute(0, 2, 1, 3)
        # 1,16,1024,320 聚合patch
        return patch_list, (*patch_size, self.embed_dim)  # (b,p,n_h*n_w,c)

class HireSharePatch(nn.Module):
    '''
    不进行patch的压缩, 但reshape把P^2合并到C中了，只是输出也扩大了,依然是通道方向的
    patch_size = 32
    128-32 / 16 + 1 = 7 * 7 = 49
    1,128,128,3-> 1, 16, 32, 32, C->1,32,32,16*C X 16*C -> 1,16,32,32,C
    example: san_cs.py
    '''
    def __init__(self, img_size, down_scale, in_chans, embed_dim, num_patch=None, stride_ratio=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.down_scale = list(down_scale)
        self.stride_ratio = stride_ratio
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
        stride_ratio = self.stride_ratio
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
                    x = x.unfold(axis, half, half // stride_ratio).transpose(-1, -2)
                elif axis == 3:
                    x = x.unfold(axis, half, half // stride_ratio).transpose(3, 2)
                x = x.reshape(bs * factor * p, c, x.shape[-2], x.shape[-1])
            x = x.view(bs, c, -1, *shape).permute(0, 2, 3, 4, 1)
        else:
            # x = self.patch_merge(x).view(self.bs, c, -1, new_h, new_w).permute(0, 2, 3, 4, 1)
            x = x.unfold(axes[0], new_h, new_h // stride_ratio).unfold(axes[1], new_w, new_w // stride_ratio). \
                reshape(b, -1, new_h * new_w).transpose(2, 1).contiguous()
        x = self.linear(x)
        # p
        x = x.reshape(bs, N, self.num_patch * p, -1).permute(0, 2, 1, 3).reshape(bs, self.num_patch * p, N, -1).contiguous()
        return x, (*shape, self.embed_dim)  # (b,p,n_h,n_w,c)

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
            # ])
            self.conv = PrimalConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = PrimalConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, bs):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class AggregateBlock(nn.Module):
    '''
    还只能是同一尺寸的，不同尺寸是MHSPatch
    将不同PatchGen方案的patch通道concat
    '''
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 逐层回放
        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.conv = PrimalConvBlock(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = PrimalConvBlock(in_channels, out_channels)

    def forward(self, x1, x2, bs): #bs*p, C, H, W
        # x1 = self.up(x1)
        B, C, H, W = x2.shape
        x2 = x2.reshape(bs, -1, C, H, W)
        x1 = x1.reshape(bs, -1, x1.shape[1], H, W)
        x = torch.cat([x2, x1], dim=2).reshape(B, -1, H, W)
        return self.conv(x)

class IPTDecoderLayer(nn.Module):
    def __init__(self, dim, hidden_dim, patch_dim=3, num_heads=8, norm_layer=nn.LayerNorm, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.,
                 mlp_ratio=1, sr_ratio=1, num_patch=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.attn = PSABlock(hidden_dim, patch_dim, num_heads, norm_layer, qkv_bias)
        self.conv = PrimalConvBlock(dim, hidden_dim, dim // 2)

    def forward(self, x_s, x_h):
        x = self.up(x_s)
        x = torch.cat([x, x_h],  dim=1)
        x = self.conv(x)
        _, D, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, 1, H*W, D)
        x = self.attn(x, H, W)
        x = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        return x

class patchMergingWithAttn(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, dim, norm_dim, hidden_dim, patch_dim=3, num_heads=8, norm_layer=nn.LayerNorm, qkv_bias=False, mlp_ratio=1, sr_ratio=1):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = PrimalConvBlock(dim, hidden_dim, dim // 2)
        self.attn = HBlock(1, norm_dim, hidden_dim, num_heads, mlp_ratio, norm_layer=norm_layer, qkv_bias=qkv_bias)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        _, D, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(-1, 1, H * W, D)
        x = self.attn(x, H, W)
        x = x.reshape(-1, H, W, D).permute(0, 3, 1, 2).contiguous()

        return x

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)