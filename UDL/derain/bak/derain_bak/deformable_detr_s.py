# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
import copy
import os
import shutil
import time
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from detr.models.transformer import build_transformer
from util.misc import is_main_process
from position_encoding import build_position_encoding
from utils.utils import MetricLogger, SmoothedValue

# from pytorch_msssim.pytorch_msssim import SSIM
from cal_ssim import SSIM
from dataset import PSNR
from utils.logger import create_logger, log_string
import imageio
import datetime

class PSNR_ycbcr(nn.Module):

    def __init__(self):
        super().__init__()
        self.gray_coeffs = torch.tensor([65.738, 129.057, 25.064],
                                        requires_grad=False).reshape((1, 3, 1, 1)) / 256
    def quantize(self, img, rgb_range):
        """metrics"""
        pixel_range = 255 / rgb_range
        img = torch.multiply(img, pixel_range)
        img = torch.clip(img, 0, 255)
        img = torch.round(img) / pixel_range
        return img

    @torch.no_grad()
    def forward(self, sr, hr, scale, rgb_range):
        """metrics"""
        sr = self.quantize(sr, rgb_range)
        gray_coeffs = self.gray_coeffs.to(sr.device)

        hr = hr.float()
        sr = sr.float()
        diff = (sr - hr) / rgb_range

        diff = torch.multiply(diff, gray_coeffs).sum(1)
        if hr.size == 1:
            return 0
        if scale != 1:
            shave = scale
        else:
            shave = scale + 6
        if scale == 1:
            valid = diff
        else:
            valid = diff[..., shave:-shave, shave:-shave]
        mse = torch.mean(torch.pow(valid, 2))
        return -10 * torch.log10(mse)

def sub_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x / 255.0

def add_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x

class BCMSLoss(torch.nn.Module):

    def __init__(self, reduction='none'):
        super(BCMSLoss, self).__init__()

        self.reduction = reduction
        self.l1_loss = torch.nn.L1Loss(reduction=reduction)
        self.l2_loss = torch.nn.MSELoss(reduction=reduction)
        self._ssim = SSIM(size_average=False, data_range=1)
        self.ind = 1

    def forward(self, x, gt):

        ind = self.ind
        # 好1%
        bce = self.bce_mse(x, gt)
        l1_loss = self.l1_loss(x, gt)
        l2_loss = self.l2_loss(x, gt)
        _ssim = self._ssim(x, gt)
        _ssim_loss = 1 - _ssim

        with torch.no_grad():
            w_1 = torch.mean(l1_loss)

            pred = torch.clamp(x, 0, 1)
            l2 = self.l2_loss(pred, gt)
            w_2 = torch.mean(l2)

            w_s = torch.mean(_ssim_loss)
            ssim_m = torch.mean(_ssim)
            w_bce = torch.mean(bce)
        # loss = _ssim_loss * \
        #        ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** (1 / ind)  # 119
        # loss = l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + 1e-8) ** ind + \
        # loss = _ssim_loss.reshape(-1, 1, 1, 1) * ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind  # 让l2朝更多方向解析，却增强l1 82
        loss = _ssim_loss.reshape(-1, 1, 1, 1) * (
                (l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind + \
               l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + (
                bce / w_bce) ** ind + 1e-8) ** ind
        # x = torch.sigmoid(x)
        # gt = torch.sigmoid(gt)
        # loss = self.l1(x, gt)
        loss = torch.mean(loss)
        return {'Loss': loss, 'l1_loss': w_1, 'mse_loss': w_2,
                'ssim_loss': w_s, 'bce_loss': w_bce, 'ssim': ssim_m}

    def bce_mse(self, x, gt):
        a = torch.exp(gt)
        b = 1
        loss = a * x - (a + b) * torch.log(torch.exp(x) + 1)
        if self.reduction == 'none':
            return -loss
        if self.reduction == 'mean':
            return -torch.mean(loss)
        else:
            assert False, "there have no self.reduction choices"


#############################################################################
# Backbone
#############################################################################

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            # return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
            return_layers = {"layer2": "0", "layer3": "1", "layer4": "2"}
            self.strides = [8, 16, 32]
            self.num_channels = [512, 1024, 2048]
        else:
            return_layers = {'layer4': "0"}
            self.strides = [32]
            self.num_channels = [2048]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)

    def forward(self, tensor_list):
        xs = self.body(tensor_list)
        out: Dict[str, List] = {}
        for name, x in xs.items():
            # m = tensor_list.mask
            # assert m is not None
            # mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = [x]  # [x, mask]
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""

    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        #num_channels = [512] if name in ('resnet18', 'resnet34') else [2048]
        assert name not in ('resnet18', 'resnet34'), "number of channels are hard coded"
        super().__init__(backbone, train_backbone, return_interm_layers)
        if dilation:
            self.strides[-1] = self.strides[-1] // 2

class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list):
        # self[0] backbone的return
        # self[1] position_embedding的return
        xs = self[0](tensor_list)
        # out: List[NestedTensor] = []
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x[0])
            # position encoding
            if self[1] != None:
                pos.append(self[1](x[0]).to(x[0].dtype))

        return out, pos


def build_backbone(args):
    position_embedding = None
    if args.position_embedding is not None:
        position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks  # Train segmentation head if the flag is provided
    backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


#############################################################################
# detr
#############################################################################
from typing import Optional, List
from torch import Tensor
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision

if float(torchvision.__version__[:3]) < 0.7:
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    Equivalent to nn.functional.interpolate, but with support for empty batch sizes.
    This will eventually be supported natively by PyTorch, and this
    class can go away.
    """
    if float(torchvision.__version__[:3]) < 0.7:
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


# Transformer Structure
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, enc_output,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(output, enc_output, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output#.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):

    def __init__(self, in_channels, nhead, hidden_dim, attn_drop=0., drop=0.,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_channels, nhead, dropout=attn_drop)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(drop)
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        #
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        # self.dropout1 = nn.Dropout(dropout)
        # self.dropout2 = nn.Dropout(dropout)

        # self.activation = _get_activation_fn(activation)

        self.mlp = Mlp(in_features=in_channels, hidden_features=hidden_dim, out_features=in_channels,
                       activation=activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout(src2)
        src = self.norm1(src)
        # C == d_model
        # HW,B,C x d_model,dim_feedforward -> HW,dim_feedforward, d_model 全连接操作 矩阵乘法
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src2 = self.mlp(src)
        src = src + src2
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, in_channels, nhead, hidden_dim, attn_drop=0., drop=0.,
                 activation="relu", normalize_before=False):
        super().__init__()

        # 关键函数nn.MultiheadAttention
        # embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None
        # query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None
        self.self_attn = nn.MultiheadAttention(in_channels, nhead, dropout=attn_drop)
        self.multihead_attn = nn.MultiheadAttention(in_channels, nhead, dropout=attn_drop)
        # Implementation of Feedforward model
        # self.linear1 = nn.Linear(in_channels, hidden_dim)
        self.dropout = nn.Dropout(drop)
        # self.linear2 = nn.Linear(hidden_dim, in_channels)
        self.norm1 = nn.LayerNorm(in_channels)
        self.norm2 = nn.LayerNorm(in_channels)
        self.norm3 = nn.LayerNorm(in_channels)
        # self.dropout1 = nn.Dropout(proj_drop)
        # self.dropout2 = nn.Dropout(proj_drop)
        # self.dropout3 = nn.Dropout(proj_drop)
        # self.activation = _get_activation_fn(activation)
        self.mlp = Mlp(in_features=in_channels, hidden_features=hidden_dim, out_features=in_channels,
                       activation=activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        '''

        self.query_embed = nn.Embedding(num_queries, hidden_dim)
            -> query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        -> tgt = torch.zeros_like(query_embed)
        -> decoder:
           -> query_pos = query_embed
           -> 计算object queries的attention的q,k

        query,key的输入是object queries: query_pos + Decoder的输入(tgt),shape都是(num_queries,b,C)
        value的输入是Decoder的输入(tgt),shape = (100,b,C) C==d_model
        '''
        q = k = self.with_pos_embed(tgt, query_pos)
        attn = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(attn)
        tgt = self.norm1(tgt)

        # query的输入是上一个attention的输出(tgt) + object queries(query_pos)
        # key的输入是Encoder的位置编码(pos) + Encoder的输出(memory)
        # value的输入是Encoder的输出(memory)

        # K.T@Q@V = (1,64,128)@(16,64,128)->(1,16)@(16,64,128) -> (1,64,128)
        #           (num_queries, b, hidden_dim)@(HW,b,hidden_dim)@(HW,b,hidden_dim)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        # tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt2 = self.mlp(tgt)
        tgt = tgt + tgt2
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class TrBlock(nn.Module):
    def __init__(self, in_channels, nhead, num_encoder_layers,  # 6, 6
                 num_decoder_layers, hidden_dim, attn_drop=0., drop=0., activation='relu', decode=True,
                 normalize_before=False, return_intermediate_dec=False):

        super(TrBlock, self).__init__()

        encoder_layer = TransformerEncoderLayer(in_channels, nhead, hidden_dim,
                                                attn_drop, drop, activation, normalize_before)
        encoder_norm = nn.LayerNorm(in_channels) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        if decode:
            decoder_layer = TransformerDecoderLayer(in_channels, nhead, hidden_dim,
                                                    attn_drop, drop, activation, normalize_before)
            decoder_norm = nn.LayerNorm(in_channels)
            self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                              return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.decode = decode
        self.in_channels = in_channels
        # self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    # TODO: query_embed, decoder
    def forward(self, src, mask, query_embed, pos_embed=None):
        '''
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        -> tgt = torch.zeros_like(query_embed)
        -> decoder
       '''
        # flatten NxCxHxW to HWxBxC，不同于PVT是BxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        if pos_embed is not None:
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
        if mask is not None:
            mask = mask.flatten(1)

        tgt = torch.zeros_like(query_embed)
        enc_output = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        if self.decode:
            dec_output = self.decoder(tgt, enc_output, memory_key_padding_mask=mask,
                                      pos=pos_embed, query_pos=query_embed)
        else:
            dec_output = enc_output

        return dec_output.permute(1, 2, 0).view(bs, c, h, w), \
               enc_output.permute(1, 2, 0).view(bs, c, h, w)  # .transpose(1, 2)


class DeformableDETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, backbone, transformer, num_queries, num_feature_levels,
                 aux_loss=True, two_stage=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.in_channels

        self.num_feature_levels = num_feature_levels
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        # IPT-based,采用(N_t, P^2 * C)作为embeded
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        # 压缩fpn输出特征的维度
        #self.input_proj = nn.Conv2d(backbone.num_channels, hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss
        self.two_stage = two_stage

        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
        if two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
    def forward(self, samples):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """

        # features, pos = self.backbone(samples)
        #
        # # src, mask = features[-1].decompose()
        # src = features[-1][0]
        # mask = None
        # # assert mask is not None
        # hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        ...


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        # self.num_classes = num_classes
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.bcmsl = BCMSLoss().cuda()
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer('empty_weight', empty_weight)

    def loss_masks(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        # src_masks = src_masks[src_idx]
        src_masks = outputs["pred_masks"]
        # # masks = [t["masks"] for t in targets]
        target_masks = targets
        # # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = masks
        target_masks = target_masks.to(src_masks)

        # upsample predictions to the target size
        # src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        # src_masks = src_masks[:, 0].flatten(1)

        # src_masks = x + interpolate(src_masks, size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)

        losses = {
            "l1_loss": F.l1_loss(src_masks, target_masks, reduction='mean'),
            # "mse_loss": F.mse_loss(src_masks, target_masks, reduction='mean'),
        }
        return losses

    def loss_ours(self, outputs, targets):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        # src_masks = src_masks[src_idx]
        src_masks = outputs["pred_masks"]
        # # masks = [t["masks"] for t in targets]
        target_masks = targets
        # # TODO use valid to mask invalid areas due to padding in loss
        # target_masks, valid = masks
        target_masks = target_masks.to(src_masks)

        # upsample predictions to the target size
        # src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
        #                         mode="bilinear", align_corners=False)
        # src_masks = src_masks[:, 0].flatten(1)

        src_masks = interpolate(src_masks, size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)

        loss_dicts = self.bcmsl(src_masks, target_masks)
        # keys = {'Loss', 'l1_loss', 'mse_loss', 'ssim_loss', 'bce_loss', 'ssim'}
        # for idx, k in enumerate(keys):
        #     loss_dicts[k] = losses[idx]

        return loss_dicts

    def get_loss(self, loss, outputs, targets):
        loss_map = {
            # 'labels': self.loss_labels,
            # 'cardinality': self.loss_cardinality,
            # 'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'ours': self.loss_ours
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        # losses = {}
        for loss in self.losses:
            # losses.update(self.get_loss(loss, outputs, targets))
            losses = self.get_loss(loss, outputs, targets)

        return losses


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, activation="gelu", drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = _get_activation_fn(activation)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class FFN(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


####################################################################################
# segmentation
####################################################################################

class DETRsegm(nn.Module):
    def __init__(self, detr, out_channel=3, freeze_detr=False, decode=True):
        super().__init__()
        self.detr = detr
        self.out_channel = out_channel
        self.decode = decode

        # if freeze_detr:
        #     for p in self.parameters():
        #         p.requires_grad_(False)
        hidden_dim = detr.transformer.in_channels
        nheads = 8
        # hidden_dim, nheads = detr.transformer.d_model, detr.transformer.nhead
        # self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0.0)
        self.mask_head = MaskHeadSmallConv(hidden_dim*2, [2048, 1024, 512], hidden_dim,
                                           out_channel=out_channel)  # + nheads 对应bbox_mask.flatten(0, 1)

    def forward(self, samples):
        features, pos = self.detr.backbone(samples)
        # 256， img_size // 4 依此类推
        batch_shape = samples.shape
        bs, c, h, w = batch_shape

        src = features[-1][0]
        mask = None
        # tensor = torch.zeros(batch_shape, dtype=samples.dtype, device=device)
        # mask = torch.ones((bs, h, w), dtype=torch.bool, device=device)
        # for img, pad_img, m in zip(samples, tensor, mask):
        #     pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
        #     m[: img.shape[1], :img.shape[2]] = False
        src_proj = self.detr.input_proj(src)
        if self.decode:
            dec_output, enc_output = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos[-1])
        else:
            dec_output, enc_output = self.detr.transformer(src_proj, mask, None, None, decode=self.decode)
        # hs, memory = self.detr.transformer(src_proj, mask, self.detr.query_embed.weight, pos_embed=None)  # pos[-1]
        # FIXME Transformer层在detr中仅用于做目标检测的工作，根据PVT的U-shaped结构可以用来做其他的
        #  注意Transformer结构的形状
        out = {}
        # outputs_class = self.detr.class_embed(hs)
        # outputs_coord = self.detr.bbox_embed(hs).sigmoid()
        # out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1]}
        # if self.detr.aux_loss:
        #     out['aux_outputs'] = self.detr._set_aux_loss(outputs_class, outputs_coord)

        # FIXME h_boxes takes the last one computed, keep this in mind
        # bbox_mask = self.bbox_attention(hs[-1], memory, mask=mask)
        '''
        TodoList
        1.nhead目前只用在了multi-head attention中，不确定与PVT等的实现是否一致
        2.decoder的输出是多个时间序列(估计是个查找匹配问题)，目前只取最后一次，其维度和self.mask_head之间的关系outputs_seg_masks? 
            *: 可以构造一个基于patches相似性的选patches操作
        3.mask、pos未采用，decoder的num_queries == P^2,P为patch_size
        4.FPN与transformer和mask_head目前效果不好
        '''
        seg_masks = self.mask_head(dec_output, [features[2][0], features[1][0], features[0][0]], args.num_queries)
        # outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries * self.out_channel, seg_masks.shape[-2],
        #                                    seg_masks.shape[-1])

        out["pred_masks"] = seg_masks  # outputs_seg_masks
        return out

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

def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)


class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim, out_channel=1):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], out_channel, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, fpns: List[Tensor], targets_size, num_queries):  # , bbox_mask: Tensor
        # x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
        # x = _expand(x, num_queries)
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)

        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)

        x = interpolate(x, size=targets_size, mode="bilinear", align_corners=False)

        x = self.out_lay(x)

        return x


class MHAttentionMap(nn.Module):
    """This is a 2D attention module, which only returns the attention softmax (no multiplication by value)"""

    def __init__(self, query_dim, hidden_dim, num_heads, dropout=0.0, bias=True):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)

        self.q_linear = nn.Linear(query_dim, hidden_dim, bias=bias)
        self.k_linear = nn.Linear(query_dim, hidden_dim, bias=bias)

        nn.init.zeros_(self.k_linear.bias)
        nn.init.zeros_(self.q_linear.bias)
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.q_linear.weight)
        self.normalize_fact = float(hidden_dim / self.num_heads) ** -0.5

    def forward(self, q, k, mask: Optional[Tensor] = None):
        q = self.q_linear(q)
        k = F.conv2d(k, self.k_linear.weight.unsqueeze(-1).unsqueeze(-1), self.k_linear.bias)
        qh = q.view(q.shape[0], q.shape[1], self.num_heads, self.hidden_dim // self.num_heads)
        kh = k.view(k.shape[0], self.num_heads, self.hidden_dim // self.num_heads, k.shape[-2], k.shape[-1])
        weights = torch.einsum("bqnc,bnchw->bqnhw", qh * self.normalize_fact, kh)

        if mask is not None:
            weights.masked_fill_(mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        weights = F.softmax(weights.flatten(2), dim=-1).view(weights.size())
        weights = self.dropout(weights)
        return weights


##########################################################################################
# deformable Dert
##########################################################################################
class Deformable_DETRsegm(nn.Module):
    def __init__(self, detr, freeze_detr=False):
        super().__init__()
        self.detr = detr

        if freeze_detr:
            for p in self.parameters():
                p.requires_grad_(False)

        hidden_dim, nheads = detr.transformer.in_channels, detr.transformer.nhead
        self.bbox_attention = MHAttentionMap(hidden_dim, hidden_dim, nheads, dropout=0)
        self.mask_head = MaskHeadSmallConv(hidden_dim, [2048, 1024, 512], hidden_dim, out_channel=3)
        self.input_proj = self.detr.input_proj
        self.num_feature_levels = self.detr.num_feature_levels

    def forward(self, samples):

        batch_shape = samples.shape
        bs, c, h, w = batch_shape
        tensor = torch.zeros(batch_shape, dtype=samples.dtype, device=device)
        mask = torch.ones((bs, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(samples, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False

        features, pos = self.detr.backbone(samples)

        mask_list = []
        for x in features:
            mask_list.append(F.interpolate(mask[None].float(), size=x.shape[-2:]).to(torch.bool)[0])

        # 256， img_size // 4 依此类推
        # batch_shape = features.shape
        # bs, c, h, w = batch_shape
        masks = []
        srcs = []
        for l, (src, mask) in enumerate(zip(features, mask_list)):
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.detr.backbone[1](src, mask).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        query_embeds = None
        if not self.detr.two_stage:
            query_embeds = self.detr.query_embed.weight

        dec_output, init_reference, inter_references, enc_outputs_class, enc_outputs_coord_unact \
            = self.detr.transformer(srcs, masks, query_embeds, pos)

        seg_masks = self.mask_head(dec_output, [features[2], features[1], features[0]], args.patch_size, args.num_queries)

        # seg_masks = self.mask_head(src_proj, bbox_mask, [features[2].tensors, features[1].tensors, features[0].tensors])
        # outputs_seg_masks = seg_masks.view(bs, self.detr.num_queries, seg_masks.shape[-2], seg_masks.shape[-1])
        out = {"pred_masks": seg_masks}

        return out


##########################################################################################
# arch
##########################################################################################
# from san_pvt import Pvt

'''
Transformer在detr中接受来自resnet的输出，只用了FPN的最后一层？进行目标检测，而分割任务是全卷积网络实现的
'''


def build_newtransformer(args):
    # return Pvt(
    #     in_channels=3,
    #     out_channels=3,
    #     hidden_dim=args.hidden_dim,
    #     nheads=args.nheads
    # )
    # return TrBlock(
    #     in_channels=args.hidden_dim,
    #     hidden_dim=args.dim_feedforward,
    #     nhead=args.nheads,
    #     num_encoder_layers=args.enc_layers,
    #     num_decoder_layers=args.dec_layers,
    #     normalize_before=args.pre_norm,
    #     return_intermediate_dec=False
    # )
    from Deformable_DETR.models.deformable_transformer import DeformableTransformer
    return DeformableTransformer(
        in_channels=args.hidden_dim,
        nhead=args.nheads,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=args.num_feature_levels,
        dec_n_points=args.dec_n_points,
        enc_n_points=args.enc_n_points,
        two_stage=False,
        two_stage_num_proposals=args.num_queries
    )


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class awin your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223

    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_newtransformer(args)

    model = DeformableDETR(
        backbone,
        transformer,
        num_queries=args.num_queries,
        num_feature_levels=args.num_feature_levels,
        aux_loss=args.aux_loss,
        two_stage=False,
    )
    if args.masks:
        model = Deformable_DETRsegm(model, freeze_detr=(args.frozen_weights is not None))

    weight_dict = {'l1_loss': 1}
    # losses = ['labels', 'boxes', 'cardinality']
    losses = []
    if args.masks:
        losses += ["masks"]  # masks
    criterion = SetCriterion(num_classes=0, weight_dict=weight_dict, eos_coef=args.eos_coef, losses=losses)
    criterion.to(device)

    return model, criterion


psnr_y = PSNR_ycbcr()
g_ssim = SSIM(size_average=False, data_range=255)
def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, max_norm):
    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    # for batch in data_loader:
    for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
        samples = batch['O'].to(device)
        targets = batch['B'].to(device)  # [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(sub_mean(samples))
        loss_dicts = criterion(outputs, sub_mean(targets))
        # losses = loss_dicts['Loss']
        weight_dict = criterion.weight_dict
        losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
        # mse_loss = loss_dicts['mse_loss']
        # loss, w_1, w_2, w_s, w_bce, ssim_m

        with torch.no_grad():
            pred = add_mean(outputs["pred_masks"])
            metric_logger.update(psnr=psnr_y(pred, targets * 255, 4, 255.0))
            metric_logger.update(ssim=torch.mean(g_ssim(pred, targets * 255)))
            # metric_logger.update(psnr=PSNR(None, None, mse_loss, False))

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        loss_dicts['Loss'] = losses

        metric_logger.update(**loss_dicts)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    log_string("TrainEpoch[{}] Averaged stats: {}".format(epoch, metric_logger))
    # 解耦，用于在main中调整日志
    return {k: meter.avg for k, meter in metric_logger.meters.items()}


################################################################################
# framework
################################################################################
def save_checkpoint(state, model_save_dir, is_best, filename='amp_checkpoint.pth.tar'):
    filename = os.path.join(model_save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_dir, 'amp_model_best.pth.tar'))

def load_model(args, model , optimizer):
    if args.resume:
        if os.path.isfile(args.resume):
            log_string("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = args.best_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            args.best_epoch = checkpoint['best_epoch']
            args.best_prec1 = checkpoint['best_psnr']
            model.load_state_dict(checkpoint['state_dict'])
            if hasattr(checkpoint, 'optimizer'):
                optimizer.load_state_dict(checkpoint['optimizer'])
            if optimizer is not None:
                if hasattr(checkpoint, 'optimizer'):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                log_string("=> loaded checkpoint '{}' (epoch {})"
                           .format(args.resume, checkpoint['epoch']))
        else:
            log_string("=> no checkpoint found at '{}'".format(args.resume))

    return model, optimizer

class EpochRunner():

    def __init__(self, args, sess, experimental_desc):
        self.args = args
        out_dir, model_save_dir, tfb_dir = create_logger(args, experimental_desc)
        self.args.out_dir = out_dir
        self.args.model_save_dir = model_save_dir
        self.args.tfb_dir = tfb_dir
        # self.tester = Tester(args)
        self.std_criterion = torch.nn.MSELoss(reduction='mean').cuda()
        self.sess = sess
        # self.ssim = SSIM().cuda()
        self.bcmsl = BCMSLoss().cuda()  # torch.nn.L1Loss().cuda()

    def eval(self, eval_loader, model, criterion, eval_sampler):
        # if self.args.distributed:
        #     eval_sampler.set_epoch(0)
        # print(self.args.distributed)
        model.init_eval_obj(args)
        model, _ = load_model(args, model, None)
        val_loss = self.eval_framework(eval_loader, model, criterion)

    def run(self, train_sampler, train_loader, model, criterion, optimizer, scale, val_loader, scheduler):
        # global best_prec1
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        log_string(model)
        model, optimizer = load_model(args, model, optimizer)
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # if self.args.distributed and not self.args.DALI:
            #     train_sampler.set_epoch(epoch)
            # adjust_learning_rate(optimizer, epoch)

            epoch_time = datetime.datetime.now()
            train_stats = train_one_epoch(model, criterion, train_loader, optimizer, self.args.device,
                                         epoch, self.args.clip_max_norm)

            # val_loss = self.validate_framework(val_loader, model, criterion, epoch)

            # if self.args.lr_scheduler and scheduler is not None:
            #     scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
            val_loss = train_stats['Loss']
            psnr = train_stats['psnr']
            is_best = psnr > self.args.best_prec1
            self.args.best_prec1 = max(psnr, self.args.best_prec1)

            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': self.args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': self.args.best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, self.args.model_save_dir, is_best)

            if epoch % self.args.print_freq == 0 or is_best:
                if is_best:
                    self.args.best_epoch = epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'state_dict': model.state_dict(),
                    'best_psnr': self.args.best_prec1,
                    'loss': val_loss,
                    'best_epoch': self.args.best_epoch,
                    # 'optimizer': optimizer.state_dict(),
                }, self.args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")

            log_string(' * Best validation Loss so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                loss=self.args.best_prec1, best_epoch=self.args.best_epoch))

            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            # log_string("one epoch time: {}".format(
            #     datetime.datetime.now() - epoch_time))
            log_string('Training time {}'.format(total_time_str))

    @torch.no_grad()
    def validate_framework(self, val_loader, model, criterion, epoch=0):

        metric_logger = MetricLogger(delimiter="  ")
        header = 'TestEpoch: [{0}/{1}]'.format(epoch, self.args.epochs)
        # switch to evaluate mode
        model.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch in metric_logger.log_every(val_loader, 1, header):
            samples = batch['O'].to(self.args.device, non_blocking=True)
            targets = batch['B'].to(self.args.device, non_blocking=True)
            # O, gt = Variable(O, requires_grad=False), Variable(gt, requires_grad=False)
            # compute output

            outputs = model(samples)
            loss_dicts = criterion(outputs, targets)

            # weight_dict = criterion.weight_dict
            # losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
            # loss_dicts['Loss'] = losses

            metric_logger.update(**loss_dicts)
            metric_logger.update(psnr=PSNR(None, None, loss_dicts['mse_loss'], False))
        log_string(header+"Averaged stats: {}".format(metric_logger))
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}

        return stats

    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion):
        args = self.args
        metric_logger = MetricLogger(delimiter="  ")
        header = 'TestEpoch: [{0}]'.format(args.start_epoch)
        # switch to evaluate mode
        model.eval()
        psnr_list = []
        # for iteration, batch in enumerate(val_loader, 1):
        saved_path = f"./my_model_results/detr/{args.eval}"

        if os.path.exists(saved_path) is False:
            os.mkdir(saved_path)

        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            # index = batch['index']
            samples = batch['O'].to(args.device, non_blocking=True)
            gt = batch['B'].to(args.device, non_blocking=True)
            filename = batch['file_name']
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
            # if args.distributed:
            #     outputs = model.module.forward(samples)  # forward_chop(sub_mean(samples))
            # else:
            outputs = model.forward_chop(samples)  # forward_chop(sub_mean(samples))
            normalized = outputs[0].mul(255 / 1.0)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

            imageio.imwrite(os.path.join(saved_path, ''.join([filename[0], '.png'])),
                            tensor_cpu.numpy())

            # pred_np = quantize(outputs.cpu().detach().numpy(), 255)
            # psnr = reduce_mean(psnr_v2(add_mean(outputs), gt * 255, 4, 255.0))
            # ssim = g_ssim(add_mean(outputs) / 255.0, gt)
            # psnr = calc_psnr(outputs.cpu().numpy(), gt.cpu().numpy() * 255, 4, 255.0)  # [0].permute(1, 2, 0)
            # metric_logger.update(**loss_dicts)
            # psnr_list.append(psnr.item())
            print(filename)
            # metric_logger.update(ssim=ssim)
            # metric_logger.update(psnr=psnr)  # PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        log_string("Averaged stats: {} ({})".format(metric_logger, np.mean(psnr_list)))

        return stats  # stats

if __name__ == "__main__":

    from dataset import derainSession
    import numpy as np
    import argparse
    import random
    from torch.backends import cudnn

    model_path = './results/train/DDETR/ddetr/model_2021-09-04-21-00/215.pth.tar'

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # * Logger
    parser.add_argument('--out_dir', metavar='DIR', default='./results',
                        help='path to save model')
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DDETR')
    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--resume',
                        default=model_path,
                        type=str, metavar='PATH',
                        # results/100L/APUnet/derain_large_V2/model_2021-04-06-23-54/487.pth.tar
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--lr', default=2e-4, type=float)
    parser.add_argument('--lr_backbone_names', default=["backbone.0"], type=str, nargs='+')
    parser.add_argument('--lr_backbone', default=2e-4, type=float)
    parser.add_argument('--lr_linear_proj_names', default=['reference_points', 'sampling_offsets'], type=str, nargs='+')
    parser.add_argument('--lr_linear_proj_mult', default=0.1, type=float)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('-b', '--batch-size', default=64, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')

    parser.add_argument('--epochs', default=1500, type=int)
    parser.add_argument('--lr_drop', default=40, type=int)
    parser.add_argument('--lr_drop_epochs', default=None, type=int, nargs='+')
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned', 'None'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--position_embedding_scale', default=2 * np.pi, type=float,
                        help="position / size * scale")
    parser.add_argument('--num_feature_levels', default=3, type=int, help='number of feature levels')#4

    # * Transformer

    parser.add_argument('--enc_layers', default=6, type=int,  # 6
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,  # 6
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,  # 2048
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=256, type=int,  # 256,由于使用了组归一和4次上采样通道变化，hidden_dim / 2^4 = 8x(mod8 ==0)
                        help="Size of the embeddings (dimension of the transformer P^2*C)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,  # 8
                        help="Number of attention heads inside the transformer's multi-head attentions")
    parser.add_argument('--num_queries', default=16, type=int,  # 100
                        help="Number of query slots")  # only used for deteted bbox nums->batch
    parser.add_argument('--pre_norm', action='store_true')
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)

    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")
    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")
    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='image2patch, set to model and dataset')
    parser.add_argument('--eval', default=None, type=str,
                        choices=[None, 'rain200H', 'rain100L', 'rain200H', 'rain100H',
                                 'test12', 'real', 'DID', 'SPA', 'DDN'],
                        help="performing evalution for patch2entire")
    parser.add_argument('--crop_batch_size', type=int, default=128,
                        help='input batch size for training')

    args = parser.parse_args()
    args.experimental_desc = "ddetr"
    args.dataset = "train"

    sess = derainSession(args)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    device = torch.device(args.device)

    ##################################################
    model, criterion = build(args)
    model.to(device)
    model_without_ddp = model
    def match_name_keywords(n, name_keywords):
        out = False
        for b in name_keywords:
            if b in n:
                out = True
                break
        return out


    # lr_backbone_names = ["backbone.0", "backbone.neck", "input_proj", "transformer.encoder"]
    # param_dicts = [
    #     {
    #         "params":
    #             [p for n, p in model_without_ddp.named_parameters()
    #              if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_backbone_names) and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad],
    #         "lr": args.lr * args.lr_linear_proj_mult,
    #     }
    # ]

    # for n, p in model_without_ddp.named_parameters():
        # if "backbone" not in n and p.requires_grad:
        # if not match_name_keywords(n, args.lr_backbone_names) and not match_name_keywords(n, args.lr_linear_proj_names) and p.requires_grad:
        #     print(n)


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    ##################################################


    ##################################################
    args.best_prec1 = 0
    args.best_prec5 = 0
    # args.best_epoch = 0
    # train_loader = iter(list(train_loader))

    # for epoch in range(args.start_epoch, args.epochs):
    #     train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm)
    runner = EpochRunner(args, sess, args.experimental_desc)

    if args.eval is not None:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.eval, False)
        runner.eval(eval_loader, model, criterion, eval_sampler)
    else:
        train_loader, train_sampler = sess.get_dataloader(args.dataset, False)


    runner.run(None, train_loader, model, criterion, optimizer, None,
               None, None)
