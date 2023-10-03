# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved
#
# @Author  : Xiao Wu
# @reference:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import functools

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
# from efficientdet.model import SeparableConvBlock, Conv2dStaticSamePadding

# from .sync_bn.inplace_abn.bn import InPlaceABNSync

# BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
BN_MOMENTUM = 0.01


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, momentum=BN_MOMENTUM)
        # self.bn3 = BatchNorm2d(planes * self.expansion,
        #                        momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class ChannelFuseBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, num_res=1, stride=1, downsample=None):
        super(ChannelFuseBlock, self).__init__()
        self.conv_up = conv3x3(8, planes, stride)
        self.epsilon = 1e-4
        self.rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.relu = nn.ReLU()
        # PanNetUnit类似于U^2 net necked U-net
        self.res_block = nn.ModuleList([])
        num_res = 1
        for i in range(num_res):
            self.res_block.append(BasicBlock(planes, planes, stride, downsample))
        # self.last_conv = conv3x3(planes, 8)

        # self.conv1 = conv3x3(8, planes, stride)
        # self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # # self.bn1 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.relu = nn.ReLU(inplace=True)
        # self.conv2 = conv3x3(planes, planes)
        # self.bn2 = nn.BatchNorm2d(8, momentum=BN_MOMENTUM)
        # # self.bn2 = BatchNorm2d(planes, momentum=BN_MOMENTUM)
        # self.downsample = downsample
        # self.stride = stride

    def forward(self, inputs):
        # Pan + Ms 1,8,H,W + 1,8,H,W
        # residual = x

        x, y = inputs[0], inputs[1]

        rs_w = self.relu(self.rs_w)
        weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)

        # y = torch.cat([x, y], dim=1)
        y = self.conv_up(y)

        out = weight[0] * x + weight[1] * y
        out = self.relu(out)
        for res_conv in self.res_block:
            out_rs = res_conv(out)
        # 跨残差层的密集连接
        if len(self.res_block) != 1:
            out_rs = out + out_rs
        # out = self.conv1(x)
        # out = self.bn1(out)
        # out = self.relu(out)
        #
        # out = self.conv2(out)
        # out = self.bn2(out)
        #
        # if self.downsample is not None:
        #     residual = self.downsample(x)

        # out = out + residual
        # out = self.relu(out)

        return out_rs


# class _make_branches(nn.Module):
#     def __init__(self, branch_index, num_branches, block, num_blocks, num_channels, stride=1):
#         super(_make_branches, self).__init__()
#         self.num_branches = num_branches
#         self.block = block
#         self.num_blocks = num_blocks
#         self.num_channels = num_channels
#
#         # 构建分支阶段的4层卷积
#         self.downsample = None
#         if stride != 1 or \
#                 self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(self.num_inchannels[branch_index],
#                           num_channels[branch_index] * block.expansion,
#                           kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
#                                momentum=BN_MOMENTUM)
#                 # BatchNorm2d(num_channels[branch_index] * block.expansion,
#                 #             momentum=BN_MOMENTUM),
#             )
#
#         layers = []
#         # 加深卷积层，仅第一层牵涉到下采样
#         # layers.append(block(self.num_inchannels[branch_index],
#         #                     num_channels[branch_index], stride, downsample))
#         self.cfb = ChannelFuseBlock(self.num_inchannels[branch_index], 8, stride, downsample)
#
#         self.num_inchannels[branch_index] = \
#             num_channels[branch_index] * block.expansion
#         for i in range(1, num_blocks[branch_index]):
#             layers.append(block(self.num_inchannels[branch_index],
#                                 num_channels[branch_index]))
#
#
#
#     def forward(self):
#         branches = []
#         for i in range(self.num_branches):
#             branches.append(
#                 self._make_one_branch(i, self.block, self.num_blocks, self.num_channels))


class HighResolutionModule(nn.Module):
    '''
    高低分支交叉 前后branches数
    '''

    def __init__(self, num_branches, blocks, num_blocks, num_inchannels, num_channels_pre_layer,
                 num_channels_cur_layer,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)
        self.epsilon = 1e-4
        self.num_inchannels = num_inchannels
        self.num_channels_pre_layer = num_channels_pre_layer
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        if num_branches == 2:
            self.transition_layers = self._our_make_transition_layer(num_channels_pre_layer, num_channels_cur_layer)

        self.relu = nn.ReLU(inplace=True)

        self.fcc_w = nn.Parameter(torch.ones(num_branches + 1, num_branches + 1, dtype=torch.float32),
                                  requires_grad=True)
        self.fcc_relu = nn.ReLU()

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            # logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            # logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            # logger.error(error_msg)
            raise ValueError(error_msg)

    # 构建具体的某一分支网络
    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):

        # # 构建分支阶段的4层卷积
        # downsample = None
        # if stride != 1 or \
        #         self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
        #     downsample = nn.Sequential(
        #         nn.Conv2d(self.num_inchannels[branch_index],
        #                   num_channels[branch_index] * block.expansion,
        #                   kernel_size=1, stride=stride, bias=False),
        #         nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
        #                        momentum=BN_MOMENTUM)
        #         # BatchNorm2d(num_channels[branch_index] * block.expansion,
        #         #             momentum=BN_MOMENTUM),
        #     )

        layers = []
        # 加深卷积层，仅第一层牵涉到下采样
        # layers.append(block(self.num_inchannels[branch_index],
        #                     num_channels[branch_index], stride, downsample))
        # layers.append(ChannelFuseBlock(self.num_inchannels[branch_index], 8, stride, downsample))

        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(block(self.num_inchannels[branch_index],
                                num_channels[branch_index]))

        return nn.Sequential(*layers)

    # 用于构建分支
    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)
    def _our_make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        # BatchNorm2d(
                        #     num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                # conv3x3s = []
                transfpn = []
                if i + 1 - num_branches_pre > 0:
                # for j in range(i + 1 - num_branches_pre):
                #     inchannels = num_channels_pre_layer[-1]
                #     outchannels = num_channels_cur_layer[i] \
                #         if j == i - num_branches_pre else inchannels
                    transfpn.append(nn.Sequential(
                        TransitionFPN(len(num_channels_cur_layer), 0, 0,
                                      kernel_size=3, stride=2, padding=1)))
                # for j in range(i + 1 - num_branches_pre):
                #     inchannels = num_channels_pre_layer[-1]
                #     outchannels = num_channels_cur_layer[i] \
                #         if j == i - num_branches_pre else inchannels
                #     transfpn.append(nn.Sequential(
                #         TransitionFPN(num_channels_cur_layer, inchannels, outchannels,
                #                       kernel_size=3, stride=2, padding=1)
                #     ))
                # conv3x3s.append(nn.Sequential(
                #     nn.Conv2d(  # TODO 除了-8外怎么平衡
                #         inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                #     nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                #     nn.ReLU(inplace=True)))

                transition_layers.append(nn.Sequential(*transfpn))
        # 获得同一个输入，不支持不同输入
        return nn.ModuleList(transition_layers)


    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                        nn.Conv2d(num_inchannels[j],
                                  num_inchannels[i],
                                  kernel_size=1,
                                  stride=1,
                                  padding=0,
                                  bias=False),
                        nn.BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                    # BatchNorm2d(num_inchannels[i], momentum=BN_MOMENTUM)))
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1,
                                          bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM)))
                            # BatchNorm2d(num_outchannels_conv3x3,
                            #             momentum=BN_MOMENTUM)))
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(nn.Sequential(
                                nn.Conv2d(num_inchannels[j],
                                          num_outchannels_conv3x3,
                                          kernel_size=3, stride=2, padding=1,
                                          bias=False),
                                nn.BatchNorm2d(num_outchannels_conv3x3, momentum=BN_MOMENTUM),
                                # BatchNorm2d(num_outchannels_conv3x3,
                                #             momentum=BN_MOMENTUM),
                                nn.ReLU(inplace=False)))
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, inputs):

        x, y = inputs[0], inputs[1]

        num_branches = self.num_branches
        fcc_w = self.fcc_relu(self.fcc_w)
        weight = fcc_w / (torch.sum(fcc_w, dim=0) + self.epsilon)

        if num_branches == 1:
            return [self.branches[0](x[0])]


        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        #################################################################################
        #        TODO:自适应分支数，匹配原版hrnet的for实现make_layer融合分支
        #################################################################################
        '''
        # x[j]  fuse_layers[i=0][j]  y
        #   |   i=0    -      w[i][0]   0 0      |          | \   | \
        #   |          /      w[i][j]   1 1                | -   | \
        #   |          /      w[i][j]   0 2                | /   | -
        # -: None
        # / : F.interpolate
        # \ : nn.Conv2d(stride=2)
        # '''
        # # 主干的三种情况的ops: weight[i][0]: -1 \2 \3
        # #                   weight[i][j]
        # x_fuse = []
        # for i in range(len(self.fuse_layers) - 1):  # x[0] + uc(x[1]) , 1 c(x[0])+ x[1] || 0 2 ,1 1 , 1 2, 2 1, 2 2
        #     y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
        #     # weight_m = fcc_w[i][0]
        #     for j in range(1, self.num_branches):
        #         weight_m = weight[i][0] if j == 1 else 1
        #         if i == j:
        #             y = weight_m * y + weight[i][j] * x[j]
        #             # y = weight[i][j] * y + weight[i][j - 1] * x[j]
        #         elif j > i:
        #             width_output = x[i].shape[-1]
        #             height_output = x[i].shape[-2]
        #             # y = weight[i][j - 1] * y + weight[i + 1][j] * F.interpolate(
        #             y = weight_m * y + weight[i][j] * F.interpolate(
        #                 self.fuse_layers[i][j](x[j]),
        #                 size=[height_output, width_output],
        #                 mode='bilinear', align_corners=True)
        #         else:
        #             y = weight_m * y + weight[i][j] * self.fuse_layers[i][j](x[j])
        #             # y = weight[i][j - 1] * y + weight[i][j - 1] * self.fuse_layers[i][j](x[j])
        #     x_fuse.append(self.relu(y))
        #
        # # if j == self.num_branches - 1:
        # '''
        #  \
        # -  1
        #   \ \
        # -    2
        #
        #  \
        # - 1
        #  \ \
        # -   2
        #    \ \
        # -     3
        #
        #  \
        # -  1
        #  \ |
        # -  2
        # '''
        #################################################################################
        #  our transition module, 仅处理新建分支操作
        #################################################################################

        # if num_branches == 2:
        #     b0_in, b1_in, b2_in = x[0], x[1], x[2]
        #
        #
        # # 32x64x64 64x32x32 8x3x32->64x32x32
        # # b0_in, b1_in, b2_in = x[0], x[1], x[2]
        # # b1_in_down = self.transition(b0_in)
        # # print(self.fuse_layers[num_branches])
        # # -> 64x32x32
        # # if num_branches == 2:
        # #     out = self.b1_down(b1_in + b0_in)
        # # else:
        # # out = self.cfs_layers((b1_in_down, b2_in))
        # out = self.b1_down(b1_in + self.b0_in_down(b0_in))  # 做步进 等价于maxpool + conv
        # if num_branches == 2:
        #     out = self.b2_conv_out(self.b1_in_down(b1_in) + self.cfs_layers((out, b2_in)))
        # # if num_branches == 3: # =3但并未新建分支，返回loss
        # #     # print(self.fuse_layers[num_branches])
        # #     out = self.b3_conv_out(x[-1] + self.b2_in_down(b2_in) + self.b2_out_down(out))  # ms = x[-1]
        #
        # # x_fuse.append(out)

        #################################################################################

        #################################################################################
        if num_branches == 2:
            x_new_branch = self.transition_layers[num_branches]((*x, y))

        x_fuse = []
        for i in range(len(self.fuse_layers)):  # x[0] + uc(x[1]) , 1 c(x[0])+ x[1] || 0 2 ,1 1 , 1 2, 2 1, 2 2
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            # weight_m = fcc_w[i][0]
            for j in range(1, self.num_branches):
                weight_m = weight[i][0] if j == 1 else 1
                if i == j:
                    y = weight_m * y + weight[i][j] * x[j]
                    # y = weight[i][j] * y + weight[i][j - 1] * x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    # y = weight[i][j - 1] * y + weight[i + 1][j] * F.interpolate(
                    y = weight_m * y + weight[i][j] * F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = weight_m * y + weight[i][j] * self.fuse_layers[i][j](x[j])
                    # y = weight[i][j - 1] * y + weight[i][j - 1] * self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))
        if num_branches == 2:
            for i in range(num_branches):
                if self.transition_layers[i] is not None:
                    if i < num_branches - 1:
                        x_fuse[i] = self.transition_layers[i](x_fuse[i])
            x_fuse.append(x_new_branch)
        # y = self.fuse_layers[j][0](x[0])
        # for jj in range(0, self.num_branches):  # jj 子块输出层级
        #     # weight_m = weight[i][0] if jj == 1 else 1
        #     # if i == jj:
        #     #     y = y + x[j]
        #     #     # y = weight[i][j] * y + weight[i][j - 1] * x[j]
        #     y = y + self.fuse_layers[num_branches][jj](x[jj + 1])
        # weight_m = weight[i][num_branches] if i == 0 else 1
        # y = weight_m * y + weight[i][num_branches] * self.fuse_layers[i][num_branches](x[])

        return x_fuse


class TransitionFPN(nn.Module):
    def __init__(self, num_branches_after_trans, inchannels=0, outchannels=0, kernel_size=3, stride=1, padding=1):
        super(TransitionFPN, self).__init__()
        self.num_branches = num_branches_after_trans  # num_branches_cur=2,3
        # if num_branches == 2:
        #     self.transition = nn.Sequential(nn.Conv2d(  # TODO 除了-8外怎么平衡
        #                                     num_inchannels[1], 128,
        #                                     kernel_size=3, stride=2, padding=1, bias=False),
        #                                     nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
        #                                     nn.ReLU(inplace=True))
        # if num_branches == 3:
        #     self.transition = nn.Sequential(nn.Conv2d(  # TODO 除了-8外怎么平衡
        #                                     128, 256,
        #                                     kernel_size=3, stride=2, padding=1, bias=False),
        #                                     nn.BatchNorm2d(256, momentum=BN_MOMENTUM),
        #                                     nn.ReLU(inplace=True))
        # self.cfs_layers = nn.Sequential(ChannelFuseBlock(inchannels[num_branches_after_trans - 1], outchannels[num_branches_after_trans]))
        if self.num_branches == 2:
            self.b0_in_down = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                                            nn.ReLU())
            self.cfs_layers = nn.Sequential(ChannelFuseBlock(64, 64))
        if self.num_branches == 3:
            self.epsilon = 1e-4
            self.b0_in_down = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(64, momentum=BN_MOMENTUM),
                                            nn.ReLU())
            self.b1_down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                         nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
                                         nn.ReLU())

            self.b1_in_down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                            nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
                                            nn.ReLU())

            self.b2_conv_out = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
                                             nn.BatchNorm2d(128, momentum=BN_MOMENTUM))  # no relu
            self.cfs_layers = nn.Sequential(ChannelFuseBlock(128, 128))

            self.rs_w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
            self.relu = nn.ReLU()

    # if self.num_branches == 3: # 没有新建分支，直接返回loss，所以不用
    #     self.b2_in_down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
    #                                     nn.BatchNorm2d(128, momentum=BN_MOMENTUM))
    #     self.b2_out_down = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
    #                                      nn.BatchNorm2d(128, momentum=BN_MOMENTUM),
    #                                      nn.ReLU())
    #
    #     self.b3_conv_out = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),
    #                                      nn.BatchNorm2d(128, momentum=BN_MOMENTUM))  # no relu
    #     self.conv3x3s = nn.Sequential(
    #         nn.Conv2d(  # TODO 除了-8外怎么平衡
    #             inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
    #         nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
    #         nn.ReLU(inplace=True))

    def forward(self, x):
        '''
        :param x: 32x64x64 | 32x64x64 64x32x32 | 32x64x64 64x32x32 128x16x16
        :param y: 8x64x64 | 8x32x32 | 8x16x16
        :return:
        '''
        # x = x[0]
        # out = self.conv3x3s(x)
        num_branches = self.num_branches
        # out = x
        #
        if num_branches == 2:
            b0_in, b1_in = x[0], x[1]  # ms = x[1]
            out = self.cfs_layers((self.b0_in_down(b0_in), b1_in))  # 做步进 等价于maxpool + conv
        if num_branches == 3:
            rs_w = self.relu(self.rs_w)
            weight = rs_w / (torch.sum(rs_w, dim=0) + self.epsilon)

            b0_in, b1_in, b2_in = x[0], x[1], x[2]  # ms = x[1]
            out = b1_in + self.b0_in_down(b0_in)  # 做步进 等价于maxpool + conv
            out = weight[0] * self.b1_in_down(b1_in) + weight[1] * self.cfs_layers((self.b1_down(out), b2_in))
        # 32x64x64 64x32x32 8x3x32->64x32x32
        # b0_in, b1_in, b2_in = x[0], x[1], x[2]
        # b1_in_down = self.transition(b0_in)
        # print(self.fuse_layers[num_branches])
        # -> 64x32x32
        # if num_branches == 2:
        #     out = self.b1_down(b1_in + b0_in)
        # else:
        # out = self.cfs_layers((b1_in_down, b2_in))
        # out = self.b1_down(b1_in + self.b0_in_down(b0_in))  # 做步进 等价于maxpool + conv
        # if num_branches == 2:
        #     out = self.b2_conv_out(self.b1_in_down(b1_in) + self.cfs_layers((out, b2_in)))
        # if num_branches == 3: # =3但并未新建分支，返回loss
        #     # print(self.fuse_layers[num_branches])
        #     out = self.b3_conv_out(x[-1] + self.b2_in_down(b2_in) + self.b2_out_down(out))  # ms = x[-1]

        # x_fuse.append(out)
        return out


################################################
# 1.为不同分辨率分支增加了对应尺寸原始图像做concat
# 2.输入接收Pan,lx,mx,sx的ms 8channel图, 可不可以用Pan到ms而不是ms到Pan，ms做主分支
################################################
class HighResolutionPanNet_V2(nn.Module):
    def __init__(self, init_channel=1, final_channel=8, mode="SUM"):
        super(HighResolutionPanNet_V2, self).__init__()
        self.first_fuse = mode
        if mode == "SUM":
            print("repeated sum head")
            init_channel = final_channel
        elif mode == "C":
            print("concat head")
            init_channel = final_channel + 1
        else:
            assert False, print("fisrt_fuse error")

        self.blocks_list = [4, [4, 4], [4, 4, 4]]
        self.channels_list = [64, [32, 64], [32, 64, 128]]
        NUM_MODULES = 1  # HighResolutionModule cell repeat
        ################################################################################################
        # stem net
        self.conv1 = nn.Conv2d(init_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        ################################################################################################
        # stage 1
        NUM_CHANNELS = 64
        self.layer1 = self._make_layer(block=Bottleneck, inplanes=64, planes=self.channels_list[0],
                                       blocks=self.blocks_list[0])
        stage1_out_channel = Bottleneck.expansion * self.channels_list[0]

        ################################################################################################
        # stage2 = stage1(stem) + downsample_branch
        # TODO: 原始版本是采用yacs做的参数设置，暂不提前优化
        # NUM_MODULES = 1
        self.Stage2_NUM_BRANCHES = 2
        NUM_BLOCKS = [4, 4]
        NUM_CHANNELS = [32, 64]

        num_channels = [
            self.channels_list[1][i] * BasicBlock.expansion for i in range(len(self.channels_list[1]))]

        self.transition1 = self._our_make_transition_layer(
            [stage1_out_channel], num_channels)
        transition_num_channels = [
            self.channels_list[2][i] * BasicBlock.expansion for i in range(len(self.channels_list[2]))]
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES, num_branches=self.Stage2_NUM_BRANCHES,
            num_blocks=self.blocks_list[1], num_inchannels=num_channels,
            num_channels_pre_layer=self.channels_list[1], num_channels_cur_layer=transition_num_channels,
            num_channels=self.channels_list[1], block=BasicBlock)

        # self.cfs_layers2 = self._make_ms_fuse(num_branches=self.Stage2_NUM_BRANCHES,
        #                                       block=ChannelFuseBlock,
        #                                       num_inchannels=num_channels,
        #                                       num_channels=self.channels_list[1],
        #                                       num_res_blocks=self.blocks_list[1])

        ################################################################################################
        # 三个分支分别卷积
        # ms_channels = 8
        # NUM_MODULES = 1
        self.Stage3_NUM_BRANCHES = 3
        NUM_BLOCKS = [4, 4, 4]
        NUM_CHANNELS = [32, 64, 128]
        num_channels = [
            self.channels_list[2][i] * BasicBlock.expansion for i in range(len(self.channels_list[2]))]
        # self.transition2 = self._make_transition_layer(
        #     pre_stage_channels, num_channels)
        # self.transition2 = self._our_make_transition_layer(
        #     pre_stage_channels, num_channels)

        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES, num_branches=self.Stage3_NUM_BRANCHES,
            num_blocks=self.blocks_list[2], num_inchannels=num_channels,
            num_channels_pre_layer=pre_stage_channels, num_channels_cur_layer=num_channels,
            num_channels=self.channels_list[2], block=BasicBlock)  # self.channels_list[2]

        # self.cfs_layers3 = self._make_ms_fuse(num_branches=self.Stage3_NUM_BRANCHES,
        #                                       block=ChannelFuseBlock,
        #                                       num_inchannels=num_channels,
        #                                       num_channels=self.channels_list[2],
        #                                       num_res_blocks=self.blocks_list[2])
        ################################################################################################
        # 保主分支，进行最终分支融合输出
        last_inp_channels = np.int(np.sum(pre_stage_channels))  # ms_channels
        FINAL_CONV_KERNEL = 1
        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=last_inp_channels, out_channels=last_inp_channels,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=final_channel,
                      kernel_size=FINAL_CONV_KERNEL, stride=1, padding=1 if FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_ms_fuse(self, num_branches, block, num_inchannels, num_channels, num_res_blocks, stride=1):
        branch_index = num_branches - 1
        downsample = None
        if stride != 1 or \
                num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM)
            )
        if torch.cuda.is_available():
            cfs_layers = nn.Sequential(ChannelFuseBlock(num_inchannels[branch_index], num_channels[branch_index],
                                                        num_res=num_res_blocks, stride=stride, downsample=downsample))
        else:
            cfs_layers = nn.Sequential(ChannelFuseBlock(num_inchannels[branch_index], num_channels[branch_index],
                                                        num_res=num_res_blocks, stride=stride, downsample=downsample))
        return cfs_layers

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        # 产生Bottleneck
        # stem主分支进行通道扩展时(64->256),下采样/2
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM)
                # BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    # 为每个分支进行构造卷积模块
    def _make_stage(self, num_modules, num_branches, num_blocks, num_inchannels,
                    num_channels_pre_layer, num_channels_cur_layer,
                    num_channels, block, fuse_method="SUM", multi_scale_output=True):

        modules = []  # HRNet的多尺度目标检测
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches,
                                     block,
                                     num_blocks,
                                     num_inchannels,
                                     num_channels_pre_layer,
                                     num_channels_cur_layer,
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output))
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    # 全连接步进卷积,产生多分支
    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        # BatchNorm2d(
                        #     num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(nn.Sequential(
                        nn.Conv2d(  # TODO 除了-8外怎么平衡
                            inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _our_make_transition_layer(self, num_channels_pre_layer, num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:  # 原有分支
                # Bottleneck的通道扩展恢复到原来
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(nn.Sequential(
                        nn.Conv2d(num_channels_pre_layer[i],
                                  num_channels_cur_layer[i],
                                  kernel_size=3,
                                  stride=1,
                                  padding=1,
                                  bias=False),
                        nn.BatchNorm2d(num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        # BatchNorm2d(
                        #     num_channels_cur_layer[i], momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                else:
                    transition_layers.append(None)
            else:
                # 最后一个连接要做步进卷积，并且nn.Sequential+1，说明是一个新分支
                # conv3x3s = []
                transfpn = []
                if i + 1 - num_branches_pre > 0:
                # for j in range(i + 1 - num_branches_pre):
                #     inchannels = num_channels_pre_layer[-1]
                #     outchannels = num_channels_cur_layer[i] \
                #         if j == i - num_branches_pre else inchannels
                    transfpn.append(nn.Sequential(
                        TransitionFPN(len(num_channels_cur_layer), 0, 0,
                                      kernel_size=3, stride=2, padding=1)))
                # for j in range(i + 1 - num_branches_pre):
                #     inchannels = num_channels_pre_layer[-1]
                #     outchannels = num_channels_cur_layer[i] \
                #         if j == i - num_branches_pre else inchannels
                #     transfpn.append(nn.Sequential(
                #         TransitionFPN(num_channels_cur_layer, inchannels, outchannels,
                #                       kernel_size=3, stride=2, padding=1)
                #     ))
                # conv3x3s.append(nn.Sequential(
                #     nn.Conv2d(  # TODO 除了-8外怎么平衡
                #         inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                #     nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                #     nn.ReLU(inplace=True)))

                transition_layers.append(nn.Sequential(*transfpn))
        # 获得同一个输入，不支持不同输入
        return nn.ModuleList(transition_layers)

    def forward(self, x, ly, my, sy):
        '''
        TODO: ly三层卷后级联x再做transition_layer
        Input Image kind: Pan or ms
        x: high resolution Image is inputed in Network, shape:[N, C, 64, 64]
        y: low resolution image is used to fine-tune PAN image to produce lms,
                  shape:[N, C, base_rsl=16, base_rsl=16],
                        [N, C, base_rsl*scale_up, base_rsl*scale_up],
                        [N, C, base_rsl*scale_up, base_rsl*scale_upper]
        :return: higher resolution image x, shape:[N, C, 64, 64]
        '''
        if self.first_fuse == "SUM":
            x = x.repeat(1, 8, 1, 1)
            x = x + ly
        if self.first_fuse == "C":
            x = torch.cat([x, ly], dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        # 64->256,64x64
        x = self.layer1(x)

        # 扩展网络主干
        '''
        ModuleList(
        (0): Sequential(
            (0): Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(32, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
            (2): ReLU()
        )
        (1): Sequential(
            (0): Sequential(
                (0): Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
                (1): BatchNorm2d(64, eps=1e-05, momentum=0.01, affine=True, track_running_stats=True)
                (2): ReLU()
                )
            )
        )
        '''
        x_list = []
        for i in range(self.Stage2_NUM_BRANCHES):
            if self.transition1[i] is not None:
                if i < self.Stage2_NUM_BRANCHES - 1:
                    x_list.append(self.transition1[i](x))
                else:
                    x_list.append(self.transition1[i]((x, my)))
            else:
                x_list.append(x)
        # (1):32,64,64
        # (2):64,32,32
        # x_list[-1] = torch.cat([x_list[-1], my], 1)

        # 对每个分支构建卷积模块 只有2个，第三个和第二个是共用的关系，不是完整的全连接
        # 在_make_fuse_layer中新增一个输出表示第三个,得到y_list此时做下采样
        # x_list.append(self.cfs_layers2((x_list[self.Stage2_NUM_BRANCHES - 1], my)))
        # x_list.append(my)
        x_list = self.stage2((x_list, sy))

        # x_list = []
        # for i in range(self.Stage3_NUM_BRANCHES):
        #     if self.transition2[i] is not None:
        #         if i < self.Stage2_NUM_BRANCHES:
        #             x_list.append(self.transition2[i](y_list[i]))
        #         else:
        #             x_list.append((self.transition2[i]((*y_list, sy))))
        #     else:
        #         x_list.append(y_list[i])
        #     if i == self.Stage3_NUM_BRANCHES - 1:
        #         x_list[i] = self.cfs_layers3((x_list[i], sy))
        # x_list[self.Stage3_NUM_BRANCHES - 1] = self.cfs_layers3((y_list[self.Stage3_NUM_BRANCHES - 1], sy))
        # x_list[-1] = torch.cat([x_list[-1], sy], 1)
        # (1):32,64,64
        # (2):64,32,32
        # (4):128,16,16
        # y_list.append(sy)
        x = self.stage3((x_list, None))

        # x_list = []
        # for i in range(self.stage4_cfg['NUM_BRANCHES']):
        #     if self.transition3[i] is not None:
        #         if i < self.stage3_cfg['NUM_BRANCHES']:
        #             x_list.append(self.transition3[i](y_list[i]))
        #         else:
        #             x_list.append(self.transition3[i](y_list[-1]))
        #     else:
        #         x_list.append(y_list[i])
        # x = self.stage4(x_list)
        ''''''
        # Upsampling
        # x[2] = torch.cat([x[2], sx], 1)

        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.interpolate(x[1], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        x2 = F.interpolate(x[2], size=(x0_h, x0_w), mode='bilinear', align_corners=True)
        # x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')

        x = torch.cat([x[0], x1, x2], 1)

        x = self.last_layer(x)

        return x

    def init_weights(self, pretrained='', ):
        # logger.info('=> init weights from normal distribution')
        print('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                # nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        if os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)
            print('=> loading pretrained model {}'.format(pretrained))
            # logger.info('=> loading pretrained model {}'.format(pretrained))
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict.keys()}
            for k, _ in pretrained_dict.items():
                print('=> loading {} pretrained model {}'.format(k, pretrained))
                # logger.info(
                #     '=> loading {} pretrained model {}'.format(k, pretrained))
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)


if __name__ == "__main__":
    import scipy.io as sio

    def load_set_V2(file_path):
        data = sio.loadmat(file_path)  # HxWxC

        # tensor type:
        lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
        # ms_hp = torch.from_numpy(get_edge(data['ms'] / 2047)).permute(2, 0, 1)  # CxHxW= 8x64x64
        # mms_hp = torch.nn.functional.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
        #                                            mode="bilinear", align_corners=True)
        # pan_hp = torch.from_numpy(get_edge(data['pan'] / 2047))   # HxW = 256x256

        ms_hp = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
        ms_hp = ms_hp.unsqueeze(dim=0)
        mms_hp = torch.nn.functional.interpolate(ms_hp, size=(ms_hp.size(2) * 2, ms_hp.size(3) * 2),
                                                 mode="bilinear", align_corners=True)
        pan_hp = torch.from_numpy(data['pan'] / 2047.0)  # HxW = 256x256

        return lms, mms_hp, ms_hp, pan_hp

    file_path = "../test_data/new_data6.mat"
    lms, mms, ms, pan = load_set_V2(file_path)

    # ckpt = "v2_120.pth"
    ckpt = "../benchmark/02-benchmark/00-Pansharpening/test_model/adam_hrfnet_concat_1e-3/857.pth.tar"
    batch_size = 1
    net = HighResolutionPanNet_V2().cuda()  # HighResolutionPanNet_V2
    weight = torch.load(ckpt)
    print(weight.keys())
    # net.load_state_dict(weight["model"])
    net.load_state_dict(weight["state_dict"])
    from torchsummaryV2 import summary
    summary(net, input_size=[(1, 64, 64), (8, 64, 64), (8, 32, 32), (8, 16, 16)])
    # from torchstat import stat
    # stat(net, input_size=[(1, 64, 64), (8, 64, 64), (8, 32, 32), (8, 16, 16)])

    # print(net)

    # pan = torch.randn(batch_size, 1, 64, 64).cuda()
    # ms = torch.randn(batch_size, 8, 16, 16).cuda()
    # mms = F.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2), mode="bilinear", align_corners=True)
    # lms = torch.randn(batch_size, 8, 64, 64).cuda()
    # out = net(pan, lms, mms, ms)
    # print(out.size())
    # net.eval()
    # build_visualize([pan.cuda().reshape([1, 1, 256, 256]).float(), lms.cuda().reshape([1, 8, 256, 256]).float(),
    #                  mms.cuda().reshape([1, 8, 128, 128]).float(), ms.cuda().reshape([1, 8, 64, 64]).float()],
    #                 model=net, target_layer=None)
    # plt.ioff()
    # plt.show()
