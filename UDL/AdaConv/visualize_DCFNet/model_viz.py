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


class HighResolutionModule(nn.Module):
    '''
    高低分支交叉
    '''
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output
        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(inplace=True)

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

        # 构建下采样分支模块到layers中
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.num_inchannels[branch_index],
                          num_channels[branch_index] * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_channels[branch_index] * block.expansion,
                               momentum=BN_MOMENTUM)
                # BatchNorm2d(num_channels[branch_index] * block.expansion,
                #             momentum=BN_MOMENTUM),
            )

        layers = []
        # 加深卷积层，仅第一层牵涉到下采样
        layers.append(block(self.num_inchannels[branch_index],
                            num_channels[branch_index], stride, downsample))

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

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                elif j > i:
                    width_output = x[i].shape[-1]
                    height_output = x[i].shape[-2]
                    y = y + F.interpolate(
                        self.fuse_layers[i][j](x[j]),
                        size=[height_output, width_output],
                        mode='bilinear', align_corners=True)
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


# -----------------------------------------------------
class HighResolutionPanNet(nn.Module):
    def __init__(self):
        super(HighResolutionPanNet, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        NUM_CHANNELS = 64
        self.layer1 = self._make_layer(block=Bottleneck, inplanes=64, planes=NUM_CHANNELS, blocks=4)
        stage1_out_channel = Bottleneck.expansion * NUM_CHANNELS

        # stage2 = stage1(stem) + downsample_branch
        # TODO: 原始版本是采用yacs做的参数设置，暂不提前优化
        NUM_MODULES = 1
        self.Stage2_NUM_BRANCHES = 2
        NUM_BLOCKS = [4, 4]
        NUM_CHANNELS = [32, 64]

        num_channels = [
            NUM_CHANNELS[i] * BasicBlock.expansion for i in range(len(NUM_CHANNELS))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES, num_branches=self.Stage2_NUM_BRANCHES,
            num_blocks=NUM_BLOCKS, num_inchannels=num_channels, num_channels=NUM_CHANNELS, block=BasicBlock)

        ms_channels = 8
        NUM_MODULES = 1
        self.Stage3_NUM_BRANCHES = 3
        NUM_BLOCKS = [4, 4, 4]
        NUM_CHANNELS = [32, 64, 128]
        num_channels = [
            NUM_CHANNELS[i] * BasicBlock.expansion for i in range(len(NUM_CHANNELS))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES, num_branches=self.Stage3_NUM_BRANCHES,
            num_blocks=NUM_BLOCKS, num_inchannels=num_channels, num_channels=NUM_CHANNELS, block=BasicBlock)

        last_inp_channels = np.int(np.sum(pre_stage_channels)) + ms_channels
        FINAL_CONV_KERNEL = 1

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            # BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=8,
                kernel_size=FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
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

    def _make_stage(self, num_modules, num_branches, num_blocks, num_inchannels,
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
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:
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
                        nn.ReLU(inplace=False)))
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
                        nn.Conv2d(
                            inchannels, outchannels, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        # BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=False)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def forward(self, x, y):
        '''
        :param x: high resolution PAN is inputed in Network, shape:[Nx1x64x64]
        :param y: low resolution ms image is used to fine-tune PAN image to produce lms
                  ,shape:[Nx8x16x16]
        :return: ms with high resolution, shape:[Nx8x64x64]
        '''
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.Stage2_NUM_BRANCHES):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.Stage3_NUM_BRANCHES):
            if self.transition2[i] is not None:
                if i < self.Stage2_NUM_BRANCHES:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage3(x_list)

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

        # Upsampling

        #
        x[2] = torch.cat([x[2], y], 1)

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


class HighResolutionPanNet_V2(nn.Module):
    def __init__(self):
        super(HighResolutionPanNet_V2, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn1 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        # self.bn2 = BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)

        # stage 1
        NUM_CHANNELS = 64
        self.layer1 = self._make_layer(block=Bottleneck, inplanes=64, planes=NUM_CHANNELS, blocks=4)
        stage1_out_channel = Bottleneck.expansion * NUM_CHANNELS

        # stage2 = stage1(stem) + downsample_branch
        # TODO: 原始版本是采用yacs做的参数设置，暂不提前优化
        NUM_MODULES = 1
        self.Stage2_NUM_BRANCHES = 2
        NUM_BLOCKS = [4, 4]
        NUM_CHANNELS = [32, 64]

        num_channels = [
            NUM_CHANNELS[i] * BasicBlock.expansion for i in range(len(NUM_CHANNELS))]
        self.transition1 = self._make_transition_layer(
            [stage1_out_channel], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES, num_branches=self.Stage2_NUM_BRANCHES,
            num_blocks=NUM_BLOCKS, num_inchannels=num_channels, num_channels=NUM_CHANNELS, block=BasicBlock)

        ms_channels = 8
        NUM_MODULES = 1
        self.Stage3_NUM_BRANCHES = 3
        NUM_BLOCKS = [4, 4, 4]
        NUM_CHANNELS = [32, 64, 128]
        num_channels = [
            NUM_CHANNELS[i] * BasicBlock.expansion for i in range(len(NUM_CHANNELS))]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            num_modules=NUM_MODULES, num_branches=self.Stage3_NUM_BRANCHES,
            num_blocks=NUM_BLOCKS, num_inchannels=num_channels, num_channels=NUM_CHANNELS, block=BasicBlock)

        last_inp_channels = np.int(np.sum(pre_stage_channels))  # ms_channels
        FINAL_CONV_KERNEL = 1

        self.last_layer = nn.Sequential(
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=last_inp_channels,
                kernel_size=1,
                stride=1,
                padding=0),
            nn.BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            # BatchNorm2d(last_inp_channels, momentum=BN_MOMENTUM),
            nn.ReLU(inplace=False),
            nn.Conv2d(
                in_channels=last_inp_channels,
                out_channels=8,
                kernel_size=FINAL_CONV_KERNEL,
                stride=1,
                padding=1 if FINAL_CONV_KERNEL == 3 else 0)
        )

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):

        # 用于stage1通道不匹配时或下采样后，提升通道
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

    def _make_stage(self, num_modules, num_branches, num_blocks, num_inchannels,
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
                                     num_channels,
                                     fuse_method,
                                     reset_multi_scale_output)
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)  # 2,3,4,4
        num_branches_pre = len(num_channels_pre_layer)  # 1,2,3,4

        transition_layers = []
        # 进行阶段间卷积层的层全连接
        for i in range(num_branches_cur):
            if i < num_branches_pre:
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
                        nn.Conv2d(
                            inchannels, outchannels - 8, kernel_size=3, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(outchannels - 8, momentum=BN_MOMENTUM),
                        # BatchNorm2d(outchannels, momentum=BN_MOMENTUM),
                        nn.ReLU(inplace=True)))
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def forward(self, x, lx, mx, sx):
        '''
        :param x: high resolution PAN is inputed in Network, shape:[Nx1x64x64]
        :param y: low resolution ms image is used to fine-tune PAN image to produce lms
                  ,shape:[Nx8x16x16]
        :return: ms with high resolution, shape:[Nx8x64x64]
        '''
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
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        # (1):32,64,64
        # (2):64,32,32
        x_list[-1] = torch.cat([x_list[-1], mx], 1)

        # 对每个分支构建卷积模块
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.Stage3_NUM_BRANCHES):
            if self.transition2[i] is not None:
                if i < self.Stage2_NUM_BRANCHES:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x_list[-1] = torch.cat([x_list[-1], sx], 1)
        # (1):32,64,64
        # (2):64,32,32
        # (4):128,16,16
        x = self.stage3(x_list)

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


# from ops import show_map
import matplotlib.pyplot as plt
import torchvision
from typing import List
import postprocess as pp

plt.ion()

# f, axarr = plt.subplots(1, 3)
f1, axarr1 = plt.subplots(1, 1)
f2, axarr2 = plt.subplots(1, 1)
f3, axarr3 = plt.subplots(1, 1)
f4, axarr4 = plt.subplots(1, 1)

class Feature_Extractor:
    def __init__(self, model, target_layer: List[str]):

        # 多级索引转化成一个key, 如:stage3.0.fuse_layers.2.1.0.1, stage3.0.branches.2.3.conv2
        self.submodule_dict = dict(list(model.named_modules())[1:])  # dict(model.named_modules())

        if target_layer is None:
            target_layer = self.submodule_dict.keys()  # list(self.submodule_dict.keys())[1:]
        else:
            for t in target_layer:
                if t not in self.submodule_dict.keys():
                    raise ValueError(f"Unable to find submodule {target_layer} in the model")


        self.target_layer = target_layer

        # self.model = model
        self.hook_handles = {}
        # for layer_name in self.target_layer:
        #     module = self.submodule_dict[layer_name]
        #     if not isinstance(module, nn.Conv2d):
        #         continue
        #     else:
        #         # Forward hook
        #         self.hook_handles.update(
        #             {layer_name: module.register_forward_hook(self._get_features_hook)})
        # Enable hooks
        self._hooks_enabled = True
        self.feature_maps = []
        self.gradient = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        if self._hooks_enabled:
            self.feature_maps.append(output.data)
            print(module, len(self.feature_maps))
            # print("input shape: {}, feature shape:{}".format(input[0].size(), output.size()))

    def _get_grads_hook(self, module, input_grad, output_grad):
        """

        :param input_grad: tuple, input_grad[0]: None
                                   input_grad[1]: weight
                                   input_grad[2]: bias
        :param output_grad:tuple,长度为1
        :return:
        """
        if self._hooks_enabled:
            self.gradient.append(output_grad[0])
            print(module, len(self.gradient))
            # print("gradient shape:{}".format(output_grad[0].size()))

    def _register_hook(self):
        for (layer_name, module) in self.submodule_dict.items():
            if layer_name in self.target_layer:
                if not isinstance(module, nn.Conv2d):
                    # TODO: 还有Relu
                    continue
                else:
                    if layer_name == "stage2.0.transition_layers.2.0.0.b2_conv_out.0":
                        continue
                    # if self.target_layer is not None and layer_name in self.target_layer:
                    self.hook_handles.update(
                                {layer_name: [
                                module.register_forward_hook(self._get_features_hook),
                                module.register_backward_hook(self._get_grads_hook)]})
                    # self.hook_handles.update({layer_name: module.register_backward_hook(self._get_grads_hook)})

    def clear_hooks(self) -> None:
        """Clear model hooks"""
        # for handle in self.hook_handles:
        #     handle.pop()
        print(self.hook_handles.keys())
        del self.hook_handles

    # 得到特征图、得到梯度信息、得到各层权重
    # 调整策略：图像颜色规范化、尺度采样到一致,去掉无用信息，映射回原图
    def forward(self, origin_pan, lms, scores, normalized):

        # self.feature_maps.append(lms)
        # self.feature_maps.reverse()
        self.gradient.reverse()
        # output = scores[0, :3, ...] * 2047.0
        # output = torch.clip_(output.permute(1, 2, 0), 0, 2047)
        # output = output.cpu().detach().numpy()
        # output = pp.linstretch(output)
        # plt.legend(loc='best')
        if isinstance(scores, list):
            output = pp.showimage8(scores[0])
            axarr3.imshow(output)
            # axarr3.set_title('Output')
            scores[0] = output
        else:
            output = pp.showimage8(scores)
            axarr3.imshow(output)
            # axarr3.set_title('Output')

        # lms = pp.showimage8(lms)
        f3.set_size_inches(7.0 / 3, 7.0 / 3)
        axarr3.set_axis_off()
        f3.savefig("Output.eps", format='eps', dpi=300, pad_inches=0, bbox_inches='tight')
        f1.set_size_inches(7.0 / 3, 7.0 / 3)
        f2.set_size_inches(7.0 / 3, 7.0 / 3)
        f4.set_size_inches(7.0 / 3, 7.0 / 3)

        print(self.hook_handles.keys())
        print(len(self.feature_maps), len(self.gradient), len(self.hook_handles))
        for idx, layer_name in enumerate(self.hook_handles.keys()):
            # module = self.submodule_dict[layer_name]
            # self.weight = module.weight.data
            # self.visualize_stn(self.feature_maps[idx], self.weight, self.gradient[idx])
            # self.gradient[idx] = pp.apply_gradient_images(self.gradient[idx], "")
            if self.feature_maps[idx].shape == self.gradient[idx].shape:
                self.visual_fg_response(origin_pan, scores, idx, layer_name)
                print("show fg: ", layer_name)
                plt.pause(1)
            # self.visual_gradient_response(origin_pan, lms, idx) # TODO: 存在通道数量差异
            # self.visual_feature([self.feature_maps[idx], self.gradient[idx]])
            # print(self.feature_maps[idx].shape, self.gradient[idx].shape)

            # plt.close('all')

    def visual_fg_response(self, origin_pan, pred_lms, idx, layer_name, is_save=False):

        axarr1.set_axis_off()
        axarr2.set_axis_off()
        axarr4.set_axis_off()
        # fig.savefig('T/d/avg{}.eps'.format(self.name), format='eps', transparent=True, dpi=300, pad_inches=0,
        #             bbox_inches='tight')
        if isinstance(pred_lms, list):
            split_channel = [32, 64, 128]
            gradient = self.gradient[idx].split(split_channel, dim=1)
            nums = 0
            for x, g in zip(pred_lms[1:], gradient):
                cam = pp.gen_grad_cam(origin_pan, x, g)
                # cam, heatmap = pp.apply_heatmap(pred_lms, cam)
                cam, heatmap = pp.apply_colormap_on_image(pred_lms[0], cam, "jet")  # jet viridis hsv

                axarr1.imshow(cam)
                # axarr1.set_title('Image cam')
                axarr2.imshow(heatmap)
                # axarr2.set_title('Image heatmap')
                x = x.squeeze(0).mean(axis=0).cpu().data.numpy()
                if nums == 0:
                    x = np.where(x > np.max(x) * 0.7, x ** 0.7, x)
                if nums == 1:
                    x = np.where(x > np.max(x) * 0.9, x ** 1.3, x)  # 10*torch.log1p(cam)
                axarr4.imshow(x)  # , cmap=plt.cm.get_cmap('jet'))
                # axarr4.set_title(layer_name+str(nums))
                if is_save:
                    fmts = ['png', 'eps']#'eps'
                    for fmt in fmts:
                        f1.savefig(f'results_viridis/test/cam_{nums}_{layer_name}.{fmt}', format=fmt, dpi=300, pad_inches=0, bbox_inches='tight')
                        f2.savefig(f'results_viridis/test/heatmap_{nums}_{layer_name}.' + fmt, format=fmt, dpi=300, pad_inches=0, bbox_inches='tight')
                        f4.savefig(f'results_viridis/test/feature_{nums}_{layer_name}.' + fmt, format=fmt, dpi=300, pad_inches=0, bbox_inches='tight')
                nums = nums + 1
            pred_lms = pred_lms[0]
        feature = F.interpolate(self.feature_maps[idx], [128, 128], mode='bilinear', align_corners=True)
        cam = pp.gen_grad_cam(origin_pan, feature, self.gradient[idx])
        # cam, heatmap = pp.apply_heatmap(pred_lms, cam)
        cam, heatmap = pp.apply_colormap_on_image(pred_lms, cam, "jet")#jet viridis hsv
        # axarr1.imshow(cam)
        # axarr1.set_title('Image cam')
        axarr2.imshow(heatmap)
        # axarr2.set_title('Image heatmap')
        # feature = F.interpolate(self.feature_maps[idx], [256, 256], mode='bilinear', align_corners=True)
        feature = feature.squeeze(0).mean(axis=0).cpu().data.numpy()
        axarr4.imshow(feature)#, cmap=plt.cm.get_cmap('jet'))
        # axarr4.set_title(layer_name)






        # fig = plt.gcf()
        # fig.set_size_inches(7.0 / 3, 7.0 / 3)  # dpi = 300, output = 700*700 pixels
        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # plt.show()


        # plt.gca().xaxis.set_major_locator(plt.NullLocator())
        # plt.gca().yaxis.set_major_locator(plt.NullLocator())
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        #fig = plt.gcf()  # 'get current figure'
        #fig.savefig(f'cam_{idx}.eps', format='eps', dpi=300)

        if is_save:
            fmts = ['eps', 'png']
            for fmt in fmts:
                f1.savefig(f'results_viridis/test/cam_1{idx}_{layer_name}.{fmt}', format=fmt, dpi=300, pad_inches=0, bbox_inches='tight')
                f2.savefig(f'results_viridis/test/heatmap_1{idx}_{layer_name}.' + fmt, format=fmt, dpi=300, pad_inches=0, bbox_inches='tight')
                f4.savefig(f'results_viridis/test/feature_1{idx}_{layer_name}.' + fmt, format=fmt, dpi=300, pad_inches=0, bbox_inches='tight')

    def visual_gradient_response(self, origin_pan, lms, idx):
        cam = pp.gen_grad_cam(origin_pan, lms, self.gradient[idx])
        cam, heatmap = pp.apply_heatmap(lms, cam)
        axarr1.imshow(cam)
        axarr1.set_title('Image cam')
        axarr2.imshow(heatmap)
        axarr2.set_title('Image heatmap')

        # cam = pp.gen_colormap(lms, lms, self.gradient[idx])
        # cam, heatmap = pp.apply_colormap_on_image(lms, cam, "hsv")
        # axarr1.imshow(cam)
        # axarr1.set_title('Image cam')
        # axarr2.imshow(heatmap)
        # axarr2.set_title('Image heatmap')

    # self.is_grid
    def visual_feature(self, tensors, is_grid=True):
        def basic_visual(tensors):
            if is_grid:
                grid_tensor = tensors[b_idx, ...].unsqueeze(0)
                # make_grid使用第一个维度去合并图像到一张
                grid_tensor = torchvision.utils.make_grid(
                    grid_tensor.permute(1, 0, 2, 3)).cpu().numpy().transpose(
                    (1, 2, 0))
                return grid_tensor[..., 0]
            # 纯numpy实现
            #       show_map(feature_maps, axarr, col=8)
            else:
                # NHWC
                grid_tensor = tensors.permute(0, 2, 3, 1).cpu().numpy()
                if channel == 3:
                    return grid_tensor[b_idx, :, :]
                else:
                    return grid_tensor[b_idx, :, :, c_idx]

        with torch.no_grad():
            if is_grid:
                c_idx = None
                # for col, tensor in enumerate(tensors):
                f, g = tensors
                batch_size, channel, _, _ = f.size()
                for b_idx in range(batch_size):
                    axarr1.imshow(basic_visual(f))
                    axarr1.set_title('Image feature')
                batch_size, channel, _, _ = g.size()
                for b_idx in range(batch_size):
                    axarr2.imshow(basic_visual(g))
                    axarr2.set_title('Image grad')
                # batch_size, channel, _, _ = w.size()
                # for b_idx in range(batch_size):
                #     axarr3.imshow(basic_visual(w))
                #     axarr3.set_title('Filter weight')

    # def visualize_stn(self, feature_maps, layer_weight, gradient, is_grid=True):
    #     with torch.no_grad():
    #         batch_size, channel, _, _ = feature_maps.size()
    #         if is_grid:
    #             for b_idx in range(batch_size):
    #                 feature_maps = feature_maps[b_idx, ...].unsqueeze(0)
    #                 # make_grid使用第一个维度去合并图像到一张
    #                 grid_feature = torchvision.utils.make_grid(
    #                     feature_maps.permute(1, 0, 2, 3)).cpu().numpy().transpose(
    #                     (1, 2, 0))
    #                 axarr.imshow(grid_feature[..., 0])
    #                 axarr.set_title('Dataset Images')
    #         # 纯numpy实现
    #         #       show_map(feature_maps, axarr, col=8)
    #         else:
    #             # NHWC
    #             feature_maps = feature_maps.permute(0, 2, 3, 1).cpu().numpy()
    #             for b_idx in range(batch_size):
    #                 if channel == 3:
    #                     axarr.imshow(feature_maps[b_idx, :, :])
    #                 else:
    #                     for c_idx in range(channel):
    #                         axarr.imshow(feature_maps[b_idx, :, :, c_idx])

    # PanSharpening 是in_ch = out_ch, 一一对应，因此用不到类别索引
    def __call__(self, origin_image, lms, scores=None, normalized=True):
        torch.set_grad_enabled(False)
        return self.forward(origin_image, lms, scores, normalized)


# 好处：可以跟test共用一套model运行过程,坏处：增加了test的代码
def build_visualize(img_tensor, model, target_layer=None, enabled_vis=False):
    # torch.enable_grad()
    criterion = nn.MSELoss().cuda()
    extractor = Feature_Extractor(model, target_layer)

    if isinstance(img_tensor, list):
        scores = model(*img_tensor)#, enabled_vis=enabled_vis)
    else:
        scores = model(img_tensor)#, enabled_vis=enabled_vis)
    extractor._hooks_enabled = True
    pan, gt = img_tensor[0], img_tensor[1]
    loss = criterion(scores[0], gt)
    model.zero_grad()
    loss.backward()

    target_layer = extractor(pan, gt, scores)

    extractor.clear_hooks()
    extractor._hooks_enabled = False


if __name__ == "__main__":
    # plt.ion()
    # gt Nx8x64x64
    # lms Nx8x64x64
    # mms Nx8x32x32
    # ms Nx8x16x16
    # pan Nx1x64x64
    from torchstat import stat

    '''
    ================================================================
    Total params: 2,745,096
    Trainable params: 2,745,096
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 329.64
    Params size (MB): 10.47
    Estimated Total Size (MB): 340.11
    ----------------------------------------------------------------
    =======================================================================================================================================================================
    Total params: 2,745,096
    Total MAdd: 7.18GMAdd
    Total Flops: 3.6GFlops
    -----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 125.57MB
    Total MAdd: 7.18GMAdd
    Total Flops: 3.6GFlops
    Total MemR+W: 266.82MB
    
    
    Total params: 2,768,040
    Total MAdd: 6.9GMAdd
    Total Flops: 3.46GFlops
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Total memory: 121.31MB
    Total MAdd: 6.9GMAdd
    Total Flops: 3.46GFlops
    Total MemR+W: 258.52MB
    
    
    ================================================================
    Total params: 2,768,040
    Trainable params: 2,768,040
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.00
    Forward/backward pass size (MB): 317.63
    Params size (MB): 10.56
    Estimated Total Size (MB): 328.18
    ----------------------------------------------------------------
            '''

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

    from model_fcc_dense_head import HighResolutionPanNet_V2 as model
    file_path = "new_data6.mat"
    lms, mms, ms, pan = load_set_V2(file_path)

    #ckpt = "v2_120.pth"
    ckpt = "./857.pth.tar"
    batch_size = 1
    net = model(mode="C").cuda()#HighResolutionPanNet_V2
    weight = torch.load(ckpt)
    print(weight.keys())
    # net.load_state_dict(weight["model"])
    net.load_state_dict(weight["state_dict"])
    from torchstat import stat
    # summary(net, input_size=[(1, 64, 64), (8, 64, 64), (8, 32, 32), (8, 16, 16)])
    # stat(net, input_size=[(1, 64, 64), (8, 64, 64), (8, 32, 32), (8, 16, 16)])

    # print(net)

    # pan = torch.randn(batch_size, 1, 64, 64).cuda()
    # ms = torch.randn(batch_size, 8, 16, 16).cuda()
    # mms = F.interpolate(ms, size=(ms.size(2) * 2, ms.size(3) * 2), mode="bilinear", align_corners=True)
    # lms = torch.randn(batch_size, 8, 64, 64).cuda()
    # out = net(pan, lms, mms, ms)
    # print(out.size())
    # net.eval()
    build_visualize([pan.cuda().reshape([1, 1, 256, 256]).float(), lms.cuda().reshape([1, 8, 256, 256]).float(),
                     mms.cuda().reshape([1, 8, 128, 128]).float(), ms.cuda().reshape([1, 8, 64, 64]).float()],
                    model=net,
                    target_layer=None
                    # target_layer=["stage3.0.fuse_layers.2.0.1.0", "stage3.0.fuse_layers.2.1.0.0","stage3.0.fuse_layers.1.0.0.0"]
                    # target_layer=["stage3.0.fuse_layers.0.1.0", "stage3.0.fuse_layers.0.2.0", "stage3.0.fuse_layers.1.2.0"]
                    #target_layer=["last_layer.3"]
                    #target_layer=["last_layer.0"]
                    , enabled_vis=False)#
    plt.ioff()
    # plt.show()

''' 
Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 1   conv1                                Conv2d(9, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 63
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 2  conv2                                Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 62
Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1), bias=False) 3                  layer1.0.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 4  layer1.0.conv2
Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False) 5                 layer1.0.conv3
Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False) 6                 layer1.0.downsample.0
Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False) 7                 layer1.1.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 8  layer1.1.conv2
Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False) 9                 layer1.1.conv3
Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False) 10                layer1.2.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 11 layer1.2.conv2
Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False) 12                layer1.2.conv3
Conv2d(256, 64, kernel_size=(1, 1), stride=(1, 1), bias=False) 13                layer1.3.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 14 layer1.3.conv2
Conv2d(64, 256, kernel_size=(1, 1), stride=(1, 1), bias=False) 15                layer1.3.conv3
Conv2d(256, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 16 transition1.0.0
Conv2d(256, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 17 transition1.1.0.0.b0_in_down.0
Conv2d(72, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 18  transition1.1.0.0.cfs_layers.0.conv_up
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 19  stage2.0.branches.0.0.conv1
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 20  stage2.0.branches.0.0.conv2
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 21  stage2.0.branches.0.1.conv1
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 22  stage2.0.branches.0.1.conv2
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 23  stage2.0.branches.0.2.conv1
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 24  stage2.0.branches.0.2.conv2
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 25  stage2.0.branches.1.0.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 26  stage2.0.branches.1.0.conv2
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 27  stage2.0.branches.1.1.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 28  stage2.0.branches.1.1.conv2
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 29  stage2.0.branches.1.2.conv1
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 30  stage2.0.branches.1.2.conv2
Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 31  #stage2.0.fuse_layers.1.0.0.0
Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 32 stage2.0.transition_layers.2.0.0.b1_down.0
Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 33 stage2.0.transition_layers.2.0.0.b1_in_down.0
Conv2d(136, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 34 #stage2.0.transition_layers.2.0.0.cfs_layers.0.conv_up  stage2.0.transition_layers.2.0.0.b2_conv_out.0
Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False) 35   #stage2.0.fuse_layers.0.1.0   
Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 36  #stage2.0.transition_layers.2.0.0.b0_in_down.0
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 37  stage3.0.branches.0.0.conv1
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 38  stage3.0.branches.0.0.conv2
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 39  stage3.0.branches.0.1.conv1
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 40  stage3.0.branches.0.1.conv2
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 41  stage3.0.branches.0.2.conv1
Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 42  stage3.0.branches.0.2.conv2
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 43
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 44
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 45
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 46
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 47
Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 48   stage3.0.branches.1.2.conv2
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 49
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 50
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 51
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 52
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 53 stage3.0.branches.2.2.conv1
Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False) 54 stage3.0.branches.2.2.conv2
Conv2d(64, 32, kernel_size=(1, 1), stride=(1, 1), bias=False) 55                   stage3.0.fuse_layers.0.1.0   111
Conv2d(128, 32, kernel_size=(1, 1), stride=(1, 1), bias=False) 56                  stage3.0.fuse_layers.0.2.0   111
Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 57   stage3.0.fuse_layers.1.0.0.0 222
Conv2d(128, 64, kernel_size=(1, 1), stride=(1, 1), bias=False) 58                  stage3.0.fuse_layers.1.2.0   111
Conv2d(32, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 59   stage3.0.fuse_layers.2.0.0.0 222
Conv2d(32, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 60  stage3.0.fuse_layers.2.0.1.0 222
Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False) 61  stage3.0.fuse_layers.2.1.0.0 222
Conv2d(224, 224, kernel_size=(1, 1), stride=(1, 1)) 62                              last_layer.0
Conv2d(224, 8, kernel_size=(1, 1), stride=(1, 1)) 63                                last_layer.3
'''