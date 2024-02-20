# -*- encoding: utf-8 -*-
"""
@File    : model_SR.py
@Time    : 2021/12/2 17:02
@Author  : Shangqi Deng
@Email   : dengsq5856@126.com
@Software: PyCharm
"""
import math

import torch
from torch import optim
from UDL.Basis.criterion_metrics import *
from UDL.hisr.HISR.Swin_poolv5.Swin import *
from UDL.pansharpening.common.evaluate import analysis_accu
from UDL.Basis.module import PatchMergeModule
from UDL.Basis.pytorch_msssim.cal_ssim import SSIM
import torch.nn.functional as F


def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):   ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):     ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

def init_w(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                # torch.nn.init.uniform_(m.weight, a=0, b=1)
            elif isinstance(m, nn.Conv2d):   ## initialization for Conv2d
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class Swinnet(PatchMergeModule):
    def __init__(self, args):
        super(Swinnet, self).__init__()
        self.args = args
        self.img_size = 64
        self.in_channels = 31
        self.embed_dim = 32
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed_dim, self.in_channels, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        self.u = nn.Sequential(
            nn.Conv2d(self.embed_dim*2, self.embed_dim*8, 3, 1, 1), nn.PixelShuffle(2), nn.LeakyReLU(0.2, True)
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(self.embed_dim*2, 31, 3, 1, 1), nn.LeakyReLU(0.2, True))
        self.t = T(img_size=self.img_size, patch_size=1, in_chans=34, embed_dim=self.embed_dim, depths=[2, 4], num_heads=[8, 8], window_size=8)
        self.visual_corresponding_name = {}
        init_weights(self.conv, self.u, self.conv1)
        init_w(self.t)
        self.visual_corresponding_name['gt'] = 'result'
        self.visual_names = ['gt', 'result']

    def forward(self, gt, rgb, lms):
        '''
        :param pan:
        :param ms:
        :return:
        '''
        self.rgb = rgb
        self.gt = gt
        self.lms = lms
        xt = torch.cat((self.lms, self.rgb), 1)  # Bx34X64x64
        w_out = self.t(xt)
        u_out = self.u(w_out)
        self.result = self.conv1(u_out) + self.lms

        return self.result

    def name(self):
        return ' net'

    def train_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['lrhsi'].cuda(), \
                           batch['rgb'].cuda()
        # x = torch.cat((up, msi), 1)
        sr = self(gt, msi, up)
        loss = self.criterion(sr, gt, *args, **kwargs)
        log_vars = {}
        with torch.no_grad():
            metrics = analysis_accu(gt, sr, 4, choices=4)
            log_vars.update(metrics)

        return {'loss': loss , 'log_vars': log_vars}

    def eval_step(self, batch, *args, **kwargs):
        gt, up, hsi, msi = batch['gt'].cuda(), \
                           batch['up'].cuda(), \
                           batch['up'].cuda(), \
                           batch['rgb'].cuda()
        # batch['lrhsi'].cuda(), \
        print(gt.shape)
        print(up.shape)
        print(hsi.shape)
        print(msi.shape)
        # print(msi.shape)
        sr1 = self.forward(gt, msi, up)
        # x = torch.cat((up, msi), 1)
        # sr1 = self.forward_chop(x)
        with torch.no_grad():
            metrics = analysis_accu(gt[0].permute(1, 2, 0), sr1[0].permute(1, 2, 0), 4)
            metrics.update(metrics)


        return sr1, metrics

    def set_metrics(self, criterion, rgb_range=1.0):
        self.rgb_range = rgb_range
        self.criterion = criterion


def build(args):
    scheduler = None
    scale = 2
    mode = "one"
    g_ssim = SSIM(size_average=True)
    loss1 = nn.L1Loss().cuda()
    loss2 = g_ssim.cuda()
    weight_dict = {'Loss': 1, 'ssim_loss': 0.1}
    losses = {'Loss': loss1, 'ssim_loss':loss2}
    criterion = SetCriterion(losses, weight_dict)
    model = Swinnet(args).cuda()
    num_params = 0
    for param in Swinnet(args).parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('Swinnet', num_params / 1e6))
    model.set_metrics(criterion)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  ## optimizer 1: Adam

    return model, criterion, optimizer, scheduler