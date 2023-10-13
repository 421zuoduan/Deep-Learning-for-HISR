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
from UDL.hisr.HISR.PSRT_KAv1_noshuffle.psrt import *
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

class PSRTnet(PatchMergeModule):
    def __init__(self, args):
        super(PSRTnet, self).__init__()
        self.args = args
        self.img_size = 64
        self.in_channels = 31
        self.embed = 48
        self.conv = nn.Sequential(
            nn.Conv2d(self.embed, self.in_channels, 3, 1, 1), nn.LeakyReLU(0.2, True)
        )
        # self.w = Block(num=2, img_size=self.img_size, in_chans=34, embed_dim=32, head=8, win_size=2)
        self.w = Block(out_num=2, inside_num=3, img_size=self.img_size, in_chans=34, embed_dim=self.embed, head=8, win_size=8)
        self.visual_corresponding_name = {}
        init_weights(self.conv)
        init_w(self.w)
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
        _, _, H, W = xt.shape
        w_out = self.w(H, W, xt)
        self.result = self.conv(w_out) + self.lms

        return self.result

    def name(self):
        return ' PSRT'

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

    def eval_step(self, batch):
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
    losses = {'Loss': loss1, 'ssim_loss': loss2}
    criterion = SetCriterion(losses, weight_dict)
    model = PSRTnet(args).cuda()
    num_params = 0
    for param in PSRTnet(args).parameters():
        num_params += param.numel()
    print('[Network %s] Total number of parameters : %.3f M' % ('PSRT', num_params / 1e6))
    model.set_metrics(criterion)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)  ## optimizer 1: Adam

    return model, criterion, optimizer, scheduler