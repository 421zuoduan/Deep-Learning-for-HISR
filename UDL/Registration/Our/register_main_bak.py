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
from torch import nn
# from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List, Optional
from torch import Tensor
# needed due to empty tensor bug in pytorch and torchvision 0.5
import torchvision
from utils.logger import create_logger, log_string
# from util.misc import is_main_process
from utils.logger_info import MetricLogger, SmoothedValue

# from pytorch_msssim.pytorch_msssim import SSIM
from cal_ssim import SSIM

import datetime
import register_dataset
from model import VxmDense
from mrband_model import DiCNN as mrband_model
import register_losses as rl
import matplotlib.pyplot as plt
# import visual as vis
# import plot_qv as qv

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


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.bcmsl = BCMSLoss().cuda()
        self.loss_dicts = {}
        # empty_weight = torch.ones(self.num_classes + 1)
        # empty_weight[-1] = self.eos_coef
        # self.register_buffer('empty_weight', empty_weight)

    def forward(self, outputs, targets, sr_loss='l1'):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        # loss_dicts = {}

        for loss_dict in self.losses.items():
            k, loss = loss_dict
            if len(outputs) > 1:
                if sr_loss == k:
                    self.loss_dicts.update({k: loss(outputs[0], targets[0])})
                else:
                # losses.update(self.get_loss(loss, outputs, targets))
                    self.loss_dicts.update({k: loss(outputs[1], targets[1])})
            else:
                self.loss_dicts.update({k: loss(outputs[0], targets[0])})
        return self.loss_dicts


##########################################################################################
# arch
##########################################################################################


def build(args):
    device = torch.device(args.device)

    registration = VxmDense(inshape=[64, 64])

    model = mrband_model(registration)
    log_string(model)

    weight_dict = {'lcc': 1, 'grad': 1, 'dist': 1, 'l1': 1}
    losses = {'lcc': rl.LCC([args.window] * 2).cuda(),
              'dist': rl.DistLossClip(clip_flag=False, inter=True).cuda(),
              'grad': rl.Bend_Penalty().cuda(),  # loss.Grad('l2').cuda(),
              'l1': nn.L1Loss(reduction='mean').cuda()
             }
    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=losses)
    criterion.to(device)

    return model, criterion


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch):
    # plt.ion()
    # f, axarr = plt.subplots(1, 3)

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 1

    # for batch in data_loader:
    for batch in metric_logger.log_every(data_loader, print_freq, header):
        # [{k: v.to(device) for k, v in t.items()} for t in targets]

        gt = batch[0].cuda(non_blocking=True)
        lms = batch[1].cuda(non_blocking=True)
        ms = batch[2].cuda(non_blocking=True)
        pan = batch[3].cuda(non_blocking=True)

        outputs, r_outputs = model(pan, lms)
        loss_dicts = criterion([outputs, r_outputs], [gt, pan])
        weight_dict = criterion.weight_dict
        losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)


        # vis.visualize_stn(axarr, lms, outputs, pan)


        # mse_loss = loss_dicts['mse_loss']
        # loss, w_1, w_2, w_s, w_bce, ssim_m

        # with torch.no_grad():
        #     metric_logger.update(psnr=PSNR(None, None, mse_loss, False))

        optimizer.zero_grad()
        losses.backward()
        # if max_norm > 0:
        #     torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        loss_dicts['Loss'] = losses
        metric_logger.update(**loss_dicts)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    log_string("Averaged stats: {}".format(metric_logger))
    # plt.ioff()
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
        # self.bcmsl = BCMSLoss().cuda()  # torch.nn.L1Loss().cuda()

    def run(self, train_sampler, train_loader, model, criterion, optimizer, scale, val_loader, scheduler):
        # global best_prec1
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        start_time = time.time()
        for epoch in range(self.args.start_epoch, self.args.epochs):
            # if self.args.distributed and not self.args.DALI:
            #     train_sampler.set_epoch(epoch)
            # adjust_learning_rate(optimizer, epoch)
            try:
                epoch_time = datetime.datetime.now()
                train_loss = train_one_epoch(model, criterion, train_loader, optimizer, self.args.device,
                                             epoch)
            except StopIteration:
                train_loader = self.sess.get_dataloader('train')
                train_loss = train_one_epoch(model, criterion, train_loader, optimizer, self.args.device,
                                             epoch)

            val_loss = self.validate_framework(val_loader, model, criterion, epoch)

            # if self.args.lr_scheduler and scheduler is not None:
            #     scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
            val_loss = val_loss['Loss']
            is_best = val_loss < self.args.best_prec1
            self.args.best_prec1 = min(val_loss, self.args.best_prec1)

            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': self.args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': self.args.best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, self.args.model_save_dir, is_best)

            if epoch % self.args.print_freq * 10 == 0 or is_best:
                if is_best:
                    self.args.best_epoch = epoch
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': self.args.arch,
                    'model': model,
                    'state_dict': model.state_dict(),
                    'best_loss': self.args.best_prec1,
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
        plt.ion()
        metric_logger = MetricLogger(delimiter="  ")
        header = 'TestEpoch: [{0}/{1}]'.format(epoch, self.args.epochs)
        # switch to evaluate mode
        model.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch in metric_logger.log_every(val_loader, 1, header):
            gt = batch[0].cuda(non_blocking=True)
            lms = batch[1].cuda(non_blocking=True)
            ms = batch[2].cuda(non_blocking=True)
            pan = batch[3].cuda(non_blocking=True)

            outputs, r_outputs = model(pan, lms)#, registration=True)

            # qv.FlowField.flow([flow[0, ...]])
            # plt.pause(1)

            loss_dicts = criterion([outputs, r_outputs], [gt, pan])

            weight_dict = criterion.weight_dict
            losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
            loss_dicts['Loss'] = losses

            metric_logger.update(**loss_dicts)
            # metric_logger.update(psnr=PSNR(None, None, loss_dicts['mse_loss'], False))
        log_string("Averaged stats: {}".format(metric_logger))
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        plt.ioff()
        return stats


if __name__ == "__main__":

    import numpy as np
    import argparse
    import random
    from torch.backends import cudnn

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # * Logger
    parser.add_argument('--out_dir', metavar='DIR', default='./results',
                        help='path to save model')
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='Vmx')

    parser.add_argument('--lr', default=1e-4, type=float)
    # parser.add_argument('--lr_backbone', default=1e-5, type=float)
    # parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('-b', '--batch-size', default=128, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--epochs', default=5000, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument('--resume',
                        default='./results/wv3/Vmx/Unet/model_2021-05-18-20-34/amp_model_best.pth.tar',
                        type=str, metavar='PATH',
                        # results/100L/APUnet/derain_large_V2/model_2021-04-06-23-54/487.pth.tar
                        help='path to latest checkpoint (default: none)')
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned', 'None'),
                        help="Type of positional embedding to use on top of the image features")

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('-w', '--window', default=9, type=int,
                        help='window size, in the LCC loss')

    args = parser.parse_args()
    args.experimental_desc = "Unet"
    args.dataset = "wv3"

    sess = register_dataset.Session(args)

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
    # param_dicts = [
    #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)

    ##################################################
    if args.resume:
        if os.path.isfile(args.resume):
            log_string("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = args.best_epoch = checkpoint['epoch']
            # best_prec1 = checkpoint['best_prec1']
            try:
                args.best_epoch = checkpoint['best_epoch']
            except:
                args.best_epoch = 0
            args.best_prec1 = checkpoint['best_loss']
            model.load_state_dict(checkpoint['state_dict'])
            if hasattr(checkpoint, 'optimizer'):
                optimizer.load_state_dict(checkpoint['optimizer'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr
            log_string("=> loaded checkpoint '{}' (epoch {})"
                       .format(args.resume, checkpoint['epoch']))
        else:
            log_string("=> no checkpoint found at '{}'".format(args.resume))

    ##################################################
    args.best_prec1 = 10000
    args.best_prec5 = 0
    args.best_epoch = 0
    train_loader = sess.get_dataloader('../../training_data/train.h5')
    val_loader = sess.get_test_dataloader('../../training_data/valid.h5')

    # train_loader = iter(list(train_loader))

    # for epoch in range(args.start_epoch, args.epochs):
    #     train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm)
    runner = EpochRunner(args, sess, args.experimental_desc)
    runner.run(None, train_loader, model, criterion, optimizer, None, val_loader, None)
