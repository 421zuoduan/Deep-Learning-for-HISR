"""
Backbone modules.
"""
import copy
import math
import os
import re
import shutil
import time
from collections import OrderedDict

import imageio
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch import nn
# from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

# from detr.models.transformer import build_transformer
# from util.misc import is_main_process
# from position_encoding import build_position_encoding
from utils.utils import MetricLogger, SmoothedValue, set_random_seed
# from pytorch_msssim.pytorch_msssim import SSIM
from cal_ssim import SSIM
from utils.logger import create_logger, log_string
import datetime
from framework import model_amp, get_grad_norm, set_weight_decay
from apex import amp
from dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
import torch.multiprocessing as mp
from common.derain_dataset import derainSession, PSNR_ycbcr
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

def partial_load_checkpoint(prefix_ckpt, state_dict, amp, dismatch_list = []):
    pretrained_dict = {}
    # dismatch_list = ['dwconv']
    if amp is not None:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            k = '.'.join(['amp', k])
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})
    else:
        for module in state_dict.items():
            # pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            if prefix_ckpt == 'IPT_pretrain.pt':
                if args.distributed:
                    k = '.'.join(['module', 'model', k])
                else:
                    k = '.'.join(['model', 'model', k])
            elif prefix_ckpt == 'IPT_derain.pt':
                if args.eval:
                    k = '.'.join(['model', k])
                else:
                    if args.distributed:
                        k = '.'.join(['module', 'model', k])
                        # ...
                    else:
                        ...
                        # k = '.'.join(['model', 'model', k])
            k = k.split('.')
            k = '.'.join(k[1:])
                # k = '.'.join(['model', 'model', k])
            # if args.eval:
            #     k = k.split('.')
            #     k = '.'.join(k[1:])
            # k.__delitem__(0)
            # k = '.'.join(k[1:])
            # k = '.'.join(['model', k])
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})

    return pretrained_dict

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
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x #/ 255

def add_mean(x):
    x = x #* 255
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range):
    """metrics"""
    hr = np.float32(hr)
    sr = np.float32(sr)
    diff = (sr - hr) / rgb_range
    #.reshape((1, 1, 3)) / 256#
    gray_coeffs = np.array([65.738, 129.057, 25.064]).reshape((1, 3, 1, 1)) / 256
    diff = np.multiply(diff, gray_coeffs).sum(1)#(1)
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
        # valid = diff[shave:-shave, shave:-shave, ...]
    # mse = np.mean(np.mean(pow(valid, 2), axis=[1, 2, 3]), axis=0)
    mse = np.mean(pow(valid, 2))
    if mse == 0:
        return 100
    try:
        psnr = -10 * math.log10(mse)
    except Exception:
        print(mse)

    return psnr


def rgb2ycbcr(img, y_only=True):
    """metrics"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt

psnr_y = PSNR_ycbcr()
g_ssim = SSIM(size_average=False, data_range=255.)

#############################################################################
# Backbone
#############################################################################


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
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute all the requested losses
        # loss_dicts = {}

        for k in self.losses.keys():#.items():
            # k, loss = loss_dict
            if k == 'Loss':
                loss = self.losses[k]

                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts


##########################################################################################
# arch
##########################################################################################
import model


def build(args):
    global model

    device = torch.device(args.device)

    model = model.Model(args, args.resume)

    if args.global_rank == 0:
        log_string(model)
    weight_dict = {'Loss': 1}
    losses = {'Loss': nn.L1Loss().cuda()}


    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=losses)
    criterion.to(device)

    return model, criterion


# plt.ion()
# f, axarr = plt.subplots(1, 3)
# fig, axes = plt.subplots(ncols=2, nrows=2)
def train_one_epoch(args, model, criterion, data_loader, optimizer, device, epoch, scaler):

    model.train()
    criterion.train()
    metric_logger = MetricLogger(delimiter="  ", dist_print=args.global_rank)
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

    for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
        for idx_scale, scale in enumerate(args.scale):
            # index = batch['index']
            samples = batch['O'].to(device)
            gt = batch['B'].to(device)
            # plt.imshow(samples[0, ...].permute(1, 2, 0).cpu()/255)
            # plt.show()
            # print(samples.shape, gt.shape)
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
            # outputs, loss_dicts = model(samples, gt, idx_scale)

            outputs = model(samples, idx_scale)
            loss_dicts = criterion(outputs, gt)
            if hasattr(model, 'ddp_step'):
                model.ddp_step(loss_dicts)

            # loss_dicts = model.model.ddp_step(loss_dicts)
            losses = loss_dicts['Loss']


            # weight_dict = criterion.weight_dict
            # losses = loss_dicts['contrast']
            # show_maps(axes, samples, gt, outputs)
            # plt.pause(0.4)
            # if epoch <= 946:
            #     losses = loss_dicts['Loss']
            # else:
            #     losses = loss_dicts['contrast']#sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
            losses = losses / args.accumulated_step

            losses.backward()
            #model.backward(optimizer, losses, scaler)
            if args.clip_max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                # print(get_grad_norm(model.parameters()))
            else:
                grad_norm = get_grad_norm(model.parameters())

            if (idx + 1) % args.accumulated_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            loss_dicts['Loss'] = losses
            metric_logger.update(**loss_dicts)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            #reduce_mean(psnr_t(outputs, gt)))#
            # p = reduce_mean(psnr_t(outputs, gt))
            # print(p)
            # if args.global_rank == 0:
            # pred_np = quantize(outputs, 255)
            # psnr = calc_psnr(pred_np.cpu().detach().numpy(), gt.cpu().numpy(), 4, 255.0)
            # print("current psnr: ", psnr)
            metric_logger.update(psnr=reduce_mean(psnr_y(outputs, gt, 4, 255.0)))  # PSNR(outputs.cpu().detach().numpy(), gt.cpu().numpy()))
            metric_logger.update(grad_norm=grad_norm)

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()

    metrics = {k: meter.avg for k, meter in metric_logger.meters.items()}


    if args.global_rank == 0:
        log_string("Averaged stats: {}".format(metric_logger))
        if args.use_tb:
            tfb_metrics = metrics.copy()
            tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'bce_loss', 'time'])
            args.train_writer.add_scalar(tfb_metrics, epoch)
            args.train_writer.flush()
    # plt.ioff()
    # 解耦，用于在main中调整日志
    return metrics#{k: meter.avg for k, meter in metric_logger.meters.items()}


################################################################################
# framework
################################################################################
def save_checkpoint(state, model_save_dir, is_best, filename='amp_checkpoint.pth.tar'):
    filename = os.path.join(model_save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_dir, 'amp_model_best.pth.tar'))

def load_checkpoint(args, model, optimizer, ignore_params=[]):
    global_rank = args.global_rank
    checkpoint = {}
    prefix_ckpt = args.resume.split('/')[-1]
    if args.resume:
        if os.path.isfile(args.resume):
            if args.distributed:
                dist.barrier()
            init_checkpoint = torch.load(args.resume, map_location='cpu')#f"cuda:{args.local_rank}"
            if init_checkpoint.get('state_dict') is None:
                checkpoint['state_dict'] = init_checkpoint
                del init_checkpoint
                torch.cuda.empty_cache()
            else:
                checkpoint = init_checkpoint
                print(checkpoint.keys())
            args.start_epoch = args.best_epoch = checkpoint.setdefault('epoch', 0)
            args.best_epoch = checkpoint.setdefault('best_epoch', 0)
            args.best_prec1 = checkpoint.setdefault('best_metric', 0)
            if args.amp is not None:
                print(checkpoint.keys())
                try:
                    amp.load_state_dict(checkpoint['amp'])
                except:
                    Warning("no loading amp_state_dict")
            # if ignore_params is not None:
            # if checkpoint.get('state_dict') is not None:
            #     ckpt = partial_load_checkpoint(checkpoint['state_dict'], args.amp, ignore_params)
            # else:
            #     print(checkpoint.keys())
            #     ckpt = partial_load_checkpoint(checkpoint, args.amp, ignore_params)
            # print(checkpoint['state_dict'].keys())
            ckpt = partial_load_checkpoint(prefix_ckpt, checkpoint['state_dict'], args.amp, ignore_params)
            # print([ name for name, param in model.named_parameters()])
            if args.global_rank == 0:
                print(ckpt.keys())
            if args.distributed:
                if args.eval:
                    x = model.module.load_state_dict(ckpt)#, strict=False)
                else:
                    # print("distributed", args.distributed)
                    x = model.load_state_dict(ckpt)#, strict=False)
                    # x = model.model.module.load_state_dict(ckpt)#, strict=False)
                    '''
                    model.module.load_state_dict(ckpt)
                    "model.tail.0.1.bias".  模型的
                    Unexpected key(s) in state_dict: "model.module.sub_mean.weight" 权重的
                    '''
            else:
                # print("single", args.distributed)
                x = model.load_state_dict(ckpt)#, strict=False)

            if global_rank == 0:
                print(x)
                log_string(f"=> loading checkpoint '{args.resume}'\n ignored_params: \n{ignore_params}")
            # else:
            #     if global_rank == 0:
            #         log_string(f"=> loading checkpoint '{args.resume}'")
            #     if checkpoint.get('state_dict') is None:
            #         model.load_state_dict(checkpoint)
            #     else:
            #         model.load_state_dict(checkpoint['state_dict'])
                # print(checkpoint['state_dict'].keys())
                # print(model.state_dict().keys())
            if optimizer is not None:
                if hasattr(checkpoint, 'optimizer'):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr
                if global_rank == 0:
                    log_string("=> loaded checkpoint '{}' (epoch {})"
                           .format(args.resume, checkpoint['epoch']))

            del checkpoint
            torch.cuda.empty_cache()

        else:
            if global_rank == 0:
                log_string("=> no checkpoint found at '{}'".format(args.resume))

        return model, optimizer

# def warp_tfb(func):



class EpochRunner():

    def __init__(self, args, sess):
        self.args = args
        out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc, dist_print=args.global_rank)
        self.args.out_dir = out_dir
        self.args.model_save_dir = model_save_dir
        self.args.tfb_dir = tfb_dir
        # self.tester = Tester(args)
        # self.std_criterion = torch.nn.MSELoss(reduction='mean').cuda()
        self.sess = sess

        # self.ssim = SSIM().cuda()
        # self.bcmsl = BCMSLoss().cuda()  # torch.nn.L1Loss().cuda()

    def run(self, model, criterion, train_loader, optimizer, val_loader, scheduler, **kwargs):
        args = self.args
        scaler = None
        #model = model_amp(args, model, criterion)
        #optimizer, scaler = model.apex_initialize(optimizer)
        #model.dist_train()
        model = dist_train_v1(self.args, model)
        _, optimizer = load_checkpoint(args, model, optimizer)
        start_time = time.time()
        # val_stats = self.validate_framework(val_loader, model, criterion, 0)
        for epoch in range(args.start_epoch, args.epochs):
            if self.args.distributed:
                kwargs.get('train_sampler').set_epoch(epoch)
            # try:
            #     epoch_time = datetime.datetime.now()
            #     train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
            #                                  epoch, scaler)
            # except:
            #     # train_loader = self.sess.get_dataloader('train', self.args.)
            #     train_loss = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
            #                                  epoch, scaler)

            # val_stats = self.validate_framework(val_loader, model, criterion, epoch)
            # epoch_time = datetime.datetime.now()
            train_stats = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
                                         epoch, scaler)

            # if self.args.lr_scheduler and scheduler is not None:
            #     scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
            loss = train_stats['Loss']
            psnr = train_stats['psnr']
            is_best = psnr > self.args.best_prec1
            self.args.best_prec1 = max(psnr, args.best_prec1)

            # save_checkpoint({
            #     'epoch': epoch + 1,
            #     'arch': self.args.arch,
            #     'state_dict': model.state_dict(),
            #     'best_prec1': self.args.best_prec1,
            #     'optimizer': optimizer.state_dict(),
            # }, self.args.model_save_dir, is_best)

            if epoch % args.print_freq == 0 or is_best:
                if is_best:
                    self.args.best_epoch = epoch
                if args.global_rank == 0: #dist.get_rank() == 0
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric': self.args.best_prec1,
                        'loss': loss,
                        'best_epoch': self.args.best_epoch,
                        'amp': amp.state_dict() if args.amp_opt_level != 'O0' else None,
                        'optimizer': optimizer.state_dict()
                    }, args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")
            if args.global_rank == 0:
                log_string(' * Best validation psnr so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                    loss=args.best_prec1, best_epoch=args.best_epoch))

                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                # log_string("one epoch time: {}".format(
                #     datetime.datetime.now() - epoch_time))
                log_string('Training time {}'.format(total_time_str))

    def eval(self, eval_loader, model, criterion, idx, eval_sampler):
        if self.args.distributed:
            eval_sampler.set_epoch(0)
        model = dist_train_v1(self.args, model)
        model, _ = load_checkpoint(self.args, model, None)
        val_loss = self.eval_framework(eval_loader, model, criterion, idx)

    # @torch.no_grad()
    # def eval_framework(self, eval_loader, model, criterion):
    #     args = self.args
    #     metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
    #     header = 'TestEpoch: [{0}]'.format(args.start_epoch)
    #     # switch to evaluate mode
    #     model.eval()
    #     # for iteration, batch in enumerate(val_loader, 1):
    #     for batch, idx in metric_logger.log_every(eval_loader, 1, header):
    #         for idx_scale, scale in enumerate(args.scale):
    #             # index = batch['index']
    #             samples = batch['O'].to(args.device, non_blocking=True)
    #             filename = batch['file_name']
    #             # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
    #
    #             outputs = model(samples, idx_scale)
    #             outputs = quantize(outputs, 255)
    #             normalized = outputs[0].mul(255 / args.rgb_range)
    #             tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
    #             imageio.imwrite(os.path.join("./my_train_derain", ''.join([filename[0][:-2], '.png'])),
    #                             tensor_cpu.numpy())
    #
    #             # pred_np = quantize(outputs.cpu().detach().numpy(), 255)
    #             # psnr = calc_psnr(pred_np, gt.cpu().numpy(), 4, 255.0)
    #             # metric_logger.update(**loss_dicts)
    #             # metric_logger.update(psnr=psnr)  # PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))
    #
    #     stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
    #     log_string("Averaged stats: {}".format(metric_logger))
    #
    #     return stats  # stats

    @torch.no_grad()
    def validate_framework(self, val_loader, model, criterion, epoch=0):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}/{1}]'.format(epoch, args.epochs)
        # switch to evaluate mode
        model.eval()
        criterion.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch, idx in metric_logger.log_every(val_loader, 1, header):
            for idx_scale, scale in enumerate(args.scale):
                # index = batch['index']
                samples = batch['O'].to(args.device, non_blocking=True)
                gt = batch['B'].to(args.device, non_blocking=True)
                # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])

                outputs, loss_dicts = model(samples, gt, idx_scale=idx_scale)
                # loss_dicts = criterion(outputs, gt)

                weight_dict = criterion.weight_dict
                losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)

                loss_dicts['Loss'] = losses


                pred_np = quantize(outputs, 255)
                psnr = calc_psnr(pred_np.cpu().detach().numpy(), gt.cpu().numpy(), 4, 255.0)
                metric_logger.update(**loss_dicts)
                metric_logger.update(psnr=psnr)#PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}


        if args.global_rank == 0:
            log_string("Averaged stats: {}".format(metric_logger))
            if args.use_tb:
                tfb_metrics = stats.copy()
                tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'bce_loss', 'time'])
                args.test_writer.add_scalar(tfb_metrics, epoch)
                args.test_writer.flush()

        # stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        return stats#stats

    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion, idx_start=0):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'EvalEpoch: [{0}] '.format(args.start_epoch)
        saved_path = f"./my_model_results/{args.dataset}"
        # switch to evaluate mode
        model.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            if idx < idx_start:
                continue
            for idx_scale, scale in enumerate(args.scale):
                # index = batch['index']
                samples = batch['O'].to(args.device, non_blocking=True)
                gt = batch['B'].to(args.device, non_blocking=True)
                filename = batch['file_name']
                # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])

                outputs = model(samples, idx_scale=idx_scale)
                outputs = quantize(outputs, 255)
                normalized = outputs[0].mul(255 / args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                imageio.imwrite(os.path.join(saved_path, ''.join([filename[0], '.png'])),
                                tensor_cpu.numpy())
                # imageio.imwrite(os.path.join("./gt", ''.join([filename[0], '.png'])), gt[0].byte().permute(1, 2, 0).cpu().numpy())
                # pred_np = quantize(outputs.cpu().detach().numpy(), 255)
                # psnr = calc_psnr(outputs.cpu().numpy(), gt.cpu().numpy(), 4, 255.0)#[0].permute(1, 2, 0)
                psnr = reduce_mean(psnr_y(outputs, gt, 4, 255.0))
                ssim = reduce_mean(g_ssim(outputs, gt))
                # metric_logger.update(**loss_dicts)
                print(args.local_rank, filename)
                metric_logger.update(ssim=ssim)
                metric_logger.update(psnr=psnr)  # PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        log_string("Averaged stats: {}".format(metric_logger))

        return stats  # stats

def main(args):
    sess = derainSession(args)
    # torch.autograd.set_detect_anomaly(True)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
        args.global_rank = args.local_rank
        runner = EpochRunner(args, sess)

    else:
        args.distributed = True
        init_dist(args.launcher, args)
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        args.global_rank = rank
        # 多机多卡
        args.local_rank = local_rank
        runner = EpochRunner(args, sess)
        print(f"[init] == args local rank: {args.local_rank}, local rank: {local_rank}, global rank: {rank} ==")


        # _, world_size = get_dist_info()
    ##################################################
    if args.use_tb:
        if args.tfb_dir is None:
            args = runner.args
        args.train_writer = SummaryWriter(args.tfb_dir + '/train')
        args.test_writer = SummaryWriter(args.tfb_dir + '/test')




    # device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model, criterion = build(args)
    # model.to(device)
    model.cuda(args.local_rank)

    # SyncBN (https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html)
    #net = nn.SyncBatchNorm.convert_sync_batchnorm(net) if cfg.MODEL.SYNCBN else net
    # if rank == 0:
    #     # from torch.utils.collect_env import get_pretty_env_info
    #     # logger.debug(get_pretty_env_info())
    #     # logger.debug(net)
    #     logger.info("\n\n\n            =======  TRAINING  ======= \n\n")
    #     logger.info(utils.count_parameters(net))
    # if rank == 0:
    #     logger.info(
    #         f"ACCURACY: TOP1 {acc1:.3f}(BEST {best_acc1:.3f}) | TOP{cfg.TRAIN.TOPK} {acck:.3f} | SAVED {checkpoint_file}"
    #     )
    ##################################################



    if args.eval:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.dataset, args.distributed)
        runner.eval(eval_loader, model, criterion, args.idx, eval_sampler)
    else:
        train_loader, train_sampler = sess.get_dataloader(args.dataset, args.distributed)
        # val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)
        # model = dist_train_v1(args, model)

        #'out_proj.bias'
        # skip = {}
        # skip_keywords = {}
        # if hasattr(model, 'no_weight_decay'):
        #     skip = model.no_weight_decay()
        # if hasattr(model, 'no_weight_decay_keywords'):
        #     skip_keywords = model.no_weight_decay_keywords()
        # parameters = set_weight_decay(model, skip, skip_keywords)

        # skip = {'out_proj.bias'}
        # parameters = []
        # for module, (name, param) in zip(model.named_modules(), model.named_parameters()):
        #     print(module, name)
        #     if name.endswith("out_proj.bias") or (name in skip):
        #         del module[1].self_attn.out_proj.bias
        #         # # no_decay.append(param)
        #         # name = name.split('.')[:-1]
        #         # name = '.'.join(name)
        #         # if hasattr(model, name):
        #         #     delattr(model[name], "out_proj.bias")
        #         #     print(f"{name} will be ignored in model.parameters")
        #         # else:
        #         #     print(f"{name} don't be found")
        # print(hasattr(model, "model.body.encoder"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        if args.once_epoch:
            train_loader = iter(list(train_loader))
        runner.run(model, criterion, train_loader, optimizer, None, None, train_sampler=train_sampler)

        # runner.run(model, criterion, optimizer, train_loader, val_loader, None, train_sampler=train_sampler)
        if args.use_tb:
            args.train_writer.close()
            args.test_writer.close()

if __name__ == "__main__":
    import numpy as np
    from options import args
    torch.cuda.empty_cache()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # os.environ["RANK"] = "0"
    set_random_seed(args.seed)
    ##################################################
    args.best_prec1 = 0
    args.best_prec5 = 0
    args.best_epoch = 0
    args.nprocs = torch.cuda.device_count()

    # print(f"deviceCount: {args.nprocs}")
    # mp.spawn(main, nprocs=2, args=(args, ))
    main(args)
    #ssim: 0.9542 (0.9496577)  psnr: 41.9144 (39.9861841)
    #For Rain100L dataset PSNR: 39.884365 SSIM: 0.985824
    '''
    单卡：
    - => loaded checkpoint './IPT_derain.pt' (epoch 0)
- Epoch: [0]  [   0/1000]  eta: 2:11:00  lr: 0.000020  grad_norm: 111.851983  Loss: 2.7973 (2.7973220)  psnr: 29.6278 (29.6278483)  time: 7.8601  data: 6.2146  max mem: 4648MB
- Epoch: [0]  [  10/1000]  eta: 0:17:31  lr: 0.000020  grad_norm: 72.526479  Loss: 3.3214 (3.2206018)  psnr: 29.2451 (29.3811368)  time: 1.0623  data: 0.5651  max mem: 6025MB
- Epoch: [0]  [  20/1000]  eta: 0:12:02  lr: 0.000020  grad_norm: 31.362339  Loss: 2.1519 (3.3017122)  psnr: 33.8066 (29.5715870)  time: 0.7376  data: 0.2963  max mem: 6025MB
- Epoch: [0]  [  30/1000]  eta: 0:10:03  lr: 0.000020  grad_norm: 36.703368  Loss: 1.9776 (3.3433307)  psnr: 35.0481 (29.5546505)  time: 0.6222  data: 0.2008  max mem: 6025MB
- Epoch: [0]  [  40/1000]  eta: 0:09:00  lr: 0.000020  grad_norm: 48.001039  Loss: 4.0759 (3.3157656)  psnr: 26.0325 (29.4501023)  time: 0.5633  data: 0.1519  max mem: 6025MB
- Epoch: [0]  [  50/1000]  eta: 0:08:21  lr: 0.000020  grad_norm: 79.158277  Loss: 2.8677 (3.2713672)  psnr: 31.0762 (29.4811592)  time: 0.5277  data: 0.1221  max mem: 6025MB
- Epoch: [0]  [  60/1000]  eta: 0:07:53  lr: 0.000020  grad_norm: 18.949162  Loss: 2.9020 (3.2747439)  psnr: 29.2536 (29.5480094)  time: 0.5036  data: 0.1022  max mem: 6025MB
    远程单卡：out_proj_bias
    - Epoch: [0]  [   0/1000]  eta: 0:48:18  lr: 0.000020  grad_norm: 123.570264  Loss: 2.6918 (2.6918161)  psnr: 28.6546 (28.6545716)  time: 2.8980  data: 0.4334  max mem: 4648MB
- Epoch: [0]  [  10/1000]  eta: 0:40:40  lr: 0.000020  grad_norm: 49.049982  Loss: 2.2581 (3.1626153)  psnr: 32.9532 (29.6151108)  time: 2.4653  data: 0.0407  max mem: 6025MB
    远程多卡
    - Epoch: [0]  [  0/125]  eta: 0:10:17  lr: 0.000020  grad_norm: 112.966178  Loss: 3.7013 (3.7013183)  psnr: 27.1939 (27.1939448)  time: 4.9423  data: 2.0401  max mem: 5093MB
- Epoch: [0]  [ 10/125]  eta: 0:05:46  lr: 0.000020  grad_norm: 50.640930  Loss: 1.9305 (3.2243563)  psnr: 33.2093 (29.6829323)  time: 3.0148  data: 0.1877  max mem: 6460MB
- Epoch: [0]  [ 20/125]  eta: 0:05:03  lr: 0.000020  grad_norm: 51.402320  Loss: 4.0979 (3.6886477)  psnr: 26.6271 (28.4781113)  time: 2.8867  data: 0.0993  max mem: 6460MB
- Epoch: [0]  [ 30/125]  eta: 0:04:31  lr: 0.000020  grad_norm: 33.136749  Loss: 4.2247 (3.5404011)  psnr: 24.3839 (28.6188555)  time: 2.8620  data: 0.0680  max mem: 6460MB
- Epoch: [0]  [ 40/125]  eta: 0:04:03  lr: 0.000020  grad_norm: 16.217282  Loss: 2.4852 (3.4424340)  psnr: 30.6702 (28.8644178)  time: 2.8591  data: 0.0519  max mem: 6460MB
        
        
    derain.pt
    
    多卡
     Averaged stats: lr: 0.000020  grad_norm: 10.125028  Loss: 1.6898 (2.5043314)  psnr: 37.1024 (33.2370122)
    eval：正常
        
        
    
    '''