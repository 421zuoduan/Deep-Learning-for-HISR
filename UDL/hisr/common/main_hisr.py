"""
Backbone modules.
"""
import copy
import math
import os
import shutil
import time
import warnings
from collections import OrderedDict
import datetime
import imageio
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch import nn
from typing import Dict, List
from UDL.Basis.auxiliary import MetricLogger, SmoothedValue, set_random_seed, get_root_logger
from UDL.Basis.framework import model_amp, get_grad_norm, set_weight_decay, load_checkpoint, save_checkpoint
from UDL.Basis.dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
import torch.multiprocessing as mp
from UDL.hisr.common.hisr_dataset import HISRSession as DataSession
from torch.utils.tensorboard import SummaryWriter
from logging import info as log_string
import numpy as np
import scipy.io as sio

try:
    from apex import amp
except:
    print(
        "you don't install apex. [Optional] Please install apex from https://www.github.com/nvidia/apex or use pytorch1.6+.")


################################################################################
# framework
################################################################################
class EpochRunner():

    def __init__(self, args):
        self.args = args
        if args.use_log:
            _, out_dir, model_save_dir, tfb_dir = get_root_logger(None, args, args.experimental_desc)#, dist_print=args.global_rank)
            self.args.out_dir = out_dir
            self.args.model_save_dir = model_save_dir
            self.args.tfb_dir = tfb_dir
        # self.sess = sess
        self.metric_logger = MetricLogger(delimiter="  ", dist_print=args.global_rank)

    def best_record(self, train_stats, metrics):
        args = self.args
        if metrics == 'max':
            val_metric = train_stats['PSNR']
            val_loss = train_stats['Loss']
            is_best = val_metric > args.best_prec1
            args.best_prec1 = max(val_metric, args.best_prec1)
        else:
            val_loss = train_stats['Loss']
            is_best = val_loss < args.best_prec1
            args.best_prec1 = min(val_loss, args.best_prec1)

        return is_best, val_loss

    def train_one_epoch(self, args, model, criterion, data_loader, optimizer, device, epoch, scaler):
        reg = args.reg
        model.train()
        header = 'Epoch: [{}]'.format(epoch)
        print_freq = len(data_loader) if args.print_freq <= 0 else args.print_freq
        metric_logger = self.metric_logger
        for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
            loss, log_vars = model(batch) #output
            loss_dicts, weight_dict = loss
            # weight_dict = criterion.weight_dict
            # losses = loss_dicts['reg_loss']
            # if reg and 'Loss' in weight_dict:
            #     weight_dict['reg_loss'] = weight_dict.pop('Loss')

            loss = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
            losses = loss / args.accumulated_step
            model.backward(optimizer, losses, scaler)
            if args.clip_max_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
                # print(get_grad_norm(model.parameters()))
            else:
                grad_norm = get_grad_norm(model.parameters())

            if idx % args.accumulated_step == 0:
                optimizer.step()
                optimizer.zero_grad()

            # torch.cuda.synchronize()
            # loss_dicts['reg_loss'] = losses
            metric_logger.update(**loss_dicts)
            metric_logger.update(lr=optimizer.param_groups[0]["lr"])
            metric_logger.update(grad_norm=grad_norm)
            metric_logger.update_dict(log_vars)

        # gather the stats from all processes
        # metric_logger.synchronize_between_processes()

        metrics = {k: meter.avg for k, meter in metric_logger.meters.items()}

        if args.global_rank == 0:
            log_string("Averaged stats: {}".format(metric_logger))
            if args.use_tb:
                tfb_metrics = metrics.copy()
                tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'time'])
                args.train_writer.add_scalar(tfb_metrics, epoch)
                args.train_writer.flush()

        # plt.ioff()
        # 解耦，用于在main中调整日志
        return metrics

    def eval(self, eval_loader, model, criterion, eval_sampler):
        if self.args.distributed:
            eval_sampler.set_epoch(0)
        print(self.args.distributed)
        model = dist_train_v1(self.args, model)
        model, _ = load_checkpoint(self.args, model, None)
        val_loss = self.eval_framework(eval_loader, model, criterion)

    def run(self, train_loader, model, criterion, optimizer, val_loader, scheduler, **kwargs):

        args = self.args
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        model = model_amp(args, model, criterion, args.reg)
        optimizer, scaler = model.apex_initialize(optimizer)
        model.dist_train()
        model, optimizer = load_checkpoint(args, model, optimizer)
        if args.start_epoch >= 1:
            args.epochs += 1
        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if self.args.distributed:
                kwargs.get('train_sampler').set_epoch(epoch)
            epoch_time = datetime.datetime.now()
            train_stats = self.train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
                                          epoch, scaler)

            # val_stats = self.validate_framework(val_loader, model, criterion, epoch)

            # if self.args.lr_scheduler and scheduler is not None:
            #     scheduler.step(epoch)

            # remember best prec@1 and save checkpoint

            is_best, val_loss = self.best_record(train_stats, args.metrics)

            if epoch % args.print_freq == 0 or is_best:
                if is_best:
                    args.best_epoch = epoch
                if args.global_rank == 0:  # dist.get_rank() == 0
                    save_checkpoint({
                        'epoch': epoch,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric': args.best_prec1,
                        'loss': val_loss,
                        'best_epoch': args.best_epoch,
                        'amp': amp.state_dict() if args.amp_opt_level != 'O0' else None,
                        'optimizer': optimizer.state_dict()
                    }, args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")
            if args.global_rank == 0:
                log_string(' * Best training metrics so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                    loss=args.best_prec1, best_epoch=args.best_epoch))

                total_time = time.time() - start_time
                total_time_str = str(datetime.timedelta(seconds=int(total_time)))
                # log_string("one epoch time: {}".format(
                #     datetime.datetime.now() - epoch_time))
                log_string('Training time {}'.format(total_time_str))

    # @torch.no_grad()
    # def validate_framework(self, val_loader, model, criterion, epoch=0):
    #     args = self.args
    #     metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
    #     header = 'TestEpoch: [{0}/{1}]'.format(epoch, args.epochs)
    #     # switch to evaluate mode
    #     model.eval()
    #     # for iteration, batch in enumerate(val_loader, 1):
    #     for batch, idx in metric_logger.log_every(val_loader, 1, header):
    #         index = batch['index']
    #         samples = batch['O'].to(args.device, non_blocking=True)
    #         gt = batch['B'].to(args.device, non_blocking=True)
    #         # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
    #
    #         outputs, loss_dicts = model(sub_mean(samples), sub_mean(gt), index)
    #         # loss_dicts = criterion(outputs, gt)
    #
    #         weight_dict = criterion.weight_dict
    #         losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)
    #
    #         loss_dicts['Loss'] = losses
    #         # pred_np = add_mean(outputs.cpu().detach().numpy())
    #         # pred_np = quantize(pred_np, 255)
    #         # gt = gt.cpu().numpy() * 255
    #         # psnr = calc_psnr(pred_np, gt, 4, 255.0)
    #         psnr = reduce_mean(psnr_v2(add_mean(outputs), gt * 255.0, 4, 255.0))
    #         metric_logger.update(**loss_dicts)
    #         metric_logger.update(
    #             psnr=psnr)  # reduce_mean(psnr_t(outputs, gt)))#PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))
    #
    #     # metric_logger.synchronize_between_processes()
    #     stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
    #
    #     if args.global_rank == 0:
    #         log_string("[{}] Averaged stats: {}".format(epoch, metric_logger))
    #         if args.use_tb:
    #             tfb_metrics = stats.copy()
    #             tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'bce_loss', 'time'])
    #             args.test_writer.add_scalar(tfb_metrics, epoch)
    #             args.test_writer.flush()
    #
    #     # stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
    #     return stats  # stats


    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}]'.format(args.start_epoch)
        saved_path = f"./my_model_results/{args.name}/{args.arch}/{args.dataset}"
        os.makedirs(saved_path, exist_ok=True)
        # switch to evaluate mode
        model.eval()
        if self.args.dataset == 'cave_x4':
            sr = torch.zeros((11, 31, 512, 512))
            # sr = torch.zeros((11, 31, 512, 512))
            save_name = saved_path + '/' + args.name + '_cave' + str(args.test_epoch) + '.mat'
        elif self.args.dataset == 'cave_x8':
            sr = torch.zeros((11, 31, 512, 512))
            save_name = saved_path + '/' + 'cave11_x8-' + args.name + str(args.test_epoch) + '.mat'
        elif self.args.dataset == 'harvard_x4':
            sr = torch.zeros((10, 31, 1000, 1000))
            # sr = torch.zeros(10, 31, 1000, 1000)
            save_name = saved_path + '/' + args.name + '_harvard_x4' + str(args.test_epoch) + '.mat'
        elif self.args.dataset == 'harvard_x8':
            # sr = torch.zeros((250, 31, 200, 200))
            sr = torch.zeros(10, 31, 1000, 1000)
            save_name = saved_path + '/' + args.name + '_harvard200_x8' + str(args.test_epoch) + '.mat'
        elif self.args.dataset == 'Chikusei_x4':
            sr = torch.zeros(6, 128, 680, 680)
            save_name = saved_path + '/' + args.name + '_Chikusei' + str(args.test_epoch) + '.mat'
        elif self.args.dataset == 'pavia_x4':
            # sr = torch.zeros((250, 31, 200, 200))
            sr = torch.zeros(4, 92, 256, 256)
            save_name = saved_path + '/' + args.name + '_pavia' + str(args.test_epoch) + '.mat'


        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            if args.distributed:
                sr1, metrics = model.module.eval_step(batch)
            else:
                sr1, metrics = model.eval_step(batch)

            # padding
            sr[idx-1] = sr1.cpu()
            metric_logger.update_dict(metrics)

        sr = sr.permute(0, 2, 3, 1).numpy()

        sio.savemat(save_name, {'output': sr[:, :, :, :]})
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}

        return stats  # stats


def main(args):
    sess = DataSession(args)
    # torch.autograd.set_detect_anomaly(True)
    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        args.distributed = False
        args.global_rank = args.local_rank
        runner = EpochRunner(args)
    else:
        args.distributed = True
        init_dist(args.launcher, args)
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        args.global_rank = rank
        # 多机多卡
        args.local_rank = local_rank
        runner = EpochRunner(args)
        print(f"[init] == args local rank: {args.local_rank}, local rank: {local_rank}, global rank: {rank} ==")

    ##################################################
    if args.use_tb:
        if args.tfb_dir is None:
            args = runner.args
        if not args.eval:
            args.train_writer = SummaryWriter(args.tfb_dir + '/train')
            args.test_writer = SummaryWriter(args.tfb_dir + '/test')

    # device = torch.device("cuda", args.local_rank)
    torch.cuda.set_device(args.local_rank)
    model, criterion, optimizer, scheduler = args.builder(args)
    # model.to(device)
    model.cuda(args.local_rank)

    ##################################################
    if args.eval:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.dataset, args.distributed)
        runner.eval(eval_loader, model, criterion, eval_sampler)

    else:
        train_loader, train_sampler = sess.get_dataloader(args.dataset, args.distributed)
        # val_loader, val_sampler = sess.get_test_dataloader(args.dataset, args.distributed)

        if args.once_epoch:
            train_loader = iter(list(train_loader))

        runner.run(train_loader, model, criterion, optimizer, None, scheduler=scheduler, train_sampler=train_sampler)

        if args.use_tb and not args.eval:
            args.train_writer.close()
            args.test_writer.close()


if __name__ == "__main__":
    # FIXME 引发日志错误
    ...
