# GPL License
# Copyright (C) 2021 , UESTC
# All Rights Reserved 
#
# @Time    : 2022/1/22 14:43
# @Author  : Xiao Wu
# @reference: 
#
"""
Backbone modules.
"""
from UDL.sisr.stereo.sr_data import sisrSession as DataSession
# from mmcv.utils import print_log as pr
from logging import info as log_string
# from UDL.mmcls.compared.mmclassification.mmcls.datasets import build_dataloader, build_dataset
# from UDL.mmcls.compared.mmclassification.mmcls.models import build_classifier
from UDL.Basis.auxiliary import MetricLogger, SmoothedValue, set_random_seed, get_root_logger
from UDL.Basis.framework import model_amp, get_grad_norm, set_weight_decay, load_checkpoint, save_checkpoint
from UDL.Basis.dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
import logging
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
import torch.multiprocessing as mp
from torch import optim
from torch.utils.tensorboard import SummaryWriter

import numpy as np

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
            val_metric = train_stats['psnr']
            val_loss = train_stats['loss_SR']
            is_best = val_metric > args.best_prec1
            args.best_prec1 = max(val_metric, args.best_prec1)
        else:
            val_loss = train_stats['loss_SR']
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
            # weight_dict = criterion.weight_dict
            # losses = loss_dicts['reg_loss']
            # if reg and 'Loss' in weight_dict:
            #     weight_dict['reg_loss'] = weight_dict.pop('Loss')

            # losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)

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
            # metric_logger.update(**loss_dicts)
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


    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}]'.format(args.start_epoch)
        saved_path = f"./my_model_results/{args.arch}/{args.dataset}"
        os.makedirs(saved_path, exist_ok=True)
        # switch to evaluate mode
        model.eval()
        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            if args.distributed:
                metrics = model.module.eval_step(batch)
            else:
                metrics = model.eval_step(batch, saved_path, is_training=0)

            metric_logger.update_dict(metrics)

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
    model, criterion, optimizer, scheduler = args.builder(args)#build_model(args)
    # model.to(device)
    model.cuda(args.local_rank)

    ##################################################
    if args.eval:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.dataset, args.distributed)
        # eval_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        # eval_loader = build_dataloader(
        #     eval_dataset,
        #     samples_per_gpu=cfg.data.samples_per_gpu,
        #     workers_per_gpu=cfg.data.workers_per_gpu,
        #     dist=args.distributed,
        #     shuffle=False,
        #     round_up=True)
        runner.eval(eval_loader, model, criterion, eval_sampler)

    else:
        # train_sampler = cfg.data.get('sampler', None)
        # datasets = [build_dataset(cfg.data.train)]
        # train_loader = [
        #     build_dataloader(
        #         ds,
        #         cfg.data.samples_per_gpu,
        #         cfg.data.workers_per_gpu,
        #         # cfg.gpus will be ignored if distributed
        #         num_gpus=len(cfg.gpu_ids),
        #         dist=args.distributed,
        #         round_up=True,
        #         seed=cfg.seed,
        #         sampler_cfg=train_sampler) for ds in datasets
        # ][0]
        train_loader, train_sampler = sess.get_dataloader(args.dataset, args.distributed)
        # val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)

        if args.once_epoch:
            train_loader = iter(list(train_loader))

        runner.run(train_loader, model, criterion, optimizer, None, scheduler=scheduler, train_sampler=train_sampler)

        if args.use_tb and not args.eval:
            args.train_writer.close()
            args.test_writer.close()


def build_model(cfg):
    scheduler = None
    criterion = None
    model = build_classifier(cfg.model)
    model.init_weights()
    optimizer = optim.Adam(model.parameters(), lr=1e-1, weight_decay=0)  ## optimizer 1: Adam
    return model, criterion, optimizer, scheduler


if __name__ == "__main__":
    from UDL.mmcls.compared.mmclassification.tools.train import parse_args
    from UDL.Basis.config import Config
    args = parse_args()
    args.use_log = True
    args.use_tb = False
    args.eval = False
    args.once_epoch = False
    args.mode = None
    args.reg = False
    args.amp = None
    args.epochs = 200
    args.resume = None
    args.start_epoch = 1
    args.print_freq = 100
    args.experimental_desc = 'Test'
    args.out_dir = "./results/"
    args.arch = "resnet"
    args.dataset = "cifar10"
    args.log_dir = "log"
    args.clip_max_norm = -1
    args.accumulated_step = 1
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.merge_args2cfg(args)
    log_string(cfg.pretty_text)
    main(cfg)
