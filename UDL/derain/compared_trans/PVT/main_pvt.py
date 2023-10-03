"""
Backbone modules.
"""
import copy
import math
import os
import shutil
import time
from collections import OrderedDict

import imageio
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
from torch import nn
from typing import Dict, List
from utils.utils import MetricLogger, SmoothedValue, set_random_seed
# from pytorch_msssim.pytorch_msssim import SSIM
from cal_ssim import SSIM
from utils.logger import create_logger, log_string
import datetime
from san_pvt import Pvt as net
from framework import model_amp, get_grad_norm, set_weight_decay
from apex import amp
from dist_utils import init_dist, dist_train_v1, get_dist_info, reduce_mean
import torch.multiprocessing as mp
from common.derain_dataset import derainSession, calc_psnr, PSNR_ycbcr, add_mean, sub_mean, quantize
from torch.utils.tensorboard import SummaryWriter

def partial_load_checkpoint(state_dict, amp, dismatch_list = []):
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
            if args.eval is not None:
                k = k.split('.')
                k = '.'.join(k[1:])
            # k = '.'.join(['model', k])
            #
            # k.__delitem__(0)
            #
            # k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'ddp'])
            # k = '.'.join([k_item for k_item in k if k_item != 'module' and k_item != 'model' and k_item != 'ddp'])  #
            if all(m not in k for m in dismatch_list):
                # print(k)
                pretrained_dict.update({k: v})

    return pretrained_dict

psnr_v2 = PSNR_ycbcr().cuda()
g_ssim = SSIM(size_average=False, data_range=1)

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
def build(args):


    device = torch.device(args.device)

    model = net(
            patch_size=args.patch_size, in_channels=3, out_channels=3, patch_stride=[2, 2, 2, 2],
            hidden_dim=[32, 64, 128, 320, 512], num_heads=[1, 2, 4, 8], mlp_ratios=[8, 8, 4, 4],
            depths=[2, 2, 2, 2], sr_ratio=[1, 1, 1, 1])

    #造成日志无法使用
    if args.global_rank == 0:
        log_string(model)

    # weight_dict = {'L1Loss': 1}
    # losses = {'L1Loss': torch.nn.L1Loss(reduction='mean').cuda()}

    weight_dict = {'Loss': 1}
    losses = {'Loss': nn.L1Loss().cuda()}

    criterion = SetCriterion(weight_dict=weight_dict,
                             losses=losses)
    criterion.to(device)

    return model, criterion

# import matplotlib.pyplot as plt
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
    psnr_list = []
    for batch, idx in metric_logger.log_every(data_loader, print_freq, header):
        # index = batch['index']
        samples = batch['O'].to(device)
        gt = batch['B'].to(device)  # [{k: v.to(device) for k, v in t.items()} for t in targets]
        # samples = batch[0].to(device)
        # gt = batch[1].to(device)
        samples = sub_mean(samples)
        gt_y = sub_mean(gt)
        # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
        outputs, loss_dicts = model(samples, gt_y)
        # loss_dicts = model.model.ddp_step(loss_dicts)
        losses = loss_dicts['Loss']
        losses = losses / args.accumulated_step

        model.backward(optimizer, losses, scaler)
        if args.clip_max_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_max_norm)
            # print(get_grad_norm(model.parameters()))
        else:
            grad_norm = get_grad_norm(model.parameters())

        if (idx + 1) % args.accumulated_step == 0:
            optimizer.step()
            optimizer.zero_grad()

        # torch.cuda.synchronize()
        loss_dicts['Loss'] = losses
        metric_logger.update(**loss_dicts)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(psnr=reduce_mean(psnr_v2(add_mean(outputs), gt * 255.0, 4, 255.0))) #reduce_mean(psnr_v2(add_mean(outputs), gt, 4, 255.0))
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
    if args.resume:
        if os.path.isfile(args.resume):
            if args.distributed:
                dist.barrier()
            init_checkpoint = torch.load(args.resume, map_location=f"cuda:{args.local_rank}")
            if init_checkpoint.get('state_dict') is None:
                checkpoint['state_dict'] = init_checkpoint
                del init_checkpoint
                torch.cuda.empty_cache()
            else:
                checkpoint = init_checkpoint
                print(checkpoint.keys())
            args.start_epoch = args.best_epoch = checkpoint.setdefault('epoch', 0)
            args.best_epoch = checkpoint.setdefault('best_epoch', 0)
            args.best_prec1 = checkpoint.setdefault('best_loss', 0)
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
            print(checkpoint['state_dict'].keys())
            ckpt = partial_load_checkpoint(checkpoint['state_dict'], args.amp, ignore_params)
            if args.distributed:
                model.module.load_state_dict(ckpt)  # , strict=False
            else:
                model.load_state_dict(ckpt)#, strict=False
            if global_rank == 0:
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


class EpochRunner():

    def __init__(self, args, sess):
        self.args = args
        if args.use_log:
            out_dir, model_save_dir, tfb_dir = create_logger(args, args.experimental_desc, dist_print=args.global_rank)
            self.args.out_dir = out_dir
            self.args.model_save_dir = model_save_dir
            self.args.tfb_dir = tfb_dir
        # self.tester = Tester(args)
        # self.std_criterion = torch.nn.MSELoss(reduction='mean').cuda()
        self.sess = sess

    def eval(self, eval_loader, model, criterion, eval_sampler):
        if self.args.distributed:
            eval_sampler.set_epoch(0)
        model.init_eval_obj(args)
        model = dist_train_v1(self.args, model)
        model, _ = load_checkpoint(self.args, model, None)
        val_loss = self.eval_framework(eval_loader, model, criterion)

    def run(self, train_loader, model, criterion, optimizer, val_loader, scheduler, **kwargs):
        args = self.args
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)
        model = model_amp(args, model, criterion)
        optimizer, scaler = model.apex_initialize(optimizer)
        model.dist_train()
        model, optimizer = load_checkpoint(args, model, optimizer)

        start_time = time.time()
        for epoch in range(args.start_epoch, args.epochs):
            if self.args.distributed:
                kwargs.get('train_sampler').set_epoch(epoch)
            epoch_time = datetime.datetime.now()
            train_stats = train_one_epoch(args, model, criterion, train_loader, optimizer, args.device,
                                         epoch, scaler)
            # val_stats = self.validate_framework(val_loader, model, criterion, epoch)

            if self.args.lr_scheduler and scheduler is not None:
                scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
            val_metric = train_stats['psnr']
            val_loss = train_stats['Loss']
            is_best = val_metric > self.args.best_prec1
            self.args.best_prec1 = max(val_metric, args.best_prec1)

            if epoch % args.print_freq == 0 or is_best:
                if is_best:
                    self.args.best_epoch = epoch
                if args.global_rank == 0: #dist.get_rank() == 0
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'arch': args.arch,
                        'state_dict': model.state_dict(),
                        'best_metric': self.args.best_prec1,
                        'loss': val_loss,
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

    @torch.no_grad()
    def validate_framework(self, val_loader, model, criterion, epoch=0):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}/{1}]'.format(epoch, args.epochs)
        # switch to evaluate mode
        model.eval()
        # for iteration, batch in enumerate(val_loader, 1):
        for batch, idx in metric_logger.log_every(val_loader, 1, header):
            index = batch['index']
            samples = batch['O'].to(args.device, non_blocking=True)
            gt = batch['B'].to(args.device, non_blocking=True)
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])

            outputs, loss_dicts = model(sub_mean(samples), sub_mean(gt), index)
            # loss_dicts = criterion(outputs, gt)

            weight_dict = criterion.weight_dict
            losses = sum(loss_dicts[k] * weight_dict[k] for k in loss_dicts.keys() if k in weight_dict)

            loss_dicts['Loss'] = losses
            # pred_np = add_mean(outputs.cpu().detach().numpy())
            # pred_np = quantize(pred_np, 255)
            # gt = gt.cpu().numpy() * 255
            # psnr = calc_psnr(pred_np, gt, 4, 255.0)
            psnr = reduce_mean(psnr_v2(add_mean(outputs), gt * 255.0, 4, 255.0))
            metric_logger.update(**loss_dicts)
            metric_logger.update(
                psnr=psnr)  # reduce_mean(psnr_t(outputs, gt)))#PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        # metric_logger.synchronize_between_processes()
        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}


        if args.global_rank == 0:
            log_string("[{}] Averaged stats: {}".format(epoch, metric_logger))
            if args.use_tb:
                tfb_metrics = stats.copy()
                tfb_metrics.pop(k for k in ['grad_norm', 'eta', 'lr', 'bce_loss', 'time'])
                args.test_writer.add_scalar(tfb_metrics, epoch)
                args.test_writer.flush()

        # stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        return stats#stats

    @torch.no_grad()
    def eval_framework(self, eval_loader, model, criterion):
        args = self.args
        metric_logger = MetricLogger(dist_print=args.global_rank, delimiter="  ")
        header = 'TestEpoch: [{0}]'.format(args.start_epoch)
        # switch to evaluate mode
        model.eval()
        psnr_list = []
        # for iteration, batch in enumerate(val_loader, 1):
        saved_path = f"./my_model_results/{args.eval}"

        if args.local_rank == 0:
            if os.path.exists(saved_path) is False:
                os.mkdir(saved_path)

        for batch, idx in metric_logger.log_every(eval_loader, 1, header):
            # index = batch['index']
            samples = batch['O'].to(args.device, non_blocking=True)
            gt = batch['B'].to(args.device, non_blocking=True)
            filename = batch['file_name']
            # gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
            if args.distributed:
                outputs = model.module.forward(sub_mean(samples))#forward_chop(sub_mean(samples))
            else:
                outputs = model.forward_chop(samples)#sub_mean(samples))
            pred = quantize(outputs * 255, 255)
            normalized = pred[0].mul(255 / args.rgb_range)
            tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()

            imageio.imwrite(os.path.join(saved_path, ''.join([filename[0], '.png'])),
                            tensor_cpu.numpy())

            # pred_np = quantize(outputs.cpu().detach().numpy(), 255)
            psnr = reduce_mean(psnr_v2(outputs * 255, gt * 255, 4, 255.0))
            ssim = g_ssim(outputs, gt)
            # psnr = calc_psnr(outputs.cpu().numpy(), gt.cpu().numpy() * 255, 4, 255.0)  # [0].permute(1, 2, 0)
            # metric_logger.update(**loss_dicts)
            psnr_list.append(psnr.item())
            print(args.local_rank, filename)
            metric_logger.update(ssim=ssim)
            metric_logger.update(psnr=psnr)  # PSNR(outputs.cpu().numpy(), gt.cpu().numpy()))

        stats = {k: meter.avg for k, meter in metric_logger.meters.items()}
        log_string("Averaged stats: {} ({})".format(metric_logger, np.mean(psnr_list)))

        return stats  # stats

def show_maps(axes, O, B, outputs):
    pred = outputs[0, ...].cpu().detach().numpy().transpose(1, 2, 0)
    gt = B[0, ...].cpu().numpy().transpose(1, 2, 0)
    axes[0, 0].imshow(O[0, ...].cpu().numpy().transpose(1, 2, 0))
    axes[0, 1].imshow(pred)
    axes[1, 0].imshow(B[0, ...].cpu().numpy().transpose(1, 2, 0))
    axes[1, 1].imshow(np.abs(pred - gt))

    # pred = outputs[1, ...].cpu().detach().numpy().transpose(1, 2, 0)
    # gt = B[1, ...].cpu().numpy().transpose(1, 2, 0)
    # axes[1, 0].imshow(O[1, ...].cpu().numpy().transpose(1, 2, 0))
    # axes[1, 1].imshow(pred)
    # axes[1, 2].imshow(B[1, ...].cpu().numpy().transpose(1, 2, 0))
    # axes[1, 3].imshow(np.abs(pred - gt))

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


    # model_without_ddp = model
    # param_dicts = [
    #     {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
    #     {
    #         "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
    #         "lr": args.lr_backbone,
    #     },
    # ]

    ##################################################
    if args.eval:
        eval_loader, eval_sampler = sess.get_eval_dataloader(args.dataset, args.distributed)
        runner.eval(eval_loader, model, criterion, eval_sampler)
    else:
        train_loader, train_sampler = sess.get_dataloader(args.dataset, args.distributed)
        # val_loader, val_sampler = sess.get_test_dataloader('test', args.distributed)

        # model = dist_train_v1(args, model)

        # skip = {}
        # skip_keywords = {}
        # if hasattr(model, 'no_weight_decay'):
        #     skip = model.no_weight_decay()
        # if hasattr(model, 'no_weight_decay_keywords'):
        #     skip_keywords = model.no_weight_decay_keywords()
        # parameters = set_weight_decay(model, skip, skip_keywords)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=1e-4)
        if args.lr_scheduler:
            from optim import lr_scheduler
            scheduler = lr_scheduler(optimizer.param_groups[0]['lr'], 600)
            scheduler.set_optimizer(optimizer, torch.optim.lr_scheduler.MultiStepLR)
            # scheduler.set_optimizer(optimizer, None)
            scheduler.get_lr_map("step_lr_100",
                                 out_file=os.path.join(args.out_dir, f"./step_lr_100_{args.experimental_desc}.png"))
        else:
            scheduler = None
        if args.once_epoch:
            train_loader = iter(list(train_loader))

        # for epoch in range(args.start_epoch, args.epochs):
        #     train_one_epoch(model, criterion, train_loader, optimizer, device, epoch, args.clip_max_norm)

        runner.run(train_loader, model, criterion, optimizer, None, scheduler=scheduler, train_sampler=train_sampler)

        if args.use_tb:
            args.train_writer.close()
            args.test_writer.close()




if __name__ == "__main__":
    import numpy as np
    from option import args
    torch.cuda.empty_cache()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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

    #8-28 modified HAttention split bug
    # rain200L
    #- Averaged stats: psnr: 40.6850 (38.7817267) (38.781726684570316)
    # rain200H
    # - Averaged stats: ssim: 0.8570 (0.8800113)  psnr: 26.2376 (29.2022589) (29.202258863449096)
    #DID
    #Averaged stats: ssim: 0.9739 (0.9188671)  psnr: 33.9409 (34.4049153) (34.40491533915202) 1200
    #- Averaged stats: ssim: 0.9413 (0.9158865)  psnr: 32.3577 (34.4011542) (34.401154200236) 1201 + - Averaged stats: ssim: 0.9589 (0.9588512)  psnr: 36.7906 (36.7906303) (36.79063034057617)
    #matlab: For all datasets PSNR: 34.406979 SSIM: 0.957361 (PR) For all datasets PSNR: 34.367583 SSIM: 0.935571
    #DDN 534
    #Averaged stats: ssim: 0.9657 (0.9219948)  psnr: 34.1281 (33.0512021) (33.05120212009975)
    #matlab: For all datasets PSNR: 33.029872 SSIM: 0.935652 psnr=33.0551, ssim=0.9487
    #
    
    
    