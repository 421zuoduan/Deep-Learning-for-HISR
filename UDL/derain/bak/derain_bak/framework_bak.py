import datetime
import time
import torch
import torch.distributed as dist
import shutil
import os
from utils import AverageMeter
# from data_gen.prefetcher import data_prefetcher
from torch.backends import cudnn
# from logger import create_logger, log_string
# from main_train import Tester
import torch.nn as nn
try:
    import apex
    from apex.parallel import DistributedDataParallel as DDP
    # from utils.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    print("Currently using torch.cuda.amp")
    try:
        from torch.cuda import amp
    except ImportError:
        raise ImportError("Please install apex from https://www.github.com/nvidia/apex or use pytorch1.6+.")

class model_amp(nn.Module):
    def __init__(self, args, model, criterion):
        super(model_amp, self).__init__()
        self.args = args
        self.model = model
        self.criterion = criterion

    def __call__(self, x, gt, bs, *args, **kwargs):
        if not self.args.amp or self.args.amp is None:
            output = self.model(x, bs)
            loss = self.criterion(output, gt, *args, **kwargs)
        else:
            # torch.amp optimization
            with amp.autocast():
                output = self.model(x)
                loss = self.criterion(output, gt)

        return output, loss

    def backward(self, optimizer, loss, scaler=None):
        if self.args.amp is not None:
            if not self.args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                # optimizer.step()
                if self.args.clip_max_norm > 0:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.args.clip_max_norm)
            else:
                # torch.amp optimization
                scaler.scale(loss).backward()
                # scaler.step(optimizer)
                # scaler.update()
        else:
            loss.backward()
            # optimizer.step()


    def apex_initialize(self, optimizer):

        scaler = None
        if self.args.amp is not None:
            cudnn.deterministic = False
            cudnn.benchmark = True
            assert torch.backends.cudnn.enabled, "Amp requires cudnn backend to be enabled."


            if not self.args.amp:
                log_string("apex optimization")
                self.model, optimizer = amp.initialize(self.model, optimizer, opt_level=self.args.amp_opt_level)
                # opt_level=args.opt_level,
                # keep_batchnorm_fp32=args.keep_batchnorm_fp32,
                # loss_scale=args.loss_scale
                # )
            else:
                log_string("torch.amp optimization")
                scaler = amp.GradScaler()

        return optimizer, scaler





def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt


def save_checkpoint(state, model_save_dir, is_best, filename='amp_checkpoint.pth.tar'):
    filename = os.path.join(model_save_dir, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(model_save_dir, 'amp_model_best.pth.tar'))

from pytorch_msssim.pytorch_msssim import SSIM
# from cal_ssim import SSIM
from dataset import PSNR


# BCELoss
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
        loss = _ssim_loss.reshape(-1, 1, 1, 1) * ((l1_loss / w_1) ** ind + (l2_loss / w_2) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind +\
               l2_loss * ((_ssim_loss.reshape(-1, 1, 1, 1) / w_s) ** ind + (l1_loss / w_1) ** ind + (bce / w_bce) ** ind + 1e-8) ** ind
        # x = torch.sigmoid(x)
        # gt = torch.sigmoid(gt)
        # loss = self.l1(x, gt)
        loss = torch.mean(loss)
        return loss, w_1, w_2, w_s, w_bce, ssim_m

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

    def run(self, train_sampler, train_loader, model, criterion, optimizer, scale, val_loader, scheduler):
        # global best_prec1
        # if self.args.start_epoch == 0:
        #     val_loss = self.validate_framework(val_loader, model, criterion, 0)

        for epoch in range(self.args.start_epoch, self.args.epochs):
            if self.args.distributed and not self.args.DALI:
                train_sampler.set_epoch(epoch)
            # adjust_learning_rate(optimizer, epoch)
            try:
                epoch_time = datetime.datetime.now()
                train_loss = self.train_framework(train_loader, model, criterion, optimizer, epoch, scale)
            except StopIteration:
                train_loader = self.sess.get_dataloader('train')
                train_loss = self.train_framework(train_loader, model, criterion, optimizer, epoch, scale)

            val_loss = self.validate_framework(val_loader, model, criterion, epoch)

            if self.args.lr_scheduler and scheduler is not None:
                scheduler.step(epoch, True)

            # remember best prec@1 and save checkpoint
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
                    'state_dict': model.state_dict(),
                    'best_loss': self.args.best_prec1,
                    'loss': val_loss,
                    'best_epoch': self.args.best_epoch,
                    # 'optimizer': optimizer.state_dict(),
                }, self.args.model_save_dir, is_best, filename=f"{epoch}.pth.tar")

            log_string(' * Best validation Loss so far@1 {loss:.7f} in epoch {best_epoch}'.format(
                loss=self.args.best_prec1, best_epoch=self.args.best_epoch))

            log_string("one epoch time: {}".format(
                datetime.datetime.now() - epoch_time))
        # except KeyboardInterrupt or RuntimeError:
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': self.args.arch,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': self.args.best_prec1,
        #     'optimizer': optimizer.state_dict(),
        # }, self.args.model_save_dir, is_best=False, filename=f"{epoch}_kbInter.pth")

    def train_framework(self, train_loader, model, criterion, optimizer, epoch, scale):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        # std_losses = AverageMeter()
        ssims = AverageMeter()
        l1 = AverageMeter()
        l2 = AverageMeter()
        bce = AverageMeter()
        ssim_losses = AverageMeter()
        psnrs = AverageMeter()
        # fig, axes = plt.subplots(2,1)
        # plt.ion()
        # psnrs = AverageMeter()
        # batch_time = AverageMeter('Time', ':6.3f')
        # data_time = AverageMeter('Data', ':6.3f')
        # losses = AverageMeter('Loss', ':.4e')
        # progress = ProgressMeter(
        #     len(train_loader),
        #     [batch_time, data_time, losses],
        #     prefix="Epoch: [{}]".format(epoch))

        model.train()
        # epoch_train_loss = []
        end = time.time()
        for iteration, batch in enumerate(train_loader, 1):
            data_time.update(time.time() - end)

            O, gt = batch['O'].cuda(self.args.gpu), batch['B'].cuda(self.args.gpu)
            gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
            # O, gt = Variable(O, requires_grad=False), Variable(gt, requires_grad=False)
            if self.args.amp:
                with amp.autocast():
                    output = model(O)
                    loss = criterion(output, gt)
                    with torch.no_grad():
                        mse_loss = self.std_criterion(output, gt)

            else:
                output = model(O)
                # pred = np.clip(output.permute(0, 2, 3, 1)[0, ...].cpu().detach().numpy() * 255, 0, 255)
                # axes[0].imshow(O.permute(0, 2, 3, 1)[0, ...].cpu().detach().numpy())
                # axes[1].imshow(pred)
                # plt.pause(1)
                # plt.show()
                loss, l1_loss, l2_loss, ssim_loss, bce_loss, ssim = self.bcmsl(output, gt)
                # with torch.no_grad():
                #     mse_loss = self.std_criterion(output, gt)
                    # ssim = self.ssim(output, gt)

            with torch.no_grad():
                psnrs.update(PSNR(None, None, l2_loss, False))
                # psnrs.update(PSNR(output.data.cpu().numpy() * 255, gt.data.cpu().numpy() * 255))
            # epoch_train_loss.append(loss.item())
            # TODO: check if need to use this codes. 原版没有,精确但增加时间开销
            if self.args.distributed:
                torch.distributed.barrier()
                reduced_loss = reduce_mean(loss, self.args.nprocs)
                losses.update(reduced_loss.item())
                reduced_mse_loss = reduce_mean(mse_loss, self.args.nprocs)
                # std_losses.update(reduced_mse_loss.item())
            else:
                if hasattr(loss, 'item'):
                    losses.update(loss.item())
                else:
                    losses.update(loss)
                if hasattr(ssim, 'item'):
                    ssims.update(ssim.item())
                else:
                    ssims.update(ssim)

                l1.update(l1_loss.item())
                l2.update(l2_loss.item())
                bce.update(bce_loss.item())
                ssim_losses.update(ssim_loss)
                # std_losses.update(mse_loss.item())

            # compute gradient and do SGD step
            optimizer.zero_grad()
            if self.args.amp is not None:
                if not self.args.amp:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    optimizer.step()
                else:
                    scale.scale(loss).backward()
                    scale.step(optimizer)
                    scale.update()
            else:
                loss.backward()
                optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if iteration % self.args.print_freq == 0:
                # progress.display(iteration)
                log_string('Epoch: [{0}][{1}/{2}]\t'
                           'Lr: {3:.7f}\t'
                           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                           'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                           'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                           'l1_loss {l1_loss.val:.5f} ({l1_loss.avg:.5f})\t'
                           'mse_Loss {mse_loss.val:.5f} ({mse_loss.avg:.5f})\t'
                           'bce_loss {bce_loss.val:.5f} ({bce_loss.avg:.5f})\t'
                           'ssim_loss {ssim_loss.val:.7f} ({ssim_loss.avg:.5f})\t'
                           'ssim {ssim.val:.5f} ({ssim.avg:.5f})\t'
                           'psnr {psnr.val:.5f} ({psnr.avg:.5f})'.format(
                    epoch, iteration, len(train_loader), optimizer.param_groups[0]["lr"],
                    batch_time=batch_time, data_time=data_time, loss=losses, l1_loss=l1, mse_loss=l2, bce_loss=bce,
                    ssim_loss=ssim_losses, ssim=ssims, psnr=psnrs))
            # scheduler.step(epoch, True)
            # print("lr:", optimizer.param_groups[0]['lr'])
            # t_loss = np.nanmean(np.array(epoch_train_loss))
            # log_string("Epoch: {}/{}  Compared training Loss:{:.7f}".format(epoch, args.epochs, t_loss))
        if self.args.use_tb:
            self.args.train_writer.add_scalar('mse_loss', losses.avg, epoch)
            self.args.train_writer.flush()

        log_string('TrainEpoch: [{0}/{1}]\t'
                   'Lr: {2:.7f}\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                   'l1_loss {l1_loss.val:.5f} ({l1_loss.avg:.5f})\t'
                   'mse_Loss {mse_loss.val:.5f}({mse_loss.avg:.5f})\t'
                   'bce_loss {bce_loss.val:.5f} ({bce_loss.avg:.5f})\t'
                   'ssim_loss {ssim_loss.val:.5f} ({ssim_loss.avg:.5f})\t'
                   'ssim {ssim.val:.5f} ({ssim.avg:.5f})\t'
                   'psnr {psnr.val:.5f} ({psnr.avg:.5f}) '.format(
            epoch, self.args.epochs, optimizer.param_groups[0]["lr"],
            batch_time=batch_time, data_time=data_time, loss=losses, mse_loss=l2, l1_loss=l1, bce_loss=bce,
            ssim_loss=ssim_losses, ssim=ssims, psnr=psnrs))
        # plt.ioff()
        return losses.avg

    def validate_framework(self, val_loader, model, criterion, epoch=0):
        batch_time = AverageMeter()
        losses = AverageMeter()
        # std_losses = AverageMeter()
        ssims = AverageMeter()
        psnrs = AverageMeter()
        l1 = AverageMeter()
        l2 = AverageMeter()
        bce = AverageMeter()
        ssim_losses = AverageMeter()

        # switch to evaluate mode
        model.eval()
        # epoch_val_loss = []
        with torch.no_grad():
            end = time.time()

            for iteration, batch in enumerate(val_loader, 1):
                O, gt = batch['O'].cuda(self.args.gpu), batch['B'].cuda(self.args.gpu)
                gt = gt.view(gt.shape[0] * gt.shape[1], gt.shape[2], gt.shape[3], gt.shape[4])
                # O, gt = Variable(O, requires_grad=False), Variable(gt, requires_grad=False)
                # compute output
                output = model(O)
                # loss = self.l1(output, gt)
                # # gt = torch.exp(gt)
                # ssim = self.ssim(output, gt)

                loss, l1_loss, l2_loss, ssim_loss, bce_loss, ssim = self.bcmsl(output, gt)
                # mse_loss = self.std_criterion(output, gt)
                # epoch_val_loss.append(loss.item())
                # measure accuracy and record loss
                if self.args.distributed:
                    torch.distributed.barrier()

                    reduced_loss = reduce_mean(loss, self.args.nprocs)
                    # loss是Tenosr的话需要检查hasattr(t, 'item')
                    losses.update(reduced_loss.item())

                    # reduced_mse_loss = reduce_mean(mse_loss, self.args.nprocs)
                    # std_losses.update(reduced_mse_loss.item())


                else:
                    # loss是Tenosr的话需要检查hasattr(t, 'item')
                    if hasattr(loss, 'item'):
                        losses.update(loss.item())
                    else:
                        losses.update(loss)
                    if hasattr(ssim, 'item'):
                        ssims.update(ssim.item())
                    else:
                        ssims.update(ssim)
                    # ssims.update(ssim_loss.item())
                    # std_losses.update(mse_loss.item())
                    l1.update(l1_loss.item())
                    l2.update(l2_loss.item())
                    bce.update(bce_loss.item())
                    ssim_losses.update(ssim_loss.item())
                    #psnrs.update(PSNR(output.data.cpu().numpy() * 255, gt.data.cpu().numpy() * 255))
                    psnrs.update(PSNR(None, None, l2_loss, False))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                # if iteration % self.args.print_freq == 0:
                #     self.sess.write('valid', loss, epoch)

                if iteration % self.args.print_freq == 0:
                    log_string('Test: [{0}/{1}]\t'
                               'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                               'Loss {loss.val:.5f} ({loss.avg:.7f})\t'
                               'l1_loss {l1_loss.val:.5f} ({l1_loss.avg:.5f})\t'
                               'mse_Loss {mse_loss.val:.7f} ({mse_loss.avg:.7f})\t'
                               'bce_loss {bce_loss.val:.5f} ({bce_loss.avg:.5f})\t'
                               'ssim_loss {ssim_loss.val:.5f} ({ssim_loss.avg:.5f})\t'
                               'ssim {ssim.val:.5f} ({ssim.avg:.5f})\t'
                               'psnr {psnr.val:.5f} ({psnr.avg:.5f})\t'.format(
                        iteration, len(val_loader), batch_time=batch_time, loss=losses, l1_loss=l1,
                        mse_loss=l2, bce_loss=bce, ssim_loss=ssim_losses, ssim=ssims, psnr=psnrs))

            # indexes = self.tester(model)

            if self.args.use_tb:
                self.args.test_writer.add_scalar('mse_loss', losses.avg, epoch)
                self.args.test_writer.flush()
            # log_string("Epoch: {}/{}  Compared validate Loss: {:.7f}".
            #            format(epoch, args.epochs, v_loss))

            log_string('TestEpoch: [{0}/{1}]\t'
                       'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                       'Loss {loss.val:.5f} ({loss.avg:.5f})\t'
                       'l1_loss {l1_loss.val:.5f} ({l1_loss.avg:.5f})\t'
                       'mse_Loss {mse_loss.val:.5f} ({mse_loss.avg:.5f})\t'
                       'bce_loss {bce_loss.val:.5f} ({bce_loss.avg:.5f})\t'
                       'ssim_loss {ssim_loss.val:.7f} ({ssim_loss.avg:.5f})\t'
                       'ssim {ssim.val:.5f} ({ssim.avg:.5f})\t'
                       'psnr {psnr.val:.5f} ({psnr.avg:.5f})\t'.format(
                epoch, self.args.epochs, batch_time=batch_time, loss=losses, l1_loss=l1,
                mse_loss=l2, bce_loss=bce, ssim_loss=ssim_losses, ssim=ssims, psnr=psnrs))
        return losses.avg
