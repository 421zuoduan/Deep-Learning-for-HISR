import os
import sys
import cv2
import argparse
import numpy as np

import torch
from torch import nn
from torch.nn import MSELoss
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter

import settings
# from dataset import TrainValDataset
from derain_dataset import TrainValDataset
from derain_dataset import derainSession
from utils.utils import set_random_seed
from model import RESCAN
from cal_ssim import SSIM
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2, nrows=1)


logger = settings.logger
# torch.cuda.manual_seed_all(66)
# torch.manual_seed(66)

torch.cuda.set_device(settings.device_id)

# print(torch.randperm(3600).tolist())
'''
2021-10-10 00:52:13,749 - INFO - train_c--loss0:0.01524 loss1:0.01524 loss2:0.01523 loss3:0.01523 ssim0:0.847 ssim1:0.8471 ssim2:0.8471 ssim3:0.8471 Loss:0.06093 lr:0.005 step:0
2021-10-10 00:52:16,150 - INFO - train_c--loss0:0.01513 loss1:0.01513 loss2:0.01513 loss3:0.01512 ssim0:0.7934 ssim1:0.7936 ssim2:0.7936 ssim3:0.7936 Loss:0.06051 lr:0.005 step:1
2021-10-10 00:52:18,426 - INFO - train_c--loss0:0.01672 loss1:0.01672 loss2:0.01671 loss3:0.01671 ssim0:0.7919 ssim1:0.7921 ssim2:0.7921 ssim3:0.7921 Loss:0.06686 lr:0.005 step:2
2021-10-10 00:52:20,743 - INFO - train_c--loss0:0.0173 loss1:0.0173 loss2:0.01729 loss3:0.01729 ssim0:0.8026 ssim1:0.8027 ssim2:0.8027 ssim3:0.8027 Loss:0.06917 lr:0.005 step:3
2021-10-10 00:52:23,045 - INFO - train_c--loss0:0.01498 loss1:0.01498 loss2:0.01497 loss3:0.01497 ssim0:0.7839 ssim1:0.7843 ssim2:0.7843 ssim3:0.7843 Loss:0.05991 lr:0.005 step:4
2021-10-10 00:52:25,348 - INFO - train_c--loss0:0.01539 loss1:0.01539 loss2:0.01538 loss3:0.01537 ssim0:0.8299 ssim1:0.8301 ssim2:0.8301 ssim3:0.8301 Loss:0.06153 lr:0.005 step:5

- Epoch: [0]  [  0/113]  eta: 0:07:23  lr: 0.005000  grad_norm: 0.518580  Loss: 0.0609 (0.0609327)  loss0: 0.0152 (0.0152390)  loss1: 0.0152 (0.0152369)  loss2: 0.0152 (0.0152305)  loss3: 0.0152 (0.0152264)  ssim0: 0.8470 (0.8470038)  ssim1: 0.8471 (0.8470976)  ssim2: 0.8471 (0.8471146)  ssim3: 0.8471 (0.8471222)  psnr: 7.5024 (7.5024176)  time: 3.9272  data: 0.3183  max mem: 1985MB
- Epoch: [0]  [  1/113]  eta: 0:06:17  lr: 0.005000  grad_norm: 0.477978  Loss: 0.0532 (0.0570878)  loss0: 0.0136 (0.0143995)  loss1: 0.0133 (0.0142647)  loss2: 0.0132 (0.0142198)  loss3: 0.0132 (0.0142038)  ssim0: 0.7966 (0.8218012)  ssim1: 0.7964 (0.8217415)  ssim2: 0.7963 (0.8217161)  ssim3: 0.7963 (0.8217137)  psnr: 8.1220 (7.8122191)  time: 3.3687  data: 0.3099  max mem: 1993MB
- Epoch: [0]  [  2/113]  eta: 0:05:53  lr: 0.005000  grad_norm: 0.446102  Loss: 0.0526 (0.0555970)  loss0: 0.0136 (0.0141431)  loss1: 0.0132 (0.0138936)  loss2: 0.0130 (0.0138005)  loss3: 0.0129 (0.0137597)  ssim0: 0.7975 (0.8137147)  ssim1: 0.7973 (0.8136085)  ssim2: 0.7971 (0.8135262)  ssim3: 0.7970 (0.8134777)  psnr: 7.4309 (7.6851214)  time: 3.1811  data: 0.3063  max mem: 1993MB
[3127, 3307, 1430, 138, 1634, 2466, 1435, 373, 2868, 57, 3031, 3249, 1913, 3560, 1951, 3266, 3121, 1078, 1907, 3350, 3554, 358, 1447, 496, 2331, 60, 716, 1528, 2177, 3073, 47, 1589, 2595, 3457, 1619, 749, 1587, 544, 1843, 1250, 1348, 1997, 3380, 304, 1602, 1425, 440, 183, 2672, 293, 2139, 789, 2483, 2340, 2468, 3430, 3191, 1486, 2944, 467, 1461, 2310, 14, 365, 581, 821, 2058, 579, 28, 667, 2892, 580, 3004, 18, 1953, 1647, 2708, 853, 3197, 1784, 2262, 2239, 2972, 188, 1822
'''


def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)


class Session:
    def __init__(self):
        self.log_dir = settings.log_dir
        self.model_dir = settings.model_dir
        ensure_dir(settings.log_dir)
        ensure_dir(settings.model_dir)
        logger.info('set log dir as %s' % settings.log_dir)
        logger.info('set model dir as %s' % settings.model_dir)

        self.net = RESCAN().cuda()
        self.crit = MSELoss().cuda()
        self.ssim = SSIM().cuda()

        self.step = 0
        self.save_steps = settings.save_steps
        self.num_workers = settings.num_workers
        self.batch_size = settings.batch_size
        self.writers = {}
        self.dataloaders = {}

        self.opt = Adam(self.net.parameters(), lr=settings.lr)
        self.sche = MultiStepLR(self.opt, milestones=[15000, 17500], gamma=0.1)

    def tensorboard(self, name):
        self.writers[name] = SummaryWriter(os.path.join(self.log_dir, name + '.events'))
        return self.writers[name]

    def write(self, name, out):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, self.step)

        out['lr'] = self.opt.param_groups[0]['lr']
        out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        logger.info(name + '--' + ' '.join(outputs))

    def get_dataloader(self, dataset_name):
        dataset = TrainValDataset(settings.data_dir, dataset_name, settings.patch_size)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return iter(self.dataloaders[dataset_name])

    def save_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        obj = {
            'net': self.net.state_dict(),
            'clock': self.step,
            'opt': self.opt.state_dict(),
        }
        torch.save(obj, ckp_path)

    def load_checkpoints(self, name):
        ckp_path = os.path.join(self.model_dir, name)
        try:
            obj = torch.load(ckp_path)
            logger.info('Load checkpoint %s' % ckp_path)
        except FileNotFoundError:
            logger.info('No checkpoint %s!!' % ckp_path)
            return
        self.net.load_state_dict(obj['net'])
        self.opt.load_state_dict(obj['opt'])
        self.step = obj['clock']
        self.sche.last_epoch = self.step

    def inf_batch(self, name, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        O, B = Variable(O, requires_grad=False), Variable(B, requires_grad=False)
        axes[0].imshow(O[0].permute(1, 2, 0).cpu().numpy())
        axes[1].imshow(B[0].permute(1, 2, 0).cpu().numpy())
        plt.savefig(f"{self.step}_{batch['filename'][0]}.png")

        R = O - B

        O_Rs = self.net(O)
        # print(O_Rs[0][0, 0, 0, 0])
        loss_list = [self.crit(O_R, R) for O_R in O_Rs]
        ssim_list = [self.ssim(O - O_R, O - R) for O_R in O_Rs]
        # print(sum(loss_list))
        if name == 'train':
            self.net.zero_grad()
            sum(loss_list).backward()
            self.opt.step()

        losses = {
            'loss%d' % i: loss.item()
            for i, loss in enumerate(loss_list)
        }
        ssimes = {
            'ssim%d' % i: ssim.item()
            for i, ssim in enumerate(ssim_list)
        }
        losses.update(ssimes)
        losses.update({'Loss': sum(loss_list)})

        self.write(name, losses)

        return O - O_Rs[-1]

    def save_image(self, name, img_lists):
        data, pred, label = img_lists
        pred = pred.cpu().data

        data, label, pred = data * 255, label * 255, pred * 255
        pred = np.clip(pred, 0, 255)

        h, w = pred.shape[-2:]

        gen_num = (6, 2)
        img = np.zeros((gen_num[0] * h, gen_num[1] * 3 * w, 3))
        for img_list in img_lists:
            for i in range(gen_num[0]):
                row = i * h
                for j in range(gen_num[1]):
                    idx = i * gen_num[1] + j
                    tmp_list = [data[idx], pred[idx], label[idx]]
                    for k in range(3):
                        col = (j * 3 + k) * w
                        tmp = np.transpose(tmp_list[k], (1, 2, 0))
                        img[row: row + h, col: col + w] = tmp

        img_file = os.path.join(self.log_dir, '%d_%s.jpg' % (self.step, name))
        cv2.imwrite(img_file, img)


def run_train_val(ckp_name='latest'):
    sess = Session()
    sess.load_checkpoints(ckp_name)

    sess.tensorboard('train_c')
    # sess.tensorboard('test_c')

    dt_train = sess.get_dataloader('train_c')
    # dt_val = sess.get_dataloader('test_c')

    while sess.step < 20000:
        sess.sche.step()
        sess.net.train()

        try:
            batch_t = next(dt_train)
        except StopIteration:
            dt_train = sess.get_dataloader('train_c')
            batch_t = next(dt_train)
        pred_t = sess.inf_batch('train_c', batch_t)

        # if sess.step % 4 == 0:
        #     sess.net.eval()
        #     try:
        #         batch_v = next(dt_val)
        #     except StopIteration:
        #         dt_val = sess.get_dataloader('test_c')
        #         batch_v = next(dt_val)
        #     pred_v = sess.inf_batch('test_c', batch_v)

        if sess.step % int(sess.save_steps / 16) == 0:
            sess.save_checkpoints('latest')
        # if sess.step % int(sess.save_steps / 2) == 0:
        #     sess.save_image('train', [batch_t['O'], pred_t, batch_t['B']])
            # if sess.step % 4 == 0:
            #     sess.save_image('val', [batch_v['O'], pred_v, batch_v['B']])
            # logger.info('save image as step_%d' % sess.step)
        if sess.step % sess.save_steps == 0:
            sess.save_checkpoints('step_%d' % sess.step)
            logger.info('save model as step_%d' % sess.step)
        sess.step += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', default='latest')
    # seed should be set in code block if __name__ == '__main__':
    # outside location will lead to difference.
    set_random_seed(1)
    args = parser.parse_args(sys.argv[1:])
    run_train_val(args.model)
