import os
import cv2
import numpy as np
import torch
from numpy.random import RandomState
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import math
import torch.nn as nn
from cal_ssim import SSIM
from logger import log_string
from image_preprocessing_t import PatchifyAugment_V2
from patch_aug_visualisation import show_patches
data_dir = "./dataset/rain100H"
import settings
patch_size = 128

def PSNR(img1, img2, mse=None, inter=True):
    PIXEL_MAX = 1
    if inter:
        b, _, _, _ = img1.shape
        mse1 = np.mean((img1 - img2) ** 2)
        img1 = np.clip(img1 * 255, 0, 255) / 255.
        img2 = np.clip(img2 * 255, 0, 255) / 255.
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return 100
        print(mse, mse1)
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        if hasattr(mse, 'item'):
            mse = mse.item()
            if mse == 0:
                return 100
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class derainSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.writers = {}
        # self.l1_losses = AverageMeter()
        # self.ssims = AverageMeter()
        # self.psnrs = AverageMeter()

    def get_dataloader(self, dataset_name):
        # dataset = TrainValDataset(dataset_name)
        dataset = NormalTrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return self.dataloaders[dataset_name]

    def get_test_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def get_eval_dataloader(self, dataset_name):
        dataset = TestDataset(dataset_name)
        # dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    def write(self, name, out, step):
        for k, v in out.items():
            self.writers[name].add_scalar(k, v, step)
        # out['lr'] = self.opt_net.param_groups[0]['lr']
        # out['step'] = self.step
        outputs = [
            "{}:{:.4g}".format(k, v)
            for k, v in out.items()
        ]
        log_string(name + '--' + ' '.join(outputs))

    def compute_loss(self, name, derain, B):
        l1_loss = self.l1(derain, B)
        ssim_loss = -self.ssim(derain, B)
        # self.l1_losses.update(l1_loss.item())
        # self.ssims.update(ssim_loss.item())
        # losses = {'L1loss': self.l1_losses.avg}
        # losses.update({'ssim': self.ssims.avg})
        # loss_both = l1_loss + ssim_loss
        # if name == 'train':
        #     loss_both.backward()

        return l1_loss  # , ssim_loss
        # self.write(name, losses, step)

    def test(self, derain, B):
        l1_loss = self.l1(derain, B)
        ssim = self.ssim(derain, B)
        psnr = PSNR(derain.data.cpu().numpy() * 255, B.data.cpu().numpy() * 255)
        # losses = {'L1loss': l1_loss}
        # losses.update({'ssim': -ssim})

        return l1_loss, ssim, psnr

class NormalTrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        # self.mat_files_derain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[0]))
        # self.mat_files_rain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[1]))
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        try:
            img_pair = cv2.imread(img_file).astype(np.float32) / 255
            img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
        except Exception:
            print(img_file)
        if settings.aug_data:
            O, B = self.crop(img_pair, aug=True)
            O, B = self.flip(O, B)
            # O, B = self.rotate(O, B)
        else:
            O, B = self.crop(img_pair, aug=False)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        # sample = {'O': torch.from_numpy(O), 'B': torch.from_numpy(B)}
        O = torch.from_numpy(np.ascontiguousarray(O))
        B = torch.from_numpy(np.ascontiguousarray(B))
        sample = {'O': O, 'B': B}

        # h, ww, c = img_pair.shape
        # w = int(ww / 2)
        # O = img_pair[: h, : w]
        # B = img_pair[: h, w:ww]
        # O = np.transpose(O, (2, 0, 1))
        # B = np.transpose(B, (2, 0, 1))
        # # sample = {'O': torch.from_numpy(O), 'B': torch.from_numpy(B)}
        # O = torch.from_numpy(np.ascontiguousarray(O))
        # B = torch.from_numpy(np.ascontiguousarray(B))
        #
        # sample = {'O': O, 'B': B}

        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi = 1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]
        B = img_pair[r: r + p_h, c: c + p_w]

        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)

        if aug:
            O = cv2.resize(O, (patch_size, patch_size))
            B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B

class TrainValDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        # self.mat_files_derain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[0]))
        # self.mat_files_rain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[1]))
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)
        self.pa = PatchifyAugment_V2(False, self.patch_size)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        try:
            img_pair = cv2.imread(img_file).astype(np.float32) / 255
            img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
        except Exception:
            print(img_file)
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        O = img_pair[: h, : w]
        B = img_pair[: h, w:ww]
        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        # sample = {'O': torch.from_numpy(O), 'B': torch.from_numpy(B)}
        O = torch.from_numpy(np.ascontiguousarray(O))
        B = torch.from_numpy(np.ascontiguousarray(B))

        # O = F.interpolate(O, scale_factor=0.5)
        # B = F.interpolate(B, scale_factor=0.5)

        O = (
            O.unfold(1, self.patch_size, self.patch_size // 2)
            .unfold(2, self.patch_size, self.patch_size // 2)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
        )

        B = (
            B.unfold(1, self.patch_size, self.patch_size // 2)
            .unfold(2, self.patch_size, self.patch_size // 2)
            .permute(1, 2, 0, 3, 4)
            .contiguous()
        )

        shape =  B.shape
        g_x, g_y, _, _, _ = B.shape
        if g_x < g_y:
            B = B.permute(1, 0, 2, 3, 4)
            O = O.permute(1, 0, 2, 3, 4)

        # print(O.shape)
        if settings.aug_data:
            # O = self.pa(O)
            # B = self.pa(B)
            # O = cv2.resize(O, (patch_size, patch_size))
            # B = cv2.resize(B, (patch_size, patch_size))
            O = O.reshape(
                O.shape[0] * O.shape[1], O.shape[2], O.shape[3], O.shape[4]
            )
            B = B.reshape(
                B.shape[0] * B.shape[1], B.shape[2], B.shape[3], B.shape[4]
            )
            O = F.interpolate(O, scale_factor=0.5, recompute_scale_factor=True)
            B = F.interpolate(B, scale_factor=0.5, recompute_scale_factor=True)
            # O = O.reshape(
            #     shape[0], shape[1], shape[2], shape[3] // 2, shape[4] //2
            # )
            # B = B.reshape(
            #     shape[0], shape[1], shape[2], shape[3] // 2, shape[4] //2
            # )


        sample = {'O': O, 'B': B}

        return sample

    def crop(self, img_pair, aug):
        patch_size = self.patch_size
        h, ww, c = img_pair.shape
        w = int(ww / 2)

        if aug:
            mini = - 1 / 4 * self.patch_size
            maxi = 1 / 4 * self.patch_size + 1
            p_h = patch_size + self.rand_state.randint(mini, maxi)
            p_w = patch_size + self.rand_state.randint(mini, maxi)
        else:
            p_h, p_w = patch_size, patch_size

        r = self.rand_state.randint(0, h - p_h)
        c = self.rand_state.randint(0, w - p_w)
        O = img_pair[r: r + p_h, c + w: c + p_w + w]#norain
        B = img_pair[r: r + p_h, c: c + p_w]#derain



        return O, B

    def flip(self, O, B):
        if self.rand_state.rand() > 0.5:
            O = np.flip(O, axis=1)
            B = np.flip(B, axis=1)
        return O, B

    def rotate(self, O, B):
        angle = self.rand_state.randint(-30, 30)
        patch_size = self.patch_size
        center = (int(patch_size / 2), int(patch_size / 2))
        M = cv2.getRotationMatrix2D(center, angle, 1)
        O = cv2.warpAffine(O, M, (patch_size, patch_size))
        B = cv2.warpAffine(B, M, (patch_size, patch_size))
        return O, B


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255
        img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
        h, ww, c = img_pair.shape
        w = int(ww / 2)
        # h_8=h%8
        # w_8=w%8
        O = np.transpose(img_pair[:, w:], (2, 0, 1))
        B = np.transpose(img_pair[:, :w], (2, 0, 1))
        sample = {'O': O, 'B': B}

        return sample



if __name__ == '__main__':

    import argparse
    import random
    from torch.backends import cudnn
    import matplotlib.pyplot as plt
    import torch.nn.functional as F
    # from gnet422.models import twostream as model

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # * Logger
    parser.add_argument('--out_dir', metavar='DIR', default='./results',
                        help='path to save model')
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DETR')
    parser.add_argument('-b', '--batch-size', default=1, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--workers', default=0, type=int)
    args = parser.parse_args()

    sess = derainSession(args)
    train_loader = sess.get_dataloader('train')
    val_loader = sess.get_eval_dataloader('test')

    pools = [2, 2, 2, 2, 2]
    depth = len(pools)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=1)
    for batch in train_loader:
        O, B = batch['O'], batch['B']
        O = O.squeeze(0)
        B = B.squeeze(0)

        # axes[0].imshow(O.numpy().transpose(1, 2, 0))
        # axes[1].imshow(B.numpy().transpose(1, 2, 0))

        show_patches(O, 1)
        show_patches(B, 2)
        plt.pause(0.4)

    plt.ioff()
