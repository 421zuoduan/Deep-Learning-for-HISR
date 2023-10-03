import os
import cv2
import numpy as np
import torch
# from numpy.random import RandomState
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import settings
from torch.autograd import no_grad
import math
import torch.nn as nn
from cal_ssim import SSIM
from utils.logger import log_string
from utils.utils import AverageMeter
from data.srdata import SRData
cv2.setNumThreads(1)

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
        # print(mse, mse1)
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    else:
        if hasattr(mse, 'item'):
            mse = mse.item()
            if mse == 0:
                return 100
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))


class PSNR_t(nn.Module):
    def __init__(self):
        super(PSNR_t, self).__init__()

    @torch.no_grad()
    def forward(self, img1, img2, mse=None, inter=True):
        PIXEL_MAX = 1
        if inter:
            b, _, _, _ = img1.shape
            # mse1 = np.mean((img1 - img2) ** 2)
            img1 = torch.clip(img1 * 255, 0, 255) / 255.
            img2 = torch.clip(img2 * 255, 0, 255) / 255.
            mse = torch.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100
            # print(mse, mse1)
            return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))
        else:
            if hasattr(mse, 'item'):
                mse = mse.item()
                if mse == 0:
                    return 100
                return 20 * torch.log10(PIXEL_MAX / torch.sqrt(mse))


class derainSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.patch_size = args.patch_size
        self.l1 = nn.L1Loss().cuda()
        self.ssim = SSIM().cuda()
        self.writers = {}
        self.args = args
        # self.l1_losses = Aver
        # ageMeter()
        # self.ssims = AverageMeter()
        # self.psnrs = AverageMeter()

    def get_dataloader(self, dataset_name, distributed):
        dataset = SRData(self.args, 'train', train=True)#TrainValDataset(dataset_name, self.patch_size)
        # dataset = TestDataset(dataset_name)
        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size, persistent_workers=(True if self.num_workers > 0 else False),
                           shuffle=(sampler is None), num_workers=self.num_workers, drop_last=True, sampler=sampler)

        return self.dataloaders[dataset_name], sampler

    def get_test_dataloader(self, dataset_name, distributed):
        # dataset = TestDataset(dataset_name)
        dataset = TrainValDataset(dataset_name, self.patch_size)
        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=False, num_workers=self.num_workers, drop_last=False, sampler=sampler)

        return self.dataloaders[dataset_name], sampler

    def get_eval_dataloader(self, dataset_name, distributed):
        dataset = TestDatasetV2(dataset_name)
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1, sampler=sampler,
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


def image_jittor():
    from torch.utils.data import DataLoader
    class TrainValDataset(Dataset):
        def __init__(self, name):
            super().__init__()
            # self.rand_state = RandomState(66)
            # self.rain_dir = os.path.join(settings.data_dir, "train/rain")
            # self.rain_dir_list = os.listdir(self.rain_dir)
            # self.derain_dir = os.path.join(settings.data_dir, "train/norain")
            # self.derain_dir_list = os.listdir(self.derain_dir)
            self.dir = os.path.join(settings.data_dir, "test_bak")
            self.dir_list = os.listdir(self.dir)
            print(self.dir_list, len(self.dir_list))

            self.patch_size = settings.patch_size
            self.file_num = len(self.dir_list)

        def __len__(self):
            return self.file_num

        def __getitem__(self, idx):
            # file_name = self.mat_files[idx % self.file_num]
            file_name = self.dir_list[idx]
            im_file = os.path.join(self.dir, file_name)

            o_file_name, suffix = im_file[4:].split('.')
            suffix = "." + suffix
            if 'x2' in im_file:

                rain_im_file = im_file[:-6] + suffix
                derain_im_file = im_file
            else:
                # file_name, suffix = im_file[2:].split('.')
                rain_im_file = im_file
                derain_im_file = im_file[:-4] + 'x2' + suffix

            # derain_name = self.derain_dir_list[idx]
            # rain_im_file = os.path.join(self.dir, rain_im_file)
            # derain_im_file = os.path.join(self.dir, derain_im_file)
            img_left = cv2.imread(rain_im_file).astype(np.float32) / 255
            img_right = cv2.imread(derain_im_file).astype(np.float32) / 255
            h, w, c = img_left.shape

            img_left = np.tile(img_left, [1, 2, 1])
            img_left[:, w:2 * w, :] = img_right
            #
            # cv2.imshow("left", img_left)
            # cv2.imshow("right", img_right)
            # cv2.waitKey(1000)
            # cv2.imwrite(settings.data_dir+"/train/"+file_name, img_left)
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            plt.imsave(settings.data_dir + "/test/" + file_name, img_left)

            return img_left, img_right

    dataset = TrainValDataset("train/rain")

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    for batch in loader:
        ...


class TrainValDataset(Dataset):
    def __init__(self, name, patch_size):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
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
            img_pair = cv2.imread(img_file).astype(np.float32)# / 255
            img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
        except Exception:
            print(img_file)
        # if settings.aug_data:
        #     O, B = self.crop(img_pair, aug=True)
        #     O, B = self.augment(O, B)
        #     # O, B = self.rotate(O, B)
        # else:
        #     O, B = self.crop(img_pair, aug=False)
        O, B = self.crop(img_pair, aug=True)
        O, B = self.augment(O, B)

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        # sample = {'O': torch.from_numpy(O), 'B': torch.from_numpy(B)}
        O = torch.from_numpy(np.ascontiguousarray(O))
        B = torch.from_numpy(np.ascontiguousarray(B))
        sample = {'index': idx, 'O': O, 'B': B}

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
        p_h = p_w = patch_size
        # if aug:
        #     mini = - 1 / 4 * patch_size
        #     maxi = 1 / 4 * patch_size + 1
        #     p_h = patch_size + self.rand_state.randint(mini, maxi)
        #     p_w = patch_size + self.rand_state.randint(mini, maxi)
        # else:
        #     p_h, p_w = patch_size, patch_size
        #
        # r = self.rand_state.randint(0, h - p_h)
        # c = self.rand_state.randint(0, w - p_w)
        r = random.randrange(0, h - p_h + 1)
        c = random.randrange(0, w - p_w + 1)



        # O = img_pair[:, w:]
        # B = img_pair[:, :w]
        O = img_pair[r: r + p_h, c + w: c + p_w + w] #rain
        B = img_pair[r: r + p_h, c: c + p_w] # norain
        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)

        # if aug:
        #     O = cv2.resize(O, (patch_size, patch_size))
        #     B = cv2.resize(B, (patch_size, patch_size))

        return O, B

    def augment(*args, hflip=True, rot=True):
        """common"""
        hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        rot90 = rot and random.random() < 0.5

        def _augment(img):
            """common"""
            if hflip:
                # img = img[:, ::-1, :]
                img = np.flip(img, axis=1)
            if vflip:
                # img = img[::-1, :, :]
                img = np.flip(img, axis=0)
            if rot90:
                img = img.transpose(1, 0, 2)
            return img

        return _augment(args[0]), _augment(args[1])#[_augment(a) for a in args]

    # def flip(self, O, B):
    #     if random.rand() > 0.5:
    #         O = np.flip(O, axis=1)
    #         B = np.flip(B, axis=1)
    #     return O, B
    #
    # def rotate(self, O, B):
    #     angle = self.rand_state.randint(-30, 30)
    #     patch_size = self.patch_size
    #     center = (int(patch_size / 2), int(patch_size / 2))
    #     M = cv2.getRotationMatrix2D(center, angle, 1)
    #     O = cv2.warpAffine(O, M, (patch_size, patch_size))
    #     B = cv2.warpAffine(B, M, (patch_size, patch_size))
    #     return O, B

class TestDatasetV2(Dataset):
    def __init__(self, name):
        super().__init__()
        self.root_dir = os.path.join(settings.data_dir, "test")
        # self.root_dir = "../../tf_repo/code/TestImg/rainL"
        # self.root_dir = os.path.join("../../tf_repo/code/TestImg/rainL", name)#settings.data_dir
        self.mat_files = os.listdir(self.root_dir)
        self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        try:
            img_pair = cv2.imread(img_file).astype(np.float32)
            img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
        except Exception:
            print(img_file)

        h, ww, c = img_pair.shape
        w = int(ww / 2)
        O = img_pair[:, : w]
        B = img_pair[:, w:ww]

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        # sample = {'O': torch.from_numpy(O), 'B': torch.from_numpy(B)}
        O = torch.from_numpy(np.ascontiguousarray(O))
        B = torch.from_numpy(np.ascontiguousarray(B))
        sample = {'index': idx, 'O': O, 'B': B, 'file_name': file_name.split('.')[0]}

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


class TestDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.root_dir = "../../tf_repo/code/TestImg/rainL"
        # self.root_dir = os.path.join("../../tf_repo/code/TestImg/rainL", name)#settings.data_dir
        self.mat_files = os.listdir(self.root_dir)
        # self.patch_size = settings.patch_size
        self.file_num = len(self.mat_files)

    def __len__(self):
        return self.file_num

    # def __getitem__(self, idx):
    #     file_name = self.mat_files[idx % self.file_num]
    #     img_file = os.path.join(self.root_dir, file_name)
    #     img_pair = cv2.imread(img_file).astype(np.float32)
    #     img_pair = cv2.cvtColor(img_pair, cv2.COLOR_BGR2RGB)
    #     h, ww, c = img_pair.shape
    #     w = int(ww / 2)
    #     # h_8=h%8
    #     # w_8=w%8
    #     O = np.transpose(img_pair[:, w:], (2, 0, 1))
    #     B = np.transpose(img_pair[:, :w], (2, 0, 1))
    #     sample = {'O': O, 'B': B, 'file_name': file_name.split('.')[0]} # 321,481
    #
    #     return sample
    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img = cv2.imread(img_file).astype(np.float32)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        O = np.transpose(img, (2, 0, 1))
        sample = {'O': O, 'file_name': file_name.split('.')[0]} # 321,481

        return sample


class ShowDataset(Dataset):
    def __init__(self, name):
        super().__init__()
        self.rand_state = RandomState(66)
        self.root_dir = os.path.join(settings.data_dir, name)
        self.img_files = sorted(os.listdir(self.root_dir))
        self.file_num = len(self.img_files)

    def __len__(self):
        return self.file_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        img_pair = cv2.imread(img_file).astype(np.float32) / 255

        h, ww, c = img_pair.shape
        w = int(ww / 2)

        # h_8 = h % 8
        # w_8 = w % 8
        if settings.pic_is_pair:
            O = np.transpose(img_pair[:, w:], (2, 0, 1))
            B = np.transpose(img_pair[:, :w], (2, 0, 1))
        else:
            O = np.transpose(img_pair[:, :], (2, 0, 1))
            B = np.transpose(img_pair[:, :], (2, 0, 1))
        sample = {'O': O, 'B': B, 'file_name': file_name}

        return sample


if __name__ == '__main__':

    # image_jittor()

    # dt = TrainValDataset('val')
    # print('TrainValDataset')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
    #
    # print()
    # dt = TestDataset('test')
    # print('TestDataset')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())
    #
    # print()
    # print('ShowDataset')
    # dt = ShowDataset('test')
    # for i in range(10):
    #     smp = dt[i]
    #     for k, v in smp.items():
    #         print(k, v.shape, v.dtype, v.mean())

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

        axes[0].imshow(O.numpy().transpose(1, 2, 0))
        axes[1].imshow(B.numpy().transpose(1, 2, 0))
        plt.pause(0.4)
        # for stride in pools:
        #     O = F.avg_pool2d(O, stride)
        #     print(O.shape, B.shape)
        #
        #     axes[0].imshow(O.numpy().transpose(1, 2, 0))
        #     axes[1].imshow(B.numpy().transpose(1, 2, 0))
        #
        #     plt.show()
        #     plt.pause(1)

    plt.ioff()
