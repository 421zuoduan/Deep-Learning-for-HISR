import os
import cv2
import imageio
import numpy as np
import torch
# from numpy.random import RandomState
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
# import settings
from torch.autograd import no_grad
import math
import torch.nn as nn
# from cal_ssim import SSIM
# from utils.logger import log_string
# from utils.utils import AverageMeter
from srdata import SRData
from DFTL.data.rcd import RainHeavy, RainHeavyTest
from DDN_DATA import DDN_Dataset

# data_dir = "D:/Datasets/derain"

# cv2.setNumThreads(1)

def rgb2ycbcr(img, y_only=True):
    """metrics"""
    img.astype(np.float32)
    if y_only:
        rlt = np.dot(img, [65.481, 128.553, 24.966]) / 255.0 + 16.0
    return rlt

def quantize(img, rgb_range):
    pixel_range = 255.0 / rgb_range
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
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] -= red_channel_mean
    x[:, 1, :, :] -= green_channel_mean
    x[:, 2, :, :] -= blue_channel_mean
    return x / 255.0

def add_mean(x):
    x = x * 255.0
    red_channel_mean = 0.4488 * 255
    green_channel_mean = 0.4371 * 255
    blue_channel_mean = 0.4040 * 255
    x[:, 0, :, :] += red_channel_mean
    x[:, 1, :, :] += green_channel_mean
    x[:, 2, :, :] += blue_channel_mean
    return x

class derainSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.batch_size = args.batch_size
        self.num_workers = args.workers
        self.patch_size = args.patch_size
        # self.l1 = nn.L1Loss().cuda()
        # self.ssim = SSIM().cuda()
        self.writers = {}
        self.args = args

        # self.l1_losses = Aver
        # ageMeter()
        # self.ssims = AverageMeter()
        # self.psnrs = AverageMeter()

    def get_dataloader(self, dataset_name, distributed):

        if dataset_name == "DDN" or dataset_name == "rain12600":
            dataset = DDN_Dataset(os.path.join(self.args.data_dir, "DDN/Rain12600"), self.patch_size, eval=False)#"../derain/dataset/Rain12600"
        elif dataset_name == "DID":
            dataset = self.DID_dataset(datasetName="pix2pix_class", dataroot=os.path.join(self.args.data_dir, "DID-MDN-datasets"),
                                       batchSize=self.batch_size, workers=self.num_workers)
        elif dataset_name in ["Rain200L", "Rain200H"]:
            # #TrainValDataset(dataset_name, self.patch_size)
            dataset = TrainValDataset(self.args.data_dir, dataset_name+"/train_c", self.patch_size)
        elif dataset_name[:-1] == "PReNetData":
            dataset_name = "Rain100" + dataset_name[-1]
            self.args.dir_data = os.path.join(self.args.data_dir, dataset_name)
            dataset = SRData(self.args, 'train', dataset_name="RainTrain" + dataset_name[-1], train=True)
        elif dataset_name in ["Rain100L", "Rain100H"]:
            self.args.dir_data = '/'.join([self.args.data_dir, dataset_name, 'train'])
            dataset = RainHeavy(self.args)

        # dataset = TestDataset(dataset_name)
        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           persistent_workers=(True if self.num_workers > 0 else False),
                           shuffle=(sampler is None), num_workers=self.num_workers, drop_last=False, sampler=sampler)

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
        # global data_dir

        if dataset_name in ['Rain200H', 'Rain200L']:

            dataset = TestDatasetV2(self.args.data_dir, dataset_name+"/test_c")

        elif dataset_name == 'test12':
            dataset = DataLoaderSeperate(os.path.join(self.args.data_dir, "test12"))

        elif dataset_name == 'DID':
            dataset = self.DID_dataset(datasetName="pix2pix_val", dataroot=os.path.join(self.args.data_dir, "DID-MDN-datasets/DID-MDN-test"),
                                     batchSize=1, workers=1)
        elif dataset_name == 'DDN':
            dataset = DDN_Dataset(os.path.join(self.args.data_dir, "DDN/Rain1400"), self.patch_size, eval=True)

        elif dataset_name[:-1] == 'PReNetData':
            dataset_name = "Rain100" + dataset_name[-1]
            args.dir_data = os.path.join(self.args.data_dir, dataset_name)
            dataset = SRData(self.args, 'train', dataset_name="RainTrain" + dataset_name[-1], train=False)

        elif dataset_name in ["Rain100L", "Rain100H"]:
            args.dir_data = '/'.join([self.args.data_dir, dataset_name, 'test'])
            dataset = RainHeavyTest(self.args)

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=1, drop_last=False, sampler=sampler)
        return self.dataloaders[dataset_name], sampler

    def DID_dataset(self, datasetName, dataroot, batchSize=64, workers=4, shuffle=True, seed=None):
        # import pdb; pdb.set_trace()
        if datasetName == 'pix2pix':
            # commonDataset = pix2pix
            from pix2pix import pix2pix as commonDataset
            # import transforms.pix2pix as transforms
            # from datasets.pix2pix import pix2pix as commonDataset
            # import transforms.pix2pix as transforms
        elif datasetName == 'pix2pix_val':
            # commonDataset = pix2pix_val
            from DID_DATA import pix2pix_val as commonDataset
            # import transforms.pix2pix as transforms
            # from datasets.pix2pix_val import pix2pix_val as commonDataset
            # import transforms.pix2pix as transforms
        elif datasetName == 'pix2pix_class':
            # commonDataset = pix2pix_class
            from DID_DATA import pix2pix_class as commonDataset
            # import transforms.pix2pix as transforms
            # from datasets.pix2pix_class import pix2pix as commonDataset
            # import transforms.pix2pix as transforms

        dataset = commonDataset(root=dataroot,
                                patch_size=self.patch_size,
                                transform=None,
                                # transforms.Compose([
                                #     transforms.Scale(originalSize),
                                #     transforms.RandomCrop(imageSize),
                                #     transforms.RandomHorizontalFlip(),
                                #     transforms.ToTensor(),
                                #     transforms.Normalize(mean, std),])
                                seed=seed)

        # else:
        #     dataset = commonDataset(root=dataroot,
        #                             transform=transforms.Compose([
        #                                 transforms.Scale(originalSize),
        #                                 # transforms.CenterCrop(imageSize),
        #                                 transforms.ToTensor(),
        #                                 transforms.Normalize(mean, std),
        #                             ]),
        #                             seed=seed)

        # dataloader = DataLoader(dataset,
        #                         batch_size=batchSize,
        #                         shuffle=shuffle,
        #                         persistent_workers=workers > 0,
        #                         num_workers=int(workers))
        return dataset#dataloader, None

# test12
class DataLoaderSeperate(Dataset):
    def __init__(self, rgb_dir):
        super(DataLoaderSeperate, self).__init__()

        # gt_dir = 'norain'  # groundtruth
        # input_dir = 'rain'  # input
        gt_dir = 'groundtruth'
        input_dir = 'rainy'

        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir).replace('\\', '/')))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir).replace('\\', '/')))
        # # noisy_files = []
        # # for file in clean_files:
        #     # changed_path = file.replace('\\', '/').split('/')
        #     # print(changed_path)
        #     # changed_path = changed_path[2:]
        #     # clean_files.append('/'.join(changed_path))
        #     # print(clean_files[-1], file)
        #
        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files]
        # self.clean_filenames = clean_files
        # self.noisy_filenames = noisy_files
        print(self.clean_filenames)
        print(self.noisy_filenames)
        # self.root_dir = os.path.join(rgb_dir, "train")
        # self.mat_files = os.listdir(self.root_dir)

        # self.img_options = img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target clean_filenames

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index = index % self.tar_size
        clean = torch.from_numpy(np.float32(imageio.imread(self.clean_filenames[tar_index])) / 255.0).permute(2, 0, 1)
        noisy = torch.from_numpy(np.float32(imageio.imread(self.noisy_filenames[tar_index])) / 255.0).permute(2, 0, 1)
        # file_name = self.mat_files[tar_index]
        # img_file = os.path.join(self.root_dir, file_name)

        # img_pair = torch.from_numpy(np.float32(load_img(img_file)))
        # h, ww, c = img_pair.shape
        # w = int(ww / 2)
        # clean = img_pair[:, :w, :]
        # noisy = img_pair[:, w:, :]
        # clean = clean.permute(2, 0, 1)
        # noisy = noisy.permute(2, 0, 1)

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        # Crop Input and Target
        ps = 48
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        # if H - ps == 0:
        #     r = 0
        #     c = 0
        # else:
        #     r = np.random.randint(0, H - ps)
        #     c = np.random.randint(0, W - ps)
        # clean = clean[:, r:r + ps, c:c + ps]
        # noisy = noisy[:, r:r + ps, c:c + ps]

        # apply_trans = transforms_aug[random.getrandbits(3)]

        # clean = getattr(augment, apply_trans)(clean)
        # noisy = getattr(augment, apply_trans)(noisy)

        return {'O': noisy, 'B': clean, 'filename': clean_filename}  # clean_filename, noisy_filename

# rain200
class TrainValDataset(Dataset):
    def __init__(self, data_dir, name, patch_size):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.root_dir = os.path.join(data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        # self.mat_files_derain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[0]))
        # self.mat_files_rain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[1]))
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)
        print("file_num", self.file_num)

    def __len__(self):
        return self.file_num
    # 1474 2425 864
    def __getitem__(self, idx):
        file_name = self.mat_files[idx % self.file_num]
        img_file = os.path.join(self.root_dir, file_name)
        try:
            img_pair = cv2.imread(img_file).astype(np.float32) / 255
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
        sample = {'index': idx, 'O': O, 'B': B, 'filename': file_name}

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
        O = img_pair[r: r + p_h, c + w: c + p_w + w]  # rain
        B = img_pair[r: r + p_h, c: c + p_w]  # norain
        # cv2.imshow("O", O)
        # cv2.imshow("B", B)
        # cv2.waitKey(1000)

        # print("coord:", [r, r + p_h, c + w, c + p_w + w, r, r + p_h, c, c + p_w])

        return O, B

    def augment(self, *args, hflip=True, rot=True):
        """common"""
        hflip = hflip and random.random() < 0.5
        # vflip = rot and random.random() < 0.5
        # rot90 = rot and random.random() < 0.5

        def _augment(img):
            """common"""
            if hflip:
                img = img[:, ::-1, :]
                # img = np.flip(img, axis=1)
            # if vflip:
            #     img = img[::-1, :, :]
            #     # img = np.flip(img, axis=0)
            # if rot90:
            #     img = img.transpose(1, 0, 2)
            return img

        return _augment(args[0]), _augment(args[1])  # [_augment(a) for a in args]

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

# rain200
class TestDatasetV2(Dataset):
    def __init__(self, data_dir, name):
        self.root_dir = os.path.join(data_dir, name)
        print("loading..", self.root_dir)
        self.mat_files = os.listdir(self.root_dir)
        # self.patch_size = settings.patch_size
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

        h, ww, c = img_pair.shape
        w = int(ww / 2)
        B = img_pair[:, : w] #norain
        O = img_pair[:, w:ww] #rain

        O = np.transpose(O, (2, 0, 1))
        B = np.transpose(B, (2, 0, 1))
        # sample = {'O': torch.from_numpy(O), 'B': torch.from_numpy(B)}
        O = torch.from_numpy(np.ascontiguousarray(O))
        B = torch.from_numpy(np.ascontiguousarray(B))
        sample = {'index': idx, 'O': O, 'B': B, 'filename': file_name.split('.')[0]}

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

if __name__ == '__main__':
    import argparse
    import random
    from torch.backends import cudnn
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    # * Logger
    parser.add_argument('--out_dir', metavar='DIR', default='./results',
                        help='path to save model')
    parser.add_argument('--log_dir', metavar='DIR', default='logs',
                        help='path to save log')
    parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                        help='useless in this script.')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='DETR')
    parser.add_argument('--print-freq', '-p', default=1, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--workers', default=1, type=int)
    parser.add_argument('--patch_size', type=int, default=100,
                        help='image2patch, set to model and dataset')
    parser.add_argument('-b', '--batch-size', default=1, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    parser.add_argument('--eval', default=None, type=str,
                        choices=[None, 'rain200H', 'rain100L', 'rain200H', 'rain100H',
                                 'test12', 'real', 'DID', 'SPA', 'DDN'],
                        help="performing evalution for patch2entire")
    parser.add_argument('--test_every', type=int, default=1000,
                        help='do test per every N batches')
    parser.add_argument('--model', default='ipt',
                        help='model name')
    parser.add_argument('--n_colors', type=int, default=3,
                        help='number of color channels to use')
    parser.add_argument('--rgb_range', type=int, default=255,
                        help='maximum value of RGB')
    parser.add_argument('--no_augment', action='store_true',
                        help='do not use data augmentation')
    parser.add_argument('--ext', type=str, default='sep',
                        help='dataset file extension')
    args = parser.parse_args()
    args.scale = [1]
    args.data_dir = "D:/Datasets/derain"

    sess = derainSession(args)
    # loader, _ = sess.get_dataloader('Rain100L', None)
    loader, _ = sess.get_eval_dataloader('Rain100H', None)

    # pools = [2, 2, 2, 2, 2]
    # depth = len(pools)

    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.deterministic = True
    plt.ion()
    fig, axes = plt.subplots(ncols=2, nrows=1)
    for batch in loader:
        O, B = batch['O'], batch['B']
        # O, B = batch[0], batch[1]
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
