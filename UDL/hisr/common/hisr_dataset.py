import os
import warnings

import cv2
import imageio
import numpy as np
import torch
# from numpy.random import RandomState
import random
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import math
import torch.nn as nn
from UDL.hisr.common.dataUPHSI import DatasetFromHdf5
from UDL.derain.common.data.common import resize_image
import h5py
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

class HISRSession():
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        self.patch_size = args.patch_size
        self.writers = {}
        self.args = args

    def get_dataloader(self, dataset_name, distributed):

        if dataset_name == "cave_x4":
            print('/'.join([self.args.data_dir, f'{dataset_name}', f'train_cave(with_up)x4.h5']))
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'{dataset_name}', f'train_cave(with_up)x4.h5']))
        elif dataset_name == "harvard_x4":
            dataset = DatasetFromHdf5(
                '/'.join([self.args.data_dir, f'/{dataset_name}', 'train_harvard(with_up)x4_rgb.h5']))
        elif dataset_name == "harvard_x8":
            dataset = DatasetFromHdf5(
                '/'.join([self.args.data_dir, f'/{dataset_name}', 'train_harvard(with_up)x8_rgb.h5']))
        elif dataset_name == "cave_x8":
            dataset = DatasetFromHdf5(
                '/'.join([self.args.data_dir, f'/{dataset_name}', 'train_cave(with_up)x8_rgb_16.h5']))
        elif dataset_name == 'Chikusei_x4':
            dataset=DatasetFromHdf5(
                    '/'.join([self.args.data_dir, f'/{dataset_name}', 'train_Chikusei.h5']))
        elif dataset_name == 'pavia_x4':
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'/{dataset_name}', 'Pavia-train64(double_max_normalization).h5']))

        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError
        # dataset = TestDataset(dataset_name)
        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.samples_per_gpu,
                           persistent_workers=(True if self.workers_per_gpu > 0 else False),
                           shuffle=(sampler is None), num_workers=self.workers_per_gpu, drop_last=False, sampler=sampler)

        return self.dataloaders[dataset_name], sampler

    def get_test_dataloader(self, dataset_name, distributed):
        '''
        对于patch的训练数据，训练中的测试是没有意义的，精度有差距
        '''
        # dataset = TestDataset(dataset_name)
        if dataset_name == "cave_x4":
            print('/'.join([self.args.data_dir, f'{dataset_name}', f'train_cave(with_up)x4.h5']))
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'{dataset_name}', f'validation_cave(with_up)x4.h5']))
        elif dataset_name == "harvard_x4":
            dataset = DatasetFromHdf5(
                '/'.join([self.args.data_dir, f'/{dataset_name}', 'train_harvard(with_up)x4_rgb.h5']))
        elif dataset_name == "cave_x8":
            dataset = DatasetFromHdf5(
                '/'.join([self.args.data_dir, f'/{dataset_name}', 'test_cave(with_up)x8_rgb.h5']))
        elif dataset_name == 'Chikusei_x4':
            dataset=DatasetFromHdf5(
                    '/'.join([self.args.data_dir, f'/{dataset_name}', 'train_Chikusei.h5']))
        elif dataset_name == 'pavia_x4':
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'/{dataset_name}', 'Pavia-validation64(double_max_normalization).h5']))
        dataset = TrainValDataset(dataset_name, self.patch_size)
        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)

        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.samples_per_gpu,
                           shuffle=False, num_workers=self.workers_per_gpu, drop_last=False, sampler=sampler)

        return self.dataloaders[dataset_name], sampler

    def get_eval_dataloader(self, dataset_name, distributed):
        # global data_dir

        if dataset_name == "cave_x4":
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'{dataset_name}', f'test_cave(with_up)x4.h5']))
        elif dataset_name == "cave_x8":
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'{dataset_name}', 'test_cave(with_up)x8_rgb.h5']))
        elif dataset_name == "harvard_x4":
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'{dataset_name}', 'test_harvard(with_up)x4_rgb.h5']))
        elif dataset_name == "harvard_x8":
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'{dataset_name}', 'test_harvard(with_up)x8_rgb.h5']))
        elif dataset_name == 'Chikusei_x4':
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'/{dataset_name}', 'test_Chikusei.h5']))
        elif dataset_name == 'pavia_x4':
            dataset = DatasetFromHdf5('/'.join([self.args.data_dir, f'/{dataset_name}', 'Pavia-test256(double_max_normalization).h5']))

        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError

        sampler = None
        if distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        # dataset = TrainValDataset(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=1,
                           shuffle=False, num_workers=0, drop_last=False, sampler=sampler)
        return self.dataloaders[dataset_name], sampler

    def DID_dataset(self, datasetName, dataroot, batchSize=64, workers=4, shuffle=True, seed=None):
        # import pdb; pdb.set_trace()
        if datasetName == 'pix2pix':
            # commonDataset = pix2pix
            from UDL.derain.common.pix2pix import pix2pix as commonDataset
            # import transforms.pix2pix as transforms
            # from datasets.pix2pix import pix2pix as commonDataset
            # import transforms.pix2pix as transforms
        elif datasetName == 'pix2pix_val':
            # commonDataset = pix2pix_val
            from UDL.derain.common.DID_DATA import pix2pix_val as commonDataset
            # import transforms.pix2pix as transforms
            # from datasets.pix2pix_val import pix2pix_val as commonDataset
            # import transforms.pix2pix as transforms
        elif datasetName == 'pix2pix_class':
            # commonDataset = pix2pix_class
            from UDL.derain.common.DID_DATA import pix2pix_class as commonDataset
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
########################################################################
# rain200
########################################################################

# class Rain200_H5Dataset(Dataset):
#     def __init__(self, data_path, patch_size, adjust_size_mode="patch"):
#         super(Dataset, self).__init__()
#
#         self.data_path = data_path
#
#
#         self.target_path = os.path.join(self.data_path, 'train_target64.h5')
#         self.input_path = os.path.join(self.data_path, 'train_input64.h5')
#
#         self.target_h5f = h5py.File(self.target_path, 'r')
#         self.input_h5f = h5py.File(self.input_path, 'r')
#
#         self.keys = list(self.target_h5f.keys())
#         random.shuffle(self.keys)
#         self.target_h5f.close()
#         self.input_h5f.close()
#
#         self.input_h5f = None
#
#         print("Data: ", len(self.keys))
#
#     def __len__(self):
#         return len(self.keys)
#
#     def __getitem__(self, index):
#         if self.input_h5f is None:
#             self.target_h5f = h5py.File(self.target_path, 'r', swmr=True, libver='latest')
#             self.input_h5f = h5py.File(self.input_path, 'r', swmr=True, libver='latest')
#
#         key = self.keys[index]
#         target = np.array(self.target_h5f[key])
#         input = np.array(self.input_h5f[key])
#
#         # target_h5f.close()
#         # input_h5f.close()
#         sample = {'index': index, 'O': input, 'B': target}#, 'filename': file_name}
#
#         return sample#torch.Tensor(input), torch.Tensor(target)

class Rain200_H5Dataset(Dataset):
    def __init__(self, data_path, patch_size, adjust_size_mode="patch"):
        super(Dataset, self).__init__()
        print("using Rain200_H5Dataset")
        self.data_path = data_path + "/train_input.h5"
        self.O = None
        with h5py.File(self.data_path, 'r') as data:
            self.len = data["B"].shape[0]

        # random.shuffle(self.keys)
        # self.target_h5f.close()
        # self.input_h5f.close()

        print("Data: ", self.len)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        if self.O is None:
            self.data = h5py.File(self.data_path, 'r', swmr=True, libver='latest')
            self.O = self.data['O']
            self.B = self.data['B']
            del self.data


        target = np.array(self.O[index])
        input = np.array(self.B[index])

        # target_h5f.close()
        # input_h5f.close()
        sample = {'index': index, 'O': input, 'B': target}#, 'filename': file_name}

        return sample#torch.Tensor(input), torch.Tensor(target)


class TrainValDataset(Dataset):
    def __init__(self, data_dir, name, patch_size, adjust_size_mode="patch"):
        super().__init__()
        # self.rand_state = RandomState(66)
        self.root_dir = os.path.join(data_dir, name)
        self.mat_files = os.listdir(self.root_dir)
        # self.mat_files_derain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[0]))
        # self.mat_files_rain = os.listdir(os.path.join(self.root_dir, self.mat_files_dir[1]))
        self.patch_size = patch_size
        self.file_num = len(self.mat_files)
        print("file_num", self.file_num)
        self.adjust_size_mode = adjust_size_mode

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
        if self.adjust_size_mode == "patch":
            O, B = self.crop(img_pair, aug=True)
        elif self.adjust_size_mode == "resize":
            h, ww, c = img_pair.shape
            w = ww // 2
            O = img_pair[:, w:]
            B = img_pair[:, :w]
            O, B = resize_image(O, B, patch_size=self.patch_size)
        else:
            print(f"{self.adjust_size_mode} is not supported in {__file__}:line 218")
            raise NotImplementedError

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
        # hflip = hflip and random.random() < 0.5
        vflip = rot and random.random() < 0.5
        # rot90 = rot and random.random() < 0.5

        def _augment(img):
            """common"""
            # if hflip:
            #     img = img[:, ::-1, :]
                # img = np.flip(img, axis=1)
            if vflip:
                img = img[::-1, :, :]
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
    parser.add_argument('--workers_per_gpu', default=1, type=int)
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
    args.samples_per_gpu = 1
    args.adjust_size_mode = "patch"

    sess = derainSession(args)
    # loader, _ = sess.get_dataloader('Rain100L', None)
    loader, _ = sess.get_eval_dataloader('Rain200H', None)

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
