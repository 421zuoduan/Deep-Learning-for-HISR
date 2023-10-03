import torch.utils.data as data
import torch
import h5py
import cv2
import numpy as np

class Dataset_Pro(data.Dataset):
    def __init__(self, file_path):
        super(Dataset_Pro, self).__init__()
        data = h5py.File(file_path, 'r')  # NxCxHxW = 0x1x2x3=8806x8x64x64
        print(data.keys())

        # tensor type:
        gt = data["gt"][...]  # convert to np tpye for CV2.filter
        self.gt = np.array(gt, dtype=np.float32) / 2047.
        print(self.gt.shape)

        pan = data["pan"][...]  # convert to np tpye for CV2.filter
        self.pan = np.array(pan, dtype=np.float32) / 2047.
        print(self.pan.shape)

        lms = data["lms"][...]  # convert to np tpye for CV2.filter
        self.lms = np.array(lms, dtype=np.float32) / 2047.
        print(self.lms.shape)



    #####必要函数
    def __getitem__(self, index):
        return {'gt': torch.from_numpy(self.gt[index, :, :, :]).float(),
                'pan': torch.from_numpy(self.pan[index, :, :, :]).float(),
                'lms': torch.from_numpy(self.lms[index, :, :, :]).float()}


            #####必要函数
    def __len__(self):
        return self.gt.shape[0]
