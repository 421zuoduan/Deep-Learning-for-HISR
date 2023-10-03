import torch.utils.data as data
import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader

class H5Dataset(data.Dataset):
    def __init__(self, file_path):
        super(H5Dataset, self).__init__()
        self.file_path = file_path
        self.dataset = None
        with h5py.File(self.file_path, 'r') as f:
            self.dataset_len = len(f["gt"])
    def __getitem__(self, index):
        if self.dataset is None:
            self.dataset = h5py.File(self.file_path, 'r')
        #TypeError: can't convert np.ndarray of type numpy.longdouble.
        # The only supported types are: float64, float32, float16, int64, int32, int16, int8, uint8, and bool
        # self.gt = torch.from_numpy(self.dataset.get("gt")[...].astype(np.float32)) / 2047.0  # NxCxHxW
        # self.ms = torch.from_numpy(self.dataset.get("ms")[...].astype(np.float32)) / 2047.0
        # self.lms = torch.from_numpy(self.dataset.get("lms")[...].astype(np.float32)) / 2047.0
        # self.pan = torch.from_numpy(self.dataset.get("pan")[...].astype(np.float32)) / 2047.0
        # return self.gt[index, :, :, :], \
        #        self.lms[index, :, :, :], \
        #        self.ms[index, :, :, :], \
        #        self.pan[index, :, :, :]

    def __len__(self):
        return self.dataset_len


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path, 'r')
        try:
            self.gt = torch.from_numpy(dataset["gt"][...]).float() / 2047.0  # NxCxHxW
            self.ms = torch.from_numpy(dataset["ms"][...]).float() / 2047.0
            self.lms = torch.from_numpy(dataset["lms"][...]).float() / 2047.0
            self.pan = torch.from_numpy(dataset["pan"][...]).float() / 2047.0
        except Exception:
            self.gt = np.array(dataset["gt"][...], dtype=np.float32)
            self.ms = np.array(dataset["ms"][...], dtype=np.float32)
            self.lms = np.array(dataset["lms"][...], dtype=np.float32)
            self.pan = np.array(dataset["pan"][...], dtype=np.float32)
            self.gt = torch.from_numpy(self.gt).mean(dim=1, keepdims=True)
            self.ms = torch.from_numpy(self.ms)
            self.lms = torch.from_numpy(self.lms)
            self.pan = torch.from_numpy(self.pan)

        print("loading data: \n"
              "gt:     {} \n"
              "lms:    {} \n"
              "ms_hp:  {} \n"
              "pan_hp: {} \n".format(self.gt.size(), self.lms.size(), self.ms.size(), self.pan.size()))

    #####必要函数
    def __getitem__(self, index):
        return self.gt[index, :, :, :], \
               self.lms[index, :, :, :], \
               self.ms[index, :, :, :], \
               self.pan[index, :, :, :]

    #####必要函数
    def __len__(self):
        return self.gt.shape[0]

class Session():
    def __init__(self, args):
        self.dataloaders = {}
        self.batch_size = args.batch_size
        self.num_workers = args.workers

    def get_dataloader(self, dataset_name):
        dataset = DatasetFromHdf5(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=self.batch_size,
                           shuffle=True, num_workers=self.num_workers, drop_last=True)
        return self.dataloaders[dataset_name]

    def get_test_dataloader(self, dataset_name):
        dataset = DatasetFromHdf5(dataset_name)
        if not dataset_name in self.dataloaders:
            self.dataloaders[dataset_name] = \
                DataLoader(dataset, batch_size=64,
                           shuffle=False, num_workers=1, drop_last=False)
        return self.dataloaders[dataset_name]

    # def get_eval_dataloader(self, dataset_name):
    #     dataset = TestDataset(dataset_name)
    #     # dataset = TrainValDataset(dataset_name)
    #     if not dataset_name in self.dataloaders:
    #         self.dataloaders[dataset_name] = \
    #             DataLoader(dataset, batch_size=1,
    #                        shuffle=False, num_workers=1, drop_last=False)
    #     return self.dataloaders[dataset_name]



if __name__ == "__main__":
    import argparse
    import postprocess as pp
    import matplotlib.pyplot as plt

    fig = plt.figure(1)
    parser = argparse.ArgumentParser(description='PyTorch showing data')

    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('-b', '--batch-size', default=1, type=int,  # 8
                        metavar='N', help='mini-batch size (default: 256)')
    args = parser.parse_args()
    sess = Session(args)
    train_loader = sess.get_dataloader('../../training_data/train.h5')
    val_loader = sess.get_test_dataloader('../../training_data/valid.h5')
    plt.ion()
    for idx, batch in enumerate(train_loader):
        gt = batch[0]
        lms = batch[1]
        ms = batch[2]
        pan = batch[3]


        gt = pp.showimage8(gt)
        lms = pp.showimage8(lms)
        ms = pp.showimage8(ms)


        ax1 = fig.add_subplot(221)
        ax1.set_title('gt')
        ax1.imshow(np.squeeze(gt), 'gray')
        plt.axis('off')

        ax2 = fig.add_subplot(222)
        ax2.set_title('pan')
        ax2.imshow(np.squeeze(pan), 'gray')
        plt.axis('off')
        ax3 = fig.add_subplot(223)
        ax3.set_title('ms')
        ax3.imshow(np.squeeze(ms), 'gray')
        plt.axis('off')
        ax4 = fig.add_subplot(224)
        ax4.set_title('lms')
        ax4.imshow(np.squeeze(lms), 'gray')
        plt.axis('off')

        plt.show()
        plt.savefig(f"./111/{idx}.jpg", dpi=300)
        plt.pause(0.4)
    plt.ioff()







