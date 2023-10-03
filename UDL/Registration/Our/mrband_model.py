# import copy
import copy

import torch
import torch.nn as nn
import math
from variance_sacling_initializer import variance_scaling_initializer
from evaluate import analysis_accu
import scipy.io as sio


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                variance_scaling_initializer(m.weight)  # method 1: initialization
                # nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')  # method 2: initialization
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.BatchNorm2d):  ## initialization for BN
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):  ## initialization for nn.Linear
                # variance_scaling_initializer(m.weight)
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)


class DiCNN(nn.Module):
    def __init__(self, registration):
        super(DiCNN, self).__init__()

        self.registration = registration

        channel = 64
        spectral_num = 8
        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=channel, kernel_size=3, stride=1, padding=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=channel, out_channels=spectral_num, kernel_size=3, stride=1, padding=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x, y, registration=True):  # x= pan; y = lms

        pan_refine, flow = self.registration(x, y, registration=True)
        # self.registration.requires_grad = False
        if registration:
            input1 = torch.cat((y, pan_refine), 1)

            rs = self.relu(self.conv1(input1))
            rs = self.relu(self.conv2(rs))
            out = self.conv3(rs)
            output = y + out

            return output, pan_refine, flow


        else:
            input1 = torch.cat((y, pan_refine), 1)  # Bsx9x64x64

            rs = self.relu(self.conv1(input1))
            rs = self.relu(self.conv2(rs))
            out = self.conv3(rs)
            output = y + out

            return output, pan_refine


##############

def load_gt_compared(file_path_gt,file_path_compared, dataset_name):
    data1 = sio.loadmat(file_path_gt)  # HxWxC
    data2 = sio.loadmat(file_path_compared)
    try:
        gt = torch.from_numpy(data1['gt']/2047.0)
    except KeyError:
        print(data1.keys())
    print(data1.keys())
    keys = "output_dmdnet_" + dataset_name
    compared_data = torch.from_numpy(data2[keys]*2047.0)
    return gt, compared_data

def load_set_V2(file_path):
    data = sio.loadmat(file_path)  # HxWxC

    # tensor type:
    lms = torch.from_numpy(data['lms'] / 2047.0).permute(2, 0, 1)  # CxHxW = 8x256x256
    ms = torch.from_numpy(data['ms'] / 2047.0).permute(2, 0, 1)  # CxHxW= 8x64x64
    ms = ms.unsqueeze(dim=0)

    pan = torch.from_numpy(data['pan'] / 2047.0)  # HxW = 256x256

    return lms, ms, pan


def test_compared_model(args):
    others_index_list = {}
    # self.others_ERGAS_list = {}
    test_gt = []
    test_ms = []
    test_pan = []
    test_lms = []
    for file_path, file_path_compared in zip(
            args.test_data, args.test_compared_data):
        dataset_name = file_path_compared.split("/")[-1][14:-4]

        lms, ms, pan = load_set_V2(file_path)
        test_ms.append(ms.cuda(args.gpu).float())  # convert to tensor type: 1xCxHxW (unsqueeze(dim=0))
        test_pan.append(
            pan.cuda(args.gpu).unsqueeze(dim=0).unsqueeze(dim=1).float())  # convert to tensor type: 1x1xHxW
        test_lms.append(lms.cuda(args.gpu).unsqueeze(dim=0).float())
        gt, test_compared_result = load_gt_compared(file_path, file_path_compared, dataset_name)  ##compared_result
        gt = (gt * 2047).cuda(args.gpu)
        test_gt.append(gt)
        test_compared_result = test_compared_result.cuda(args.gpu)
        others_index = analysis_accu(gt, test_compared_result, 4, compute_index=["SAM", "ERGAS"])
        others_index_list[dataset_name] = [others_index["SAM"], others_index["ERGAS"]]

    # index_list = copy.deepcopy(others_index_list)

    return test_lms, test_ms, test_pan, test_gt, others_index_list#, index_list


class Tester():
    def __init__(self, args):
        self.test_lms, self.test_ms, self.test_pan, self.test_gt, \
        self.others_index_list = test_compared_model(args)

    def __call__(self, model):
        for idx, (dataset_name, others_indexes) in enumerate(self.others_index_list.items()):
            out2, _ = model(self.test_pan[idx], self.test_lms[idx])
            result_our = torch.squeeze(out2).permute(1, 2, 0)
            result_our = result_our * 2047
            accu = analysis_accu(self.test_gt[idx], result_our, 4, compute_index=["CC", "SAM", "PSNR", "ERGAS", "SSIM"])
            # self.index_list[dataset_name] = [our_SAM, our_ERGAS]
            print(f'[{dataset_name}]:'
                       f'our_CC: {accu["CC"]}, our_PSNR: {accu["PSNR"]}, '
                       f'our_SSIM: {accu["SSIM"]},\n'
                       f'our_SAM: {accu["SAM"]} our_ERGAS: {accu["ERGAS"]}')
            print(
                '[{}]: dmdnet_SAM: {} dmdnet_ERGAS: {}'.format(dataset_name, others_indexes[0], others_indexes[1]))
        # return self.index_list

import os
from model import VxmDense
def test(args, model_path):

    if os.path.isfile(model_path):
        print("loading model")
        checkpoint = torch.load(model_path)
        registration = VxmDense(inshape=[256, 256])

        net = DiCNN(registration).cuda()

        pretrained_dict = {}
        for module in checkpoint['state_dict'].items():
            #pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'grid' not in module}
            # 2. overwrite entries in the existing state dict
            k, v = module
            if 'grid' not in k:
                pretrained_dict.update({k: v})

        net.load_state_dict(pretrained_dict, strict=False)

    else:
        print("loading error")
        raise NotImplementedError


    evaluator = Tester(args)

    evaluator(net)












if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Pansharpening with rehistration Testing')
    parser.add_argument('--gpu', default=0, type=int,  # os.environ["CUDA_VISIBLE_DEVICES"]的映射约束下的顺序
                        help='GPU id to use.')

    args = parser.parse_args()

    args.test_data = ["./newdata6.mat", "./newdata7.mat"]
    args.test_compared_data = ["./00-compared/output_dmdnet_newdata6.mat",
                               "./00-compared/output_dmdnet_newdata7.mat"]

    model_path = "./results/wv3/Vmx/Unet/model_2021-05-19-23-48/531.pth.tar"
    #our_SAM: 14.478013660689342 our_ERGAS: 9.35779446477504
    test(args, model_path)