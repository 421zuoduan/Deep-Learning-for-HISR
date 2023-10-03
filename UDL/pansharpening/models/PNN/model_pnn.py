import torch
import torch.nn as nn
import math
from UDL.Basis.variance_sacling_initializer import variance_scaling_initializer

"""
CHANGES: no padding in conv_layers
    1. rows 43-48: padding set to zero
    2. rows 54-54: prototype of function forward is changed (takes x=cat(lms,pan) only)
"""


# -------------Initialization----------------------------------------
def init_weights(*modules):
    for module in modules:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):  ## initialization for Conv2d
                # try:
                #     import tensorflow as tf
                #     tensor = tf.get_variable(shape=m.weight.shape, initializer=tf.variance_scaling_initializer(seed=1))
                #     m.weight.data = tensor.eval()
                # except:
                #     print("try error, run variance_scaling_initializer")
                # variance_scaling_initializer(m.weight)
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


class PNN(nn.Module):
    def __init__(self, spectral_num, criterion, channel=64):
        super(PNN, self).__init__()

        self.criterion = criterion

        # ConvTranspose2d: output = (input - 1)*stride + outpading - 2*padding + kernelsize
        self.conv1 = nn.Conv2d(in_channels=spectral_num + 1, out_channels=channel, kernel_size=9, stride=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=channel, out_channels=32, kernel_size=5, stride=1,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=spectral_num, kernel_size=5, stride=1,
                               bias=True)

        self.relu = nn.ReLU(inplace=True)

        # init_weights(self.conv1, self.conv2, self.conv3)

    def forward(self, x):  # x = cat(lms,pan)
        input1 = x  # Bsx9x64x64

        rs = self.relu(self.conv1(input1))
        rs = self.relu(self.conv2(rs))
        output = self.conv3(rs)

        return output

    def train_step(self, data, *args, **kwargs):
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        blk = self.blk

        gt = gt[:, :, blk:-blk, blk:-blk]
        lms = torch.cat([lms, pan], dim=1)

        sr = self(lms)

        loss = self.criterion(sr, gt, *args, **kwargs)

        return sr, loss

    def eval_step(self, data, *args, **kwargs):
        blk = self.blk
        gt, lms, ms, pan = data['gt'].cuda(), data['lms'].cuda(), \
                           data['ms'].cuda(), data['pan'].cuda()
        lms = torch.cat([lms, pan], dim=1)
        sr = self(lms)

        return sr, gt[:, blk:-blk, blk:-blk, :]

    @classmethod
    def set_blk(cls, blk):
        cls.blk = blk

# ----------------- End-Main-Part ------------------------------------
