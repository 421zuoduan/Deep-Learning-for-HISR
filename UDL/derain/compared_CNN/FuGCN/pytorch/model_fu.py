# This is a pytorch version code of this paper:
# X. Fu, Q. Qi, Z.-J. Zha, Y. Zhu, X. Ding. “Rain Streak Removal via Dual Graph Convolutional Network”, AAAI, 2021.

import torch
import torch.nn as nn
from torch import optim

class basic(nn.Module):
    def __init__(self, in_ch=3, out_ch=72):
        super(basic, self).__init__()
        self.basic_fea0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.basic_fea1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, input):
        basic_fea0 = self.basic_fea0(input)
        basic_fea1 = self.basic_fea1(basic_fea0)
        return basic_fea1


class BasicUnit(nn.Module):
    def __init__(self, in_ch=72, out_ch=72):
        super(BasicUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, dilation=3, padding=3)
        self.conv4 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=1, dilation=3, padding=3)
        self.F_DCM = nn.Conv2d(in_channels=out_ch + out_ch + out_ch + out_ch, out_channels=out_ch, kernel_size=1,
                               stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv1 = self.relu(self.conv1(input))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(input))
        conv4 = self.relu(self.conv4(conv3))
        tmp = torch.cat([conv1, conv2, conv3, conv4], dim=1)
        F_DCM = self.relu(self.F_DCM(tmp))
        return F_DCM + input


##完整backbone
class Net(nn.Module):
    def __init__(self, channels=72, criterion=None):
        super(Net, self).__init__()

        self.criterion = criterion

        self.basic = basic()
        self.encode0 = BasicUnit()
        self.encode1 = BasicUnit()
        self.encode2 = BasicUnit()
        self.encode3 = BasicUnit()
        self.encode4 = BasicUnit()
        self.midle_layer = BasicUnit()
        self.deconv4 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder4 = BasicUnit()
        self.deconv3 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder3 = BasicUnit()
        self.deconv2 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder2 = BasicUnit()
        self.deconv1 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder1 = BasicUnit()
        self.deconv0 = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=1, stride=1, padding=0)
        self.decoder0 = BasicUnit()
        self.decoding_end = nn.Conv2d(in_channels=channels * 2, out_channels=channels, kernel_size=3, stride=1,
                                      padding=1)
        self.res = nn.Conv2d(in_channels=channels, out_channels=3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        basic = self.basic(input)
        encode0 = self.encode0(basic)
        encode1 = self.encode1(encode0)
        encode2 = self.encode2(encode1)
        encode3 = self.encode3(encode2)
        encode4 = self.encode4(encode3)
        middle = self.midle_layer(encode4)
        decoder4 = self.deconv4(torch.cat([middle, encode4], dim=1))
        decoder4 = self.decoder4(decoder4)
        decoder3 = self.deconv3(torch.cat([decoder4, encode3], dim=1))
        decoder3 = self.decoder3(decoder3)
        decoder2 = self.deconv2(torch.cat([decoder3, encode2], dim=1))
        decoder2 = self.decoder2(decoder2)
        decoder1 = self.deconv1(torch.cat([decoder2, encode1], dim=1))
        decoder1 = self.decoder1(decoder1)
        decoder0 = self.deconv0(torch.cat([decoder1, encode0], dim=1))
        decoder0 = self.decoder0(decoder0)

        decoder_end = self.relu(self.decoding_end(torch.cat([decoder0, basic], dim=1)))
        res = self.res(decoder_end)

        return res + input

    def train_step(self, batch, *args, **kwargs):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        derain = self(O)

        loss = self.criterion(derain, B, *args, **kwargs)

        return derain, loss

    def eval_step(self, batch):
        O, B = batch['O'].cuda(), batch['B'].cuda()
        derain = self(O)

        return derain, B

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, losses, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: n able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relatiumber of object categories, omitting the special no-object category
            matcher: moduleve classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.weight_dict = weight_dict
        self.losses = losses
        self.loss_dicts = {}

    def forward(self, outputs, targets, *args, **kwargs):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # lms = kwargs.get('lms')
        # outputs = outputs + lms  # outputs: hp_sr
        # Compute all the requested losses

        for k in self.losses.keys():
            # k, loss = loss_dict
            if k == 'Loss':
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets)})
            else:
                loss = self.losses[k]
                loss_dicts = loss(outputs, targets, *args)
                if isinstance(loss_dicts, dict):
                    self.loss_dicts.update(loss(outputs, targets, *args))
                else:
                    self.loss_dicts.update({k: loss(outputs, targets, *args)})

        return self.loss_dicts

def build_FuGCN(args):
    scheduler = None

    loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
    weight_dict = {'Loss': 1}
    losses = {'Loss': loss}
    criterion = SetCriterion(losses, weight_dict)
    model = Net(criterion=criterion).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)  ## optimizer 1: Adam

    return model, criterion, optimizer, scheduler

