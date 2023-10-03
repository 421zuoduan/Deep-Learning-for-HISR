import os
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from .model_fusionnet import FusionNet
import numpy as np
from UDL.Basis.auxiliary import MetricLogger, SmoothedValue, log_string
from UDL.Basis.framework import get_grad_norm
from UDL.Basis.dist_utils import reduce_mean
# from ..evaluation.ps_evaluate import testToPanshaprening
# from evaluate import analysis_accu

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



def build_fusionnet(args):
    scheduler = None
    if "wv" in args.dataset:
        spectral_num = 8
    else:
        spectral_num = 4


    loss = nn.MSELoss(size_average=True).cuda()  ## Define the Loss function
    weight_dict = {'Loss': 1}
    losses = {'Loss': loss}
    criterion = SetCriterion(losses, weight_dict)
    model = FusionNet(spectral_num, criterion).cuda()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)   ## optimizer 1: Adam

    return model, criterion, optimizer, scheduler
#
# class testFusionNet(testToPanshaprening):
#     def __init__(self, model,
#                  file_path="wuxiao/new_data6.mat",
#                  file_path_compared="./output_dmdnet_newdata6.mat"):
#         super().__init__(model, file_path, file_path_compared)
#
#     def __call__(self):
#         with torch.no_grad():
#             if self.loader is None:
#                 out2 = self.model([self.test_lms, self.test_pan])#self.test_pan, self.test_lms, self.test_mms, self.test_ms)
#             else:
#                 out2 = self.model()
#         result_our = torch.squeeze(out2).permute(1, 2, 0)
#         result_our = result_our * 2047
#         our_CC, our_PSNR, our_SSIM, our_SAM, our_ERGAS = analysis_accu(self.gt, result_our, 4)
#         log_string(f'our_CC: {our_CC}, our_PSNR: {-our_PSNR}, '
#                    f'our_SSIM: {our_SSIM},\n'
#                    f'our_SAM: {our_SAM} our_ERGAS: {our_ERGAS}')
#         # log_string('dmdnet_SAM: {} dmdnet_ERGAS: {}'.format(self.others_SAM, self.others_ERGAS))
