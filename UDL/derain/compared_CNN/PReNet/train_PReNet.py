import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DerainDataset import *
from util import *
from torch.optim.lr_scheduler import MultiStepLR
from SSIM import SSIM
from networks import *
from derain_dataset import derainSession
import matplotlib.pyplot as plt
from utils.utils import set_random_seed

# 168963
parser = argparse.ArgumentParser(description="PReNet_train")
parser.add_argument("--preprocess", type=bool, default=True, help='run prepare_data or not')
parser.add_argument("--batch_size", type=int, default=18, help="Training batch size")
parser.add_argument('--workers', default=0, type=int)
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=[30, 50, 80], help="When to decay learning rate")
parser.add_argument("--lr", type=float, default=1e-3, help="initial learning rate")
parser.add_argument("--save_path", type=str, default="logs/PReNet_test", help='path to save models and log files')
parser.add_argument("--save_freq", type=int, default=1, help='save intermediate model')
parser.add_argument("--data_dir", type=str, default="D:/Datasets/derain",
                    help='path to training data')  # datasets/train/Rain12600
parser.add_argument('--dataset', default='PReNetDataL', type=str,
                    choices=[None, 'Rain200H', 'Rain100L', 'Rain200H',
                             'Rain100H', 'PReNetDataL', 'PReNetDataH', 'DID', 'SPA', 'DDN',
                             'test12', 'real', ],
                    help="set dataset name for training"
                         "real/test12 is eval-only")

parser.add_argument("--use_gpu", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
parser.add_argument('--model', default='PReNet',
                    help='model name')

#Data
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--patch_size', type=int, default=100,
                    help='image2patch, set to model and dataset')
parser.add_argument('--test_every', type=int, default=22,
                    help='do test per every N batches')
parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                    help='train dataset name')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
opt = parser.parse_args()
opt.scale = [1]

if opt.use_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

fig, axes = plt.subplots(ncols=2, nrows=1)

'''
[121, 160, 174, 248, 91, 46, 377, 131, 335, 261, 190, 45, 88, 168, 33, 309, 158, 250, 344, 10, 26, 264, 214, 211, 75, 387, 220, 6, 233, 351, 386, 362, 363, 381, 285, 108, 179, 243, 162, 218, 301, 330, 365, 255, 107, 195, 18, 238, 113, 138, 288, 9, 188, 25, 53, 176, 227, 353, 115, 222, 154, 105, 0, 374, 117, 119, 127, 144, 13, 388, 23, 369, 272, 43, 252, 210, 62, 173, 383, 42, 159, 35, 148, 286, 331, 83, 265, 343, 294, 307, 16, 60, 51, 231, 327, 354, 183, 287, 1, 322, 48, 394, 92, 314, 364, 156, 85, 230, 11, 52, 389, 106, 373, 253, 391, 361, 166, 136, 300, 325, 328, 152, 319, 281, 221, 196, 169, 236, 128, 12, 27, 232, 312, 80, 103, 3, 392, 276, 297, 263, 147, 47, 332, 185, 216, 310, 334, 212, 98, 346, 321, 305, 256, 348, 270, 260, 382, 380, 97, 122, 371, 70, 151, 228, 65, 274, 145, 78, 208, 77, 129, 341, 31, 266, 291, 61, 95, 177, 209, 311, 7, 38, 199, 225, 342, 258, 170, 28, 4, 306, 149, 178, 94, 102, 242, 219, 347, 14, 171, 17, 376, 245, 175, 137, 241, 336, 293, 150, 197, 213, 370, 22, 356, 82, 279, 116, 329, 355, 57, 37, 110, 247, 303, 81, 390, 143, 200, 139, 93, 254, 130, 87, 163, 249, 292, 278, 277, 313, 315, 295, 198, 358, 180, 359, 237, 333, 20, 257, 164, 124, 289, 90, 340, 229, 39, 223, 64, 126, 203, 114, 29, 268, 172, 15, 302, 123, 66, 226, 194, 49, 44, 79, 271, 165, 189, 89, 372, 234, 134, 235, 205, 385, 326, 251, 296, 21, 118, 167, 349, 193, 187, 215, 360, 157, 366, 339, 317, 357, 224, 262, 69, 84, 204, 324, 104, 111, 338, 146, 161, 2, 337, 141, 184, 284, 140, 153, 367, 298, 68, 246, 5, 191, 384, 207, 267, 217, 67, 316, 206, 73, 395, 244, 24, 34, 72, 181, 135, 120, 41, 55, 56, 275, 19, 63, 36, 50, 304, 40, 352, 323, 8, 100, 125, 192, 71, 308, 112, 240, 142, 368, 269, 96, 259, 86, 133, 299, 76, 318, 182, 155, 99, 375, 101, 393, 202, 186, 201, 59, 54, 378, 345, 282, 290, 379, 273, 30, 132, 283, 109, 320, 58, 350, 239, 32, 280, 74]
inner randperm to generate idx:
  [74, 43, 76, 124, 369, 187, 103, 55, 266, 131, 54, 220, 353, 252, 18, 57, 177, 47, 206, 249, 125, 199, 112, 135, 382, 140, 215, 289, 281, 130, 293, 68, 240, 13, 169, 34, 304, 265, 32, 257, 24, 247, 9, 8, 148, 287, 342, 213, 309, 71, 90, 170, 386, 208, 394, 94, 153, 0, 113, 56, 21, 114, 343, 277, 188, 111, 306, 278, 378, 285, 336, 44, 67, 37, 30, 189, 272, 260, 222, 72, 310, 326, 51, 70, 28, 156, 339, 162, 389, 347, 139, 234, 264, 292, 233, 38, 348, 308, 147, 345, 323, 163, 330, 96, 19, 251, 16, 193, 104, 212, 352, 387, 254, 295, 366, 123, 224, 370, 25, 105, 225, 145, 41, 371, 106, 290, 355, 379, 300, 217, 65, 311, 356, 316, 381, 282, 322, 243, 171, 197, 182, 184, 338, 374, 73, 166, 185, 179, 196, 350, 23, 261, 10, 359, 75, 141, 100, 92, 36, 235, 29, 299, 99, 231, 269, 204, 108, 207, 85, 377, 246, 161, 321, 331, 250, 255, 178, 164, 286, 119, 132, 143, 136, 146, 373, 167, 59, 200, 211, 35, 380, 172, 128, 158, 221, 280, 367, 344, 42, 368, 2, 118, 53, 270, 358, 214, 313, 192, 198, 175, 302, 160, 115, 314, 77, 320, 137, 294, 50, 219, 150, 66, 157, 383, 82, 230, 256, 340, 227, 341, 84, 312, 121, 195, 142, 81, 129, 327, 354, 165, 107, 209, 391, 237, 52, 40, 375, 69, 20, 39, 346, 155, 241, 58, 248, 216, 151, 333, 88, 176, 134, 93, 173, 22, 80, 362, 258, 279, 61, 210, 365, 275, 1, 98, 325, 14, 79, 174, 45, 263, 203, 328, 276, 244, 31, 236, 218, 329, 109, 385, 26, 349, 337, 46, 283, 64, 305, 11, 101, 238, 181, 267, 144, 296, 122, 388, 274, 223, 372, 271, 202, 152, 154, 239, 168, 284, 149, 91, 262, 27, 3, 95, 190, 307, 7, 191, 361, 116, 228, 126, 62, 242, 89, 12, 60, 332, 49, 357, 259, 15, 120, 78, 229, 97, 6, 317, 273, 232, 390, 86, 301, 201, 351, 194, 335, 324, 180, 226, 253, 4, 395, 110, 319, 102, 360, 186, 268, 205, 117, 376, 315, 33, 17, 298, 133, 63, 127, 83, 334, 288, 183, 48, 291, 318, 87, 245, 138, 393, 297, 159, 363, 392, 364, 384, 5, 303]'''

def main():
    print('Loading dataset ...\n')
    sess = derainSession(opt)
    loader_train, _ = sess.get_dataloader(opt.dataset, False)
    # dataset_train = Dataset(data_path=opt.data_path)
    # loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batch_size, shuffle=True)
    # print("# of training samples: %d\n" % int(len(dataset_train)))
    set_random_seed(1)
    print(torch.randperm(396).tolist())
    # Build model
    model = PReNet(recurrent_iter=opt.recurrent_iter, use_GPU=opt.use_gpu)
    print_network(model)

    # loss function
    # criterion = nn.MSELoss(size_average=False)
    criterion = SSIM()

    # Move to GPU
    if opt.use_gpu:
        model = model.cuda()
        criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.2)  # learning rates

    # record training
    writer = SummaryWriter(opt.save_path)

    # load the lastest model
    # initial_epoch = findLastCheckpoint(save_dir=opt.save_path)
    # if initial_epoch > 0:
    #     print('resuming by loading epoch %d' % initial_epoch)
    #     model.load_state_dict(torch.load(os.path.join(opt.save_path, 'net_epoch%d.pth' % initial_epoch)))

    # start training
    step = 0
    for epoch in range(0, opt.epochs):
        scheduler.step(epoch)
        for param_group in optimizer.param_groups:
            print('learning rate %f' % param_group["lr"])

        ## epoch training start
        for i, batch in enumerate(loader_train, 0):
            input_train = batch['O']
            target_train = batch['B']
            axes[0].imshow(input_train[0].permute(1, 2, 0).cpu().numpy())
            axes[1].imshow(target_train[0].permute(1, 2, 0).cpu().numpy())
            plt.savefig(f"{i}_{batch['filename'][0]}.png")

            # (input_train, target_train)
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            input_train, target_train = Variable(input_train), Variable(target_train)

            if opt.use_gpu:
                input_train, target_train = input_train.cuda(), target_train.cuda()

            out_train, _ = model(input_train)
            pixel_metric = criterion(target_train, out_train)
            loss = -pixel_metric

            loss.backward()
            optimizer.step()

            # # training curve
            # model.eval()
            # out_train, _ = model(input_train)
            out_train = torch.clamp(out_train, 0., 1.)
            psnr_train = batch_PSNR(out_train, target_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f, pixel_metric: %.4f, PSNR: %.4f" %
                  (epoch+1, i+1, len(loader_train), loss.item(), pixel_metric.item(), psnr_train))

            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                # writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        ## epoch training end

        # # log the images
        # model.eval()
        # out_train, _ = model(input_train)
        # out_train = torch.clamp(out_train, 0., 1.)
        # im_target = utils.make_grid(target_train.data, nrow=8, normalize=True, scale_each=True)
        # im_input = utils.make_grid(input_train.data, nrow=8, normalize=True, scale_each=True)
        # im_derain = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', im_target, epoch+1)
        # writer.add_image('rainy image', im_input, epoch+1)
        # writer.add_image('deraining image', im_derain, epoch+1)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_latest.pth'))
        if epoch % opt.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(opt.save_path, 'net_epoch%d.pth' % (epoch + 1)))


if __name__ == "__main__":
    # if opt.preprocess:
    #     if opt.data_path.find('RainTrainH') != -1:
    #         print(opt.data_path.find('RainTrainH'))
    #         prepare_data_RainTrainH(data_path=opt.data_path, patch_size=100, stride=80)
    #     elif opt.data_path.find('RainTrainL') != -1:
    #
    #         prepare_data_RainTrainL(data_path=opt.data_path, patch_size=100, stride=80)
    #     elif opt.data_path.find('Rain12600') != -1:
    #         prepare_data_Rain12600(data_path=opt.data_path, patch_size=100, stride=100)
    #     else:
    #         print('unkown datasets: please define prepare data function in DerainDataset.py')

    main()
