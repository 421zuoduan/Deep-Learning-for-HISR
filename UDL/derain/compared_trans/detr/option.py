import argparse
import platform
import os

task = "derain"

script_path = os.path.dirname(os.path.dirname(__file__))
script_path = script_path.split(task)[0]

model_path = f'{script_path}/results/100L/DETR/detr/model_2021-09-02-21-59/497.pth.tar'

parser = argparse.ArgumentParser(description='PyTorch Training')
# * Logger
parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{task}',
                    help='path to save model')
parser.add_argument('--log_dir', metavar='DIR', default='logs',
                    help='path to save log')
parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                    help='useless in this script.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='DETR')
parser.add_argument('--use-tb', default=False, type=bool)

parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--lr_backbone', default=1e-5, type=float)
parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('-b', '--batch-size', default=280, type=int,  # 8
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--epochs', default=1500, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    # results/100L/APUnet/derain_large_V2/model_2021-04-06-23-54/487.pth.tar
                    help='path to latest checkpoint (default: none)')
# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='None', type=str, choices=('sine', 'learned', 'None'),
                    help="Type of positional embedding to use on top of the image features")
# * Transformer

parser.add_argument('--clip_max_norm', default=0, type=float,
                    help='gradient clipping max norm')

parser.add_argument('--enc_layers', default=6, type=int,  # 6
                    help="Number of encoding layers in the transformer")
parser.add_argument('--dec_layers', default=6, type=int,  # 6
                    help="Number of decoding layers in the transformer")
parser.add_argument('--dim_feedforward', default=2048, type=int,  # 2048
                    help="Intermediate size of the feedforward layers in the transformer blocks")
parser.add_argument('--hidden_dim', default=256, type=int,  # 256,由于使用了组归一和4次上采样通道变化，hidden_dim / 2^4 = 8x(mod8 ==0)
                    help="Size of the embeddings (dimension of the transformer P^2*C)")
parser.add_argument('--dropout', default=0.1, type=float,
                    help="Dropout applied in the transformer")
parser.add_argument('--nheads', default=8, type=int,  # 8
                    help="Number of attention heads inside the transformer's multi-head attentions")
parser.add_argument('--num_queries', default=16, type=int,  # 100
                    help="Number of query slots")  # only used for deteted bbox nums->batch
parser.add_argument('--pre_norm', action='store_true')

# * Segmentation
parser.add_argument('--masks', action='store_true', default=True,
                    help="Train segmentation head if the flag is provided")
# Loss
parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                    help="Disables auxiliary decoding losses (loss at each layer)")
parser.add_argument('--eos_coef', default=0.1, type=float,
                    help="Relative classification weight of the no-object class")
# Model parameters
parser.add_argument('--frozen_weights', type=str, default=None,
                    help="Path to the pretrained model. If set, only the mask head will be trained")

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')
parser.add_argument('--patch_size', type=int, default=128,
                    help='image2patch, set to model and dataset')
parser.add_argument('--eval', default=False, type=bool,
                    help="performing evalution for patch2entire")
parser.add_argument('--dataset', default='Rain200L', type=str,
                    choices=[None, 'Rain200H', 'Rain100L', 'Rain200H', 'Rain100H',
                             'test12', 'real', 'DID', 'SPA', 'DDN'],
                    help="performing evalution for patch2entire")
parser.add_argument('--crop_batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--accumulated-step', default=1, type=int)

## DDP
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', default=0, type=int,
                    help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
parser.add_argument('--backend', default='nccl', type=str,  # gloo
                    help='distributed backend')
parser.add_argument('--dist-url', default='env://',
                    type=str,
                    help='url used to set up distributed training')

## AMP
parser.add_argument('--amp', default=None, type=bool,
                    help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')


args = parser.parse_args()
args.once_epoch = False
# log_string(args)#引发日志错误
args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
print(args.launcher)
assert args.accumulated_step > 0


if platform.system() == 'Linux':
    args.data_dir = '/home/office-409/Datasets/derain'
if platform.system() == "Windows":
    args.data_dir = 'D:/Datasets/derain'
args.experimental_desc = "Test"
