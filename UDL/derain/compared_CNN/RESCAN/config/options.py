import argparse
import platform
import os

script_path = os.path.dirname(os.path.dirname(__file__))
script_path = script_path.split('compared_CNN')[0]

model_path = f'{script_path}/results/100H/HPT_new/OneAttn/model_2021-08-30-14-47/.pth.tar'

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
# * Logger
parser.add_argument('--use-log', default=True
                    , type=bool)
parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results',
                    help='path to save model')
parser.add_argument('--log_dir', metavar='DIR', default='logs',
                    help='path to save log')
parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                    help='useless in this script.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='HPT_new')
parser.add_argument('--use-tb', default=False, type=bool)

# * Backbone
parser.add_argument('--backbone', default='resnet50', type=str,
                    help="Name of the convolutional backbone to use")
parser.add_argument('--dilation', action='store_true',
                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
parser.add_argument('--position_embedding', default='learned', type=str, choices=('sine', 'learned', 'None'),
                    help="Type of positional embedding to use on top of the image features")

parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                    help='start epoch')

## Train
parser.add_argument('--patch_size', type=int, default=64,
                    help='image2patch, set to model and dataset')
parser.add_argument('--lr', default=5e-3, type=float)  # 1e-4 2e-4 8
# parser.add_argument('--lr_backbone', default=1e-5, type=float)
# parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('-b', '--batch-size', default=64, type=int,  # 8
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--epochs', default=400, type=int)
parser.add_argument('--workers', default=8, type=int)
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--accumulated-step', default=1, type=int)
parser.add_argument('--clip_max_norm', default=0, type=float,
                    help='gradient clipping max norm')

parser.add_argument('--lr_scheduler', default=True, type=bool)

## Data
parser.add_argument('--model', default='rescan',
                    help='model name')
parser.add_argument('--test_every', type=float, default=156.25,
                    # {r:1000, epoch:100}-RCDNet, 即RESCAN {r:156.25,epoch:400} or {r:625, epoch:100}
                    help='do test per every N batches')
parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                    help='train dataset name')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')

# Benchmark
parser.add_argument('--eval', default=False, type=bool, help="performing evalution for patch2entire")
parser.add_argument('--dataset', default='Rain200L', type=str,
                    choices=[None, 'Rain200H', 'Rain100L', 'Rain200H',
                             'Rain100H', 'PReNetData', 'DID', 'SPA', 'DDN',
                             'test12', 'real', ],
                    help="set dataset name for training"
                         "real/test12 is eval-only")
parser.add_argument('--crop_batch_size', type=int, default=128,
                    help='input batch size for training')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')

## DDP
parser.add_argument(
    '--launcher',
    choices=['none', 'pytorch', 'slurm', 'mpi'],
    default='none',
    help='job launcher')
parser.add_argument('--local_rank', default=0, type=int,
                    help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
# parser.add_argument('--world-size', default=2, type=int,
#                     help='number of distributed processes, = gpus * nnodes')
parser.add_argument('--backend', default='nccl', type=str,  # gloo
                    help='distributed backend')
parser.add_argument('--dist-url', default='env://',
                    type=str,  # 'tcp://224.66.41.62:23456'
                    help='url used to set up distributed training')

## AMP
parser.add_argument('--amp', default=None, type=bool,
                    help="False is apex, besides True is  torch1.6+, which has supports amp ops to reduce gpu memory and speed up training")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')

args = parser.parse_args()

# log_string(args)#引发日志错误

# assert args.opt_level != 'O0' and args.amp != None, print("you must have apex or torch.cuda.amp")
args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
# args.launcher = 'none' if dist.is_available() else args.launcher
print(args.launcher)
assert args.accumulated_step > 0
# args.opt_level = 'O1'
args.scale = [1]
if platform.system() == 'Linux':
    args.data_dir = '/home/office-409/Datasets/derain'
if platform.system() == "Windows":
    args.data_dir = 'D:/Datasets/derain'
args.experimental_desc = "Test"
args.once_epoch = False