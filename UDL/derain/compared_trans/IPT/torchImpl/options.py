import argparse
import platform
import os

task = "derain"

script_path = os.path.dirname(os.path.dirname(__file__))
script_path = script_path.split(task)[0]

# model_path = './IPT_derain.pt'
model_path = f'{script_path}/results/100L/HPT/ipt/model_2021-08-17-02-49/571.pth.tar'
# model_path = './IPT_pretrain.pt'
parser = argparse.ArgumentParser(description='PyTorch Derain Training')
# * Logger
parser.add_argument('--out_dir', metavar='DIR', default=f'{script_path}/results/{task}',
                    help='path to save model')
parser.add_argument('--log_dir', metavar='DIR', default='logs',
                    help='path to save log')
parser.add_argument('--tfb_dir', metavar='DIR', default=None,
                    help='useless in this script.')
parser.add_argument('--arch', '-a', metavar='ARCH', default='IPT')
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
parser.add_argument('--test_every', type=int, default=1000,
                    help='do test per every N batches')
parser.add_argument('--data_train', type=str, default='RainTrainL',  # DIV2K
                    help='train dataset name')

parser.add_argument('--lr', default=2e-5, type=float)  # 1e-4 2e-4 8
# parser.add_argument('--lr_backbone', default=1e-5, type=float)
# parser.add_argument('--weight_decay', default=1e-4, type=float)
parser.add_argument('-b', '--batch-size', default=8, type=int,  # 8
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=1, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--device', default='cuda',
                    help='device to use for training / testing')
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--workers', default=4, type=int)
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--accumulated-step', default=1, type=int)
parser.add_argument('--clip_max_norm', default=0, type=float,
                    help='gradient clipping max norm')

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
#####
parser.add_argument('--precision', type=str, default='single',
                    choices=('single', 'half'),
                    help='FP precision for test (single | half)')
parser.add_argument('--cpu', action='store_true',
                    help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1,
                    help='number of GPUs')
parser.add_argument('--save', type=str, default='/cache/results/ipt/',
                    help='file name to save')
parser.add_argument('--load', type=str, default='',
                    help='file name to load')
parser.add_argument('--save_models', action='store_true',
                    help='save all intermediate models')
parser.add_argument('--print_every', type=int, default=100,
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_results', action='store_true', default=True,
                    help='save output results')
parser.add_argument('--save_gt', action='store_true',
                    help='save low-resolution and high-resolution images together')
parser.add_argument('--model', default='ipt',
                    help='model name')
parser.add_argument('--n_feats', type=int, default=64,
                    help='number of feature maps')
parser.add_argument('--shift_mean', default=True,
                    help='subtract pixel mean from the input')
parser.add_argument('--rgb_range', type=int, default=255,
                    help='maximum value of RGB')
parser.add_argument('--n_colors', type=int, default=3,
                    help='number of color channels to use')
parser.add_argument('--no_augment', action='store_true',
                    help='do not use data augmentation')
parser.add_argument('--scale', type=str, default='1',  # 2+3+4+1+1+1
                    help='super resolution scale')
parser.add_argument('--patch_size', type=int, default=48,
                    help='output patch size')
parser.add_argument('--patch_dim', type=int, default=3)
parser.add_argument('--num_heads', type=int, default=12)
parser.add_argument('--num_layers', type=int, default=12)
parser.add_argument('--dropout_rate', type=float, default=0)
parser.add_argument('--no_norm', action='store_true')
parser.add_argument('--freeze_norm', action='store_true')
parser.add_argument('--post_norm', action='store_true')
parser.add_argument('--no_mlp', action='store_true')
parser.add_argument('--pos_every', action='store_true')
parser.add_argument('--no_pos', action='store_true')
parser.add_argument('--num_queries', type=int, default=6)
parser.add_argument('--crop_batch_size', type=int, default=64,
                    help='input batch size for training')
parser.add_argument('--dataset', default='Rain100L', type=str,
                    choices=[None, 'Rain200H', 'Rain100L', 'Rain200H', 'Rain100H',
                             'test12', 'real', 'DID', 'SPA', 'DDN'],
                    help="performing evalution for patch2entire")
parser.add_argument('--eval', default=False, type=bool, help="performing evalution for patch2entire")
parser.add_argument('--ext', type=str, default='sep',
                    help='dataset file extension')


args = parser.parse_args()
args.once_epoch = False
args.data_train = args.data_train.split('+')
args.scale = list(map(lambda x: int(x), args.scale.split('+')))
args.idx = 0
# log_string(args)#引发日志错误
args.amp_opt_level = 'O0' if args.amp == None else args.amp_opt_level
print(args.launcher)
if 'IPT_derain.pt' in model_path:
    args.num_queries = 1
if 'IPT_pretrain.pt' in model_path:
    args.num_queries = 6
if platform.system() == 'Linux':
    args.data_dir = '/home/office-409/Datasets/derain'
if platform.system() == "Windows":
    args.data_dir = 'D:/Datasets/derain'

assert args.accumulated_step > 0

args.experimental_desc = "Test"