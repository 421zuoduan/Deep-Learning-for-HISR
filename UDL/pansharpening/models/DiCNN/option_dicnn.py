import argparse
from UDL.Basis.option import panshaprening_cfg, Config
import os
import warnings

cfg = Config(panshaprening_cfg())

script_path = os.path.dirname(os.path.dirname(__file__))
root_dir = script_path.split(cfg.task)[0]

model_path = f'{root_dir}/results/{cfg.task}/wv3/DiCNN/Test/model_2021-11-27-16-15/5000.pth.tar'

parser = argparse.ArgumentParser(description='PyTorch Pansharpening Training')
# * Logger
parser.add_argument('--out_dir', metavar='DIR', default=f'{root_dir}/results/{cfg.task}',
                    help='path to save model')
# * Training
parser.add_argument('--lr', default=2e-4, type=float)  # 1e-4 2e-4 8
parser.add_argument('--lr_scheduler', default=True, type=bool)
parser.add_argument('--samples_per_gpu', default=64, type=int,  # 8
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--print-freq', '-p', default=200, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--epochs', default=5000, type=int)
parser.add_argument('--workers_per_gpu', default=0, type=int)
parser.add_argument('--resume',
                    default=model_path,
                    type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
# * Model and Dataset
parser.add_argument('--arch', '-a', metavar='ARCH', default='DiCNN', type=str,
                    choices=['PanNet', 'DiCNN', 'PNN', 'FusionNet'])
parser.add_argument('--dataset', default='wv3_singleMat', type=str,
                    choices=[None, 'wv2', 'wv3', 'wv4', 'qb', 'gf',
                             'wv2_hp', ...,
                             'fr', 'wv3_singleMat', 'wv3_multi_exm1258'],
                    help="performing evalution for patch2entire")
parser.add_argument('--eval', default=True, type=bool,
                    help="performing evalution for patch2entire")

args = parser.parse_args()
args.start_epoch = args.best_epoch = 1
args.experimental_desc = "Test"

cfg.merge_args2cfg(args)
print(cfg.pretty_text)


# * Importantly
warning = f"you are using {args.dataset}, note that FusionNet, DiCNN, PNN don't have high-pass filter"

warnings.warn(warning)
