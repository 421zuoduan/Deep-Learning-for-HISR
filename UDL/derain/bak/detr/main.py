# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--frozen_weights', type=str, default=None,
                        help="Path to the pretrained model. If set, only the mask head will be trained")
    # * Backbone
    parser.add_argument('--backbone', default='resnet50', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")

    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,#6
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,#6
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=512, type=int,#2048 512
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,#256 128
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,#8
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=30, type=int,#100 30
                        help="Number of query slots")
    parser.add_argument('--pre_norm', action='store_true')#输入先进行LayerNorm再进行project，仅此而已，默认为True

    # * Segmentation
    parser.add_argument('--masks', action='store_true', default=True,
                        help="Train segmentation head if the flag is provided")

    # Loss
    parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
                        help="Disables auxiliary decoding losses (loss at each layer)")
    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=5, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")
    # * Loss coefficients
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=5, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.1, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--coco_path', type=str, default='../dataset/coco2017')#D:/Datasets
    parser.add_argument('--coco_panoptic_path', type=str)
    parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)

    # distributed training parameters
    parser.add_argument('--local_rank', default=0, type=int,
                        help="host rank must be 0 and python -m torch.distributed.launch main.py need args.local_rank")
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser


# from torchstat import stat


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    if args.frozen_weights is not None:
        assert args.masks, "Frozen training is meant for segmentation only"
    print(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model, criterion, postprocessors = build_model(args)
    model.to(device)
    # stat(model, input_size=[(3, 768, 1151)])
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    param_dicts = [
        {"params": [p for n, p in model_without_ddp.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model_without_ddp.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.lr_drop)

    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.batch_size, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=utils.collate_fn, num_workers=args.num_workers)
    data_loader_val = DataLoader(dataset_val, args.batch_size, sampler=sampler_val,
                                 drop_last=False, collate_fn=utils.collate_fn, num_workers=args.num_workers)

    if args.dataset_file == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if args.frozen_weights is not None:
        checkpoint = torch.load(args.frozen_weights, map_location='cpu')
        model_without_ddp.detr.load_state_dict(checkpoint['model'])

    output_dir = Path(args.output_dir)
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(model, criterion, postprocessors,
                                              data_loader_val, base_ds, device, args.output_dir)
        if args.output_dir:
            utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth")
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, criterion, data_loader_train, optimizer, device, epoch,
            args.clip_max_norm)
        lr_scheduler.step()
        if args.output_dir:
            checkpoint_paths = [output_dir / 'checkpoint.pth']
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % args.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args,
                }, checkpoint_path)

        test_stats, coco_evaluator = evaluate(
            model, criterion, postprocessors, data_loader_val, base_ds, device, args.output_dir
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}

        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / 'eval').mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ['latest.pth']
                    if epoch % 50 == 0:
                        filenames.append(f'{epoch:03}.pth')
                    for name in filenames:
                        torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                   output_dir / "eval" / name)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

'''
多卡：
Epoch: [0]  [   0/7393]  eta: 19:16:10  lr: 0.000100  class_error: 99.34  loss: 72.5958 (72.5958)  loss_bbox: 5.2292 (5.2292)  loss_bbox_0: 5.1029 (5.1029)  loss_bbox_1: 5.1010 (5.1010)  loss_bbox_2: 5.0929 (5.0929)  loss_bbox_3: 5.1683 (5.1683)  loss_bbox_4: 5.2111 (5.2111)  loss_ce: 4.7281 (4.7281)  loss_ce_0: 4.5401 (4.5401)  loss_ce_1: 4.8207 (4.8207)  loss_ce_2: 4.6494 (4.6494)  loss_ce_3: 4.6423 (4.6423)  loss_ce_4: 4.6360 (4.6360)  loss_dice: 0.9409 (0.9409)  loss_giou: 2.0687 (2.0687)  loss_giou_0: 2.1002 (2.1002)  loss_giou_1: 2.1082 (2.1082)  loss_giou_2: 2.1060 (2.1060)  loss_giou_3: 2.0870 (2.0870)  loss_giou_4: 2.0764 (2.0764)  loss_mask: 0.1863 (0.1863)  cardinality_error_unscaled: 22.7500 (22.7500)  cardinality_error_0_unscaled: 22.5625 (22.5625)  cardinality_error_1_unscaled: 22.7500 (22.7500)  cardinality_error_2_unscaled: 22.7500 (22.7500)  cardinality_error_3_unscaled: 22.7500 (22.7500)  cardinality_error_4_unscaled: 22.7500 (22.7500)  class_error_unscaled: 99.3421 (99.3421)  loss_bbox_unscaled: 1.0458 (1.0458)  loss_bbox_0_unscaled: 1.0206 (1.0206)  loss_bbox_1_unscaled: 1.0202 (1.0202)  loss_bbox_2_unscaled: 1.0186 (1.0186)  loss_bbox_3_unscaled: 1.0337 (1.0337)  loss_bbox_4_unscaled: 1.0422 (1.0422)  loss_ce_unscaled: 4.7281 (4.7281)  loss_ce_0_unscaled: 4.5401 (4.5401)  loss_ce_1_unscaled: 4.8207 (4.8207)  loss_ce_2_unscaled: 4.6494 (4.6494)  loss_ce_3_unscaled: 4.6423 (4.6423)  loss_ce_4_unscaled: 4.6360 (4.6360)  loss_dice_unscaled: 0.9409 (0.9409)  loss_giou_unscaled: 1.0343 (1.0343)  loss_giou_0_unscaled: 1.0501 (1.0501)  loss_giou_1_unscaled: 1.0541 (1.0541)  loss_giou_2_unscaled: 1.0530 (1.0530)  loss_giou_3_unscaled: 1.0435 (1.0435)  loss_giou_4_unscaled: 1.0382 (1.0382)  loss_mask_unscaled: 0.1863 (0.1863)  time: 9.3833  data: 0.5977  max mem: 2850
Epoch: [0]  [  10/7393]  eta: 4:33:51  lr: 0.000100  class_error: 91.82  loss: 65.4168 (66.5790)  loss_bbox: 5.1612 (5.1667)  loss_bbox_0: 5.0160 (5.0394)  loss_bbox_1: 4.9513 (4.9877)  loss_bbox_2: 5.0726 (5.0806)  loss_bbox_3: 5.0746 (5.1051)  loss_bbox_4: 5.1069 (5.1300)  loss_ce: 3.6880 (3.7829)  loss_ce_0: 3.7046 (3.7351)  loss_ce_1: 3.6683 (3.7826)  loss_ce_2: 3.5149 (3.6273)  loss_ce_3: 3.4638 (3.6152)  loss_ce_4: 3.5070 (3.6587)  loss_dice: 0.9409 (0.9339)  loss_giou: 2.0943 (2.1123)  loss_giou_0: 2.1276 (2.1381)  loss_giou_1: 2.1336 (2.1474)  loss_giou_2: 2.1124 (2.1279)  loss_giou_3: 2.1033 (2.1226)  loss_giou_4: 2.0981 (2.1200)  loss_mask: 0.1649 (0.1654)  cardinality_error_unscaled: 6.4375 (9.7500)  cardinality_error_0_unscaled: 6.0625 (8.4375)  cardinality_error_1_unscaled: 7.0625 (9.4091)  cardinality_error_2_unscaled: 7.5625 (9.8182)  cardinality_error_3_unscaled: 6.0625 (8.1250)  cardinality_error_4_unscaled: 6.0625 (8.9432)  class_error_unscaled: 100.0000 (97.9280)  loss_bbox_unscaled: 1.0322 (1.0333)  loss_bbox_0_unscaled: 1.0032 (1.0079)  loss_bbox_1_unscaled: 0.9903 (0.9975)  loss_bbox_2_unscaled: 1.0145 (1.0161)  loss_bbox_3_unscaled: 1.0149 (1.0210)  loss_bbox_4_unscaled: 1.0214 (1.0260)  loss_ce_unscaled: 3.6880 (3.7829)  loss_ce_0_unscaled: 3.7046 (3.7351)  loss_ce_1_unscaled: 3.6683 (3.7826)  loss_ce_2_unscaled: 3.5149 (3.6273)  loss_ce_3_unscaled: 3.4638 (3.6152)  loss_ce_4_unscaled: 3.5070 (3.6587)  loss_dice_unscaled: 0.9409 (0.9339)  loss_giou_unscaled: 1.0471 (1.0562)  loss_giou_0_unscaled: 1.0638 (1.0691)  loss_giou_1_unscaled: 1.0668 (1.0737)  loss_giou_2_unscaled: 1.0562 (1.0640)  loss_giou_3_unscaled: 1.0517 (1.0613)  loss_giou_4_unscaled: 1.0491 (1.0600)  loss_mask_unscaled: 0.1649 (0.1654)  time: 2.2256  data: 0.0636  max mem: 4397

单卡：
Epoch: [0]  [    0/59143]  eta: 1 day, 6:46:23  lr: 0.000100  class_error: 100.00  loss: 74.6359 (74.6359)  loss_ce: 4.6751 (4.6751)  loss_bbox: 5.5621 (5.5621)  loss_giou: 2.0310 (2.0310)  loss_mask: 0.1904 (0.1904)  loss_dice: 0.9732 (0.9732)  loss_ce_0: 4.5167 (4.5167)  loss_bbox_0: 5.3059 (5.3059)  loss_giou_0: 2.0320 (2.0320)  loss_ce_1: 4.9539 (4.9539)  loss_bbox_1: 5.4531 (5.4531)  loss_giou_1: 2.0861 (2.0861)  loss_ce_2: 4.7154 (4.7154)  loss_bbox_2: 5.5278 (5.5278)  loss_giou_2: 2.1188 (2.1188)  loss_ce_3: 4.7491 (4.7491)  loss_bbox_3: 5.5238 (5.5238)  loss_giou_3: 2.1022 (2.1022)  loss_ce_4: 4.5454 (4.5454)  loss_bbox_4: 5.5490 (5.5490)  loss_giou_4: 2.0250 (2.0250)  loss_ce_unscaled: 4.6751 (4.6751)  class_error_unscaled: 100.0000 (100.0000)  loss_bbox_unscaled: 1.1124 (1.1124)  loss_giou_unscaled: 1.0155 (1.0155)  cardinality_error_unscaled: 26.0000 (26.0000)  loss_mask_unscaled: 0.1904 (0.1904)  loss_dice_unscaled: 0.9732 (0.9732)  loss_ce_0_unscaled: 4.5167 (4.5167)  loss_bbox_0_unscaled: 1.0612 (1.0612)  loss_giou_0_unscaled: 1.0160 (1.0160)  cardinality_error_0_unscaled: 26.0000 (26.0000)  loss_ce_1_unscaled: 4.9539 (4.9539)  loss_bbox_1_unscaled: 1.0906 (1.0906)  loss_giou_1_unscaled: 1.0431 (1.0431)  cardinality_error_1_unscaled: 26.0000 (26.0000)  loss_ce_2_unscaled: 4.7154 (4.7154)  loss_bbox_2_unscaled: 1.1056 (1.1056)  loss_giou_2_unscaled: 1.0594 (1.0594)  cardinality_error_2_unscaled: 26.0000 (26.0000)  loss_ce_3_unscaled: 4.7491 (4.7491)  loss_bbox_3_unscaled: 1.1048 (1.1048)  loss_giou_3_unscaled: 1.0511 (1.0511)  cardinality_error_3_unscaled: 26.0000 (26.0000)  loss_ce_4_unscaled: 4.5454 (4.5454)  loss_bbox_4_unscaled: 1.1098 (1.1098)  loss_giou_4_unscaled: 1.0125 (1.0125)  cardinality_error_4_unscaled: 26.0000 (26.0000)  time: 1.8731  data: 0.5411  max mem: 2711
Epoch: [0]  [   10/59143]  eta: 21:34:30  lr: 0.000100  class_error: 100.00  loss: 65.3820 (64.5699)  loss_ce: 3.8150 (3.8616)  loss_bbox: 5.0271 (4.9222)  loss_giou: 2.0310 (1.9427)  loss_mask: 0.1640 (0.1643)  loss_dice: 0.9150 (0.9037)  loss_ce_0: 3.7657 (3.6742)  loss_bbox_0: 4.9614 (4.8394)  loss_giou_0: 2.0320 (1.9577)  loss_ce_1: 3.6083 (3.8050)  loss_bbox_1: 4.8953 (4.8184)  loss_giou_1: 2.0842 (1.9724)  loss_ce_2: 3.5338 (3.6707)  loss_bbox_2: 5.0366 (4.9103)  loss_giou_2: 2.0564 (1.9578)  loss_ce_3: 3.4120 (3.6511)  loss_bbox_3: 5.0403 (4.9314)  loss_giou_3: 2.0243 (1.9460)  loss_ce_4: 3.7089 (3.7622)  loss_bbox_4: 5.0917 (4.9263)  loss_giou_4: 2.0250 (1.9524)  loss_ce_unscaled: 3.8150 (3.8616)  class_error_unscaled: 100.0000 (86.7796)  loss_bbox_unscaled: 1.0054 (0.9844)  loss_giou_unscaled: 1.0155 (0.9713)  cardinality_error_unscaled: 13.5000 (14.3182)  loss_mask_unscaled: 0.1640 (0.1643)  loss_dice_unscaled: 0.9150 (0.9037)  loss_ce_0_unscaled: 3.7657 (3.6742)  loss_bbox_0_unscaled: 0.9923 (0.9679)  loss_giou_0_unscaled: 1.0160 (0.9789)  cardinality_error_0_unscaled: 4.5000 (7.1364)  loss_ce_1_unscaled: 3.6083 (3.8050)  loss_bbox_1_unscaled: 0.9791 (0.9637)  loss_giou_1_unscaled: 1.0421 (0.9862)  cardinality_error_1_unscaled: 12.5000 (13.0909)  loss_ce_2_unscaled: 3.5338 (3.6707)  loss_bbox_2_unscaled: 1.0073 (0.9821)  loss_giou_2_unscaled: 1.0282 (0.9789)  cardinality_error_2_unscaled: 14.5000 (14.6364)  loss_ce_3_unscaled: 3.4120 (3.6511)  loss_bbox_3_unscaled: 1.0081 (0.9863)  loss_giou_3_unscaled: 1.0122 (0.9730)  cardinality_error_3_unscaled: 4.5000 (8.3182)  loss_ce_4_unscaled: 3.7089 (3.7622)  loss_bbox_4_unscaled: 1.0183 (0.9853)  loss_giou_4_unscaled: 1.0125 (0.9762)  cardinality_error_4_unscaled: 18.0000 (14.9091)  time: 1.3135  data: 0.0567  max mem: 3914
Epoch: [0]  [   20/59143]  eta: 21:40:57  lr: 0.000100  class_error: 100.00  loss: 62.0359 (63.0958)  loss_ce: 3.4938 (3.6344)  loss_bbox: 4.7008 (4.7978)  loss_giou: 2.0650 (1.9940)  loss_mask: 0.1513 (0.1564)  loss_dice: 0.9150 (0.9131)  loss_ce_0: 3.3972 (3.5557)  loss_bbox_0: 4.6230 (4.7027)  loss_giou_0: 2.0725 (2.0089)  loss_ce_1: 3.4022 (3.6351)  loss_bbox_1: 4.5681 (4.6580)  loss_giou_1: 2.0842 (2.0195)  loss_ce_2: 3.4046 (3.5680)  loss_bbox_2: 4.6114 (4.7473)  loss_giou_2: 2.0689 (2.0085)  loss_ce_3: 3.4120 (3.5516)  loss_bbox_3: 4.6602 (4.7883)  loss_giou_3: 2.0601 (1.9951)  loss_ce_4: 3.4052 (3.5774)  loss_bbox_4: 4.6692 (4.7814)  loss_giou_4: 2.0587 (2.0025)  loss_ce_unscaled: 3.4938 (3.6344)  class_error_unscaled: 100.0000 (93.0750)  loss_bbox_unscaled: 0.9402 (0.9596)  loss_giou_unscaled: 1.0325 (0.9970)  cardinality_error_unscaled: 6.5000 (10.6905)  loss_mask_unscaled: 0.1513 (0.1564)  loss_dice_unscaled: 0.9150 (0.9131)  loss_ce_0_unscaled: 3.3972 (3.5557)  loss_bbox_0_unscaled: 0.9246 (0.9405)  loss_giou_0_unscaled: 1.0362 (1.0044)  cardinality_error_0_unscaled: 4.5000 (6.9286)  loss_ce_1_unscaled: 3.4022 (3.6351)  loss_bbox_1_unscaled: 0.9136 (0.9316)  loss_giou_1_unscaled: 1.0421 (1.0098)  cardinality_error_1_unscaled: 6.5000 (10.0476)  loss_ce_2_unscaled: 3.4046 (3.5680)  loss_bbox_2_unscaled: 0.9223 (0.9495)  loss_giou_2_unscaled: 1.0344 (1.0043)  cardinality_error_2_unscaled: 6.0000 (10.6667)  loss_ce_3_unscaled: 3.4120 (3.5516)  loss_bbox_3_unscaled: 0.9320 (0.9577)  loss_giou_3_unscaled: 1.0300 (0.9975)  cardinality_error_3_unscaled: 4.5000 (7.5476)  loss_ce_4_unscaled: 3.4052 (3.5774)  loss_bbox_4_unscaled: 0.9338 (0.9563)  loss_giou_4_unscaled: 1.0294 (1.0012)  cardinality_error_4_unscaled: 7.0000 (11.0000)  time: 1.2926  data: 0.0088  max mem: 4921
'''