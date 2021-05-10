#!/usr/bin/env python

# --------------------------------------------------------
# FCN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Train a Fully Convolutional Network (FCN) on image segmentation database."""

import torch
torch.manual_seed(0)
import torch.nn.parallel
import torch.backends.cudnn as cudnn
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import torch.optim
import torch.utils.data
# torch.backends.cudnn.enabled=False

import argparse
import numpy as np
import sys
import os
import os.path as osp
import cv2

from lib.fcn.config import cfg
from lib.fcn.train_dcm import DeepCodeMatchingTraining
from lib.fcn.process_cfg import process_cfg_and_build_logger
from lib.datasets.factory import get_dataset
from ycb_render.ycb_renderer import YCBRenderer
from lib.utils.print_and_log import print_and_log
import lib.networks as networks
import pprint

def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id',
                        help='GPU device id to use [0]',
                        default='0', type=str)
    parser.add_argument('--abs_gpu_id', dest='abs_gpu_id',
                        help='for mox.hyak, the absolute gpuid if specified',
                        default='-1', type=str)
    parser.add_argument('--ref_checkpoint', dest='ref_checkpoint',
                        help='number of images (k) trained',
                        default=-1, type=int)
    parser.add_argument('--det_checkpoint', dest='det_checkpoint',
                        help='number of images (k) trained',
                        default=-1, type=int)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    parser.add_argument('--class_det', dest='class_name_det',
                        help='category name that det net trained on, including all or none',
                        default='none', type=str)
    parser.add_argument('--class_ref', dest='class_name_ref',
                        help='category name for ref net to train on, including all or none',
                        default='none', type=str)
    parser.add_argument('--class_test', dest='class_name_test',
                        help='category name for ref net to train on, including all or none',
                        default='none', type=str)

    parser.add_argument('--trainset', dest='trainset_name',
                        help='dataset that det net trained on',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--testset', dest='testset_name',
                        help='dataset to val on',
                        default='shapenet_scene_train', type=str)

    parser.add_argument('--phase', dest='phase',
                        help='be DET or REF',
                        default='DET', type=str)
    parser.add_argument('--tracking', help='turn on tracking mode', action='store_true')

    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--vis', help='turn on visualizing mode', action='store_true')
    parser.add_argument('--no_log', help='mute the output to log file', action='store_true')
    parser.add_argument('--no_cache', help='not use the presaved cache', action='store_true')
    parser.add_argument('--trainer', dest='trainer',
                        help='trainer type (DCM or FEWSHOT)',
                        default='DCM', type=str)

    if len(sys.argv) == 1:
        parser.print_and_log_help()
        sys.exit(1)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    import random
    random.seed(1234)
    np.random.seed(1234)
    # random.seed(234)
    # np.random.seed(234)
    args = parse_args()
    # print(args)
    # process cfg based on args and other rules
    process_cfg_and_build_logger(args, 'TEST')
    if args.tracking:
        cfg.TEST.REF.TRACKING = True

    output_dir_train = cfg.TRAIN.REF.OUTPUT_DIR if args.phase == 'REF' else cfg.TRAIN.DET.OUTPUT_DIR
    if args.phase == 'REF':
        if args.ref_checkpoint == -1:
            pretrained_model = os.path.join(output_dir_train, "REF_model_final.pth")
            if not os.path.exists(pretrained_model):
                pretrained_model = os.path.join(output_dir_train, "ref_model_final.pth")
        else:
            pretrained_model = os.path.join(output_dir_train, "checkpoint_{:08d}".format(args.ref_checkpoint))
        cfg.NETWORK.REF.PRETRAINED = pretrained_model
    elif args.phase == 'DET':
        if args.det_checkpoint == -1:
            pretrained_model = os.path.join(output_dir_train, "DET_model_final.pth")
            if not os.path.exists(pretrained_model):
                pretrained_model = os.path.join(output_dir_train, "det_model_final.pth")
        else:
            pretrained_model = os.path.join(output_dir_train, "checkpoint_{:08d}".format(args.det_checkpoint))
        cfg.NETWORK.DET.PRETRAINED = pretrained_model
    else:
        raise KeyError

    # print args and cfg
    print_and_log('Called with args:')
    print_and_log(pprint.pformat(args))
    print_and_log('Using config:')
    print_and_log(pprint.pformat(cfg))

    # build dataset
    testset = get_dataset(args.testset_name.split('@')[1], args.testset_name.split('@')[0], 'TEST')
    print_and_log('Use dataset `{:s}` for testing'.format(testset.name))

    # if torch.cuda.device_count() > 1:
    #     cfg.GPUNUM = torch.cuda.device_count()
    torch.cuda.set_device(cfg.GPU_ID[0])
    print("Let's use", len(cfg.GPU_ID), "GPUs!")

    if args.trainer == 'DCM':
        train = DeepCodeMatchingTraining(cfg, args, None, testset, mode='TEST', phase=args.phase)
    elif args.trainer == 'FEWSHOT':
        train = DeepCodeMatchingFewShotTraining(cfg, args, None, testset, mode='TEST', phase=args.phase)
    else:
        raise KeyError
    train.setup()

    with torch.no_grad():
        train.test()
    print(args.cfg_file)
    print(output_dir_train) 
