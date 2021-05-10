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
torch.cuda.manual_seed(0)
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
from lib.fcn.train_fewshot import DeepCodeMatchingFewShotTraining
from lib.fcn.process_cfg import process_cfg_and_build_logger
from lib.datasets.factory import get_dataset
from ycb_render.ycb_renderer import YCBRenderer
from lib.utils.print_and_log import print_and_log
from lib.utils.merge_datasets import merge_datasets
import lib.networks as networks
import time

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
    parser.add_argument('--startiter', dest='startiter',
                        help='the starting iter (resume)',
                        default=-1, type=int)
    parser.add_argument('--pretrained', dest='pretrained',
                        help='initialize with pretrained checkpoint',
                        default=None, type=str)
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default=None, type=str)

    parser.add_argument('--class_det', dest='class_name_det',
                        help='category name that det net trained on, including all or none',
                        default='none', type=str)
    parser.add_argument('--class_ref', dest='class_name_ref',
                        help='category name for ref net to train on, including all or none',
                        default='none', type=str)
    parser.add_argument('--class_val', dest='class_name_val',
                        help='category name for ref net to train on, including all or none',
                        default='none', type=str)

    parser.add_argument('--trainset', dest='trainset_name',
                        help='dataset that det net trained on, split with comma',
                        default='shapenet_scene_train', type=str)
    parser.add_argument('--valset', dest='valset_name',
                        help='dataset to val on',
                        default='shapenet_scene_train', type=str)

    parser.add_argument('--phase', dest='phase',
                        help='be DET or REF',
                        default='DET', type=str)

    parser.add_argument('--rand', dest='randomize',
                        help='randomize (do not use a fixed seed)',
                        action='store_true')
    parser.add_argument('--vis', help='turn on visualizing mode', action='store_true')
    parser.add_argument('--no_log', help='mute the output to log file', action='store_true')
    parser.add_argument('--no_cache', help='not use the presaved cache', action='store_true')
    parser.add_argument('--large_scale', help='turn on large scale mode', action='store_true')
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
    random.seed(123)
    np.random.seed(123)
    args = parse_args()
    # print(args)

    # process cfg based on args and other rules
    process_cfg_and_build_logger(args, 'TRAIN')

    print_and_log(args)

    start_time = time.time()
    phase = args.phase
    num_epoch = cfg.TRAIN.REF.NUM_EPOCH if phase=='REF' else cfg.TRAIN.DET.NUM_EPOCH

    for i_epoch in range(num_epoch):
        # build dataset
        if args.large_scale or i_epoch==0:
            trainset_names = args.trainset_name.split('+')
            datasets = [get_dataset(setname.split('@')[1], setname.split('@')[0], 'TRAIN')
                        for setname in trainset_names]
            for d in datasets:
                print_and_log("{}, {}".format(d._image_set, d._roidb.__len__()))
            if len(trainset_names) > 1:
                dataset = merge_datasets(datasets)
            else:
                dataset = datasets[0]
            print_and_log("merged, {}".format(dataset._roidb.__len__()))
            print_and_log("__len__, {}".format(dataset.__len__()))
            print_and_log('Use dataset `{:s}` for training'.format(dataset.name))

            if i_epoch == 0 or args.valset_name == 'train@google_scanned':
                # the class_names param will be ignored in ycbv dataset
                testset = get_dataset(args.valset_name.split('@')[1], args.valset_name.split('@')[0], 'VAL',
                                      class_names=dataset.get_class_names_all())
                print_and_log("__len__, {}".format(testset.__len__()))
                print_and_log('Use dataset `{:s}` for testing'.format(testset.name))
            else:
                pass

            torch.cuda.set_device(cfg.GPU_ID[0])
            print("Let's use", len(cfg.GPU_ID), "GPUs!")

            if i_epoch == 0:
                if args.trainer == 'DCM':
                    train = DeepCodeMatchingTraining(cfg, args, dataset, testset, mode='TRAIN', phase=phase)
                elif args.trainer == 'FEWSHOT':
                    train = DeepCodeMatchingFewShotTraining(cfg, args, dataset, testset, mode='TRAIN', phase=phase)
                else:
                    raise KeyError
                train.setup()
            else:
                train.round_step(dataset, testset)
                pass

        if phase=='REF':
            train.train_refine(i_epoch)
        else:
            if cfg.NETWORK.DET.NAME.startswith('dpn'):
                train.train_det_pointnet(i_epoch)
            else:
                train.train_det(i_epoch)

        if cfg.VAL.FREQUENCY_EPOCH > 0 and i_epoch % cfg.VAL.FREQUENCY_EPOCH == 0:
            train.set_val()
            if args.trainer=='DCM':
                if train._dataset._class_names_all != train._dataset_test._class_names_all:
                    train.clean_renderer('train')
                    train.build_renderer('test')
                    train.update_constants_for_val()
                with torch.no_grad():
                    if phase == 'REF':
                        train.test_refine()
                    elif cfg.NETWORK.DET.NAME.startswith('dpn'):
                        train.test_det_pointnet()
                    else:
                        train.test_det_quick()
            elif args.trainer == 'FEWSHOT':
                with torch.no_grad():
                    train.test_det()
            train.set_train()
        if args.large_scale:
            train.clean_buffer()
    train.finish_train()
    train.close()
    elapsed_time = time.time() - start_time
    print_and_log("Training Using ".format(time.strftime("%H:%M:%S", time.gmtime(elapsed_time))))
