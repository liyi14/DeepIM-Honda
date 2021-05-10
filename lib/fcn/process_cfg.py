import os
import logging
import time
import pprint

from lib.fcn.config import cfg, cfg_from_file, get_output_dir
from lib.utils.print_and_log import print_and_log

def parse_class_ids(class_names, dataset='ycb'):
    if dataset == 'ycb':
        classes_all = (
            '__background__', '002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle',  # 4
            '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana',  # 9
            '019_pitcher_base', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block',  # 15
            '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick')
    elif dataset == 'goo':
        classes_all = ()
    elif dataset == 'cur':
        classes_all = ()
    elif dataset == 'rea':
        classes_all = ()
    else:
        raise NotImplementedError
    postfix = ''
    if class_names[0].lower() == 'all':
        class_ids = list(range(len(classes_all)))
        postfix = '_all'
    elif class_names[0].lower() == 'none':
        class_ids = []
        postfix = '_none'
    else:
        class_ids = [0]
        for class_name in class_names:
            if class_name in classes_all:
                class_ids.append(classes_all.index(class_name))
            else:
                raise Exception('class {} not found'.format(class_name))
            postfix += '_{}'.format(class_name[:3])
    return class_ids, postfix

def build_logger(args):
    # deal with logger
    if args.no_log:
        cfg.NO_LOG = True
    if args.no_cache:
        cfg.NO_CACHE = True
    output_dir = cfg.TRAIN.USING.OUTPUT_DIR if cfg.MODE=='TRAIN' else cfg.TEST.USING.OUTPUT_DIR
    logger = create_logger(cfg, output_dir, False)
    cfg.logger = logger

    return output_dir

def create_logger(cfg, final_output_path, pre='', temp_flie=False):
    cfg_name = cfg.CFG_NAME
    if temp_flie:
        log_file = 'temp_{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))
    else:
        log_file = '{}_{}.log'.format(cfg_name, time.strftime('%Y-%m-%d-%H-%M'))

    if pre:
        log_file = pre + '_' + log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=os.path.join(final_output_path, log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    return logger

def process_cfg_and_build_logger(args, mode='TEST'):
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
        cfg.MODE = mode
        cfg.CFG_NAME = os.path.basename(args.cfg_file).split('.')[0]

    dataset_name = args.trainset_name.split('@')[1][:3]
    class_names_in_args = args.class_name_ref.split(',')
    cfg.TRAIN.REF.CLASSES, postfix_ref = parse_class_ids(class_names_in_args, dataset_name)

    imageset_names_in_args = args.trainset_name.split('+')
    cfg.DATA.IMAGE_SETS = imageset_names_in_args

    if mode == 'TRAIN':
        dataset_name = args.valset_name.split('@')[1][:3]
        class_names_in_args = args.class_name_val.split(',')
        cfg.VAL.CLASSES, _ = parse_class_ids(class_names_in_args, dataset_name)
    elif mode == 'TEST':
        dataset_name = args.testset_name.split('@')[1][:3]
        class_names_in_args = args.class_name_test.split(',')
        cfg.TEST.CLASSES, _ = parse_class_ids(class_names_in_args, dataset_name)
    else:
        raise ValueError


    # build logger
    cfg.TRAIN.REF.OUTPUT_DIR = get_output_dir(cfg.CFG_NAME + postfix_ref, args.trainset_name, 'ref')
    if cfg.MODE == 'TEST':
        if args.phase == 'REF':
            cfg.TEST.REF.OUTPUT_DIR = get_output_dir(cfg.CFG_NAME+postfix_ref, args.testset_name, 'ref')

    cfg.PHASE = args.phase
    if cfg.PHASE == 'REF':
        cfg.TRAIN.USING = cfg.TRAIN.REF
        cfg.TEST.USING = cfg.TEST.REF
    else:
        raise KeyError

    output_dir = build_logger(args)

    print_and_log('Output will be saved to `{:s}`'.format(output_dir))

    if args.vis:
        cfg[mode].VISUALIZE = True

    # number of image per batch in dataloader
    if cfg.PHASE == 'REF':
        cfg.DATA.IMS_PER_BATCH = cfg.TRAIN.REF.IMS_PER_BATCH \
            if cfg.MODE == 'TRAIN' else cfg.TEST.REF.IMS_PER_BATCH

    cfg.TRAIN.SYNTHESIZE = (cfg.TRAIN.USING.SYN_RATIO>0 or cfg.TRAIN.USING.ALL_SYN)

    # device
    cfg.GPU_ID = [int(i) for i in args.gpu_id.split(',')]
    # all available GPUs
    if cfg.GPU_ID[0] == -1:
        import torch
        cfg.GPU_ID = list(range(torch.cuda.device_count()))

    cfg.ABS_GPU_ID = [int(i) for i in args.abs_gpu_id.split(',')]
    if args.abs_gpu_id[0] == -1:
        cfg.ABS_GPU_ID = cfg.GPU_ID

    print_and_log('GPU device {}'.format(args.gpu_id))

