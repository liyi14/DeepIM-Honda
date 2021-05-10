# --------------------------------------------------------
# Deep Iterative Matching Network
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Yi Li
# --------------------------------------------------------
from __future__ import print_function, division
import os
import logging
import time

def build_logger(cfg, args):
    # deal with logger
    if args.no_log:
        cfg.NO_LOG = True
    if args.no_cache:
        cfg.NO_CACHE = True
    output_dir = cfg.TRAIN.USING.OUTPUT_DIR
    logger = create_logger(cfg, output_dir, False)
    cfg.logger = logger

    if cfg.NO_LOG:
        cfg.tf_logger = None
    else:
        from tensorboardX import SummaryWriter
        log_dir = os.path.join(
            output_dir, 'tf_log')
        if os.path.exists(log_dir):
            import shutil
            shutil.rmtree(log_dir)
        cfg.tf_logger = SummaryWriter(log_dir)
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

# def get_output_path(cfg, image_set):
#     # set up logger
#     root_output_path = os.path.join(cfg.ROOT_DIR, 'output', cfg.EXP_DIR)
#     if not os.path.exists(root_output_path):
#         os.makedirs(root_output_path)
#     assert os.path.exists(root_output_path), '{} does not exist'.format(root_output_path)
#
#     cfg_name = cfg.CFG_NAME
#     config_output_path = os.path.join(root_output_path, '{}'.format(cfg_name))
#     if not os.path.exists(config_output_path):
#         os.makedirs(config_output_path)
#
#     image_sets = [iset for iset in image_set.split('+')]
#     final_output_path = os.path.join(config_output_path, '{}'.format('_'.join(image_sets)))
#     if not os.path.exists(final_output_path):
#         os.makedirs(final_output_path)
#     return final_output_path

