# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""FCN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

import os
import os.path as osp
import numpy as np
import math
# `pip install easydict` if you don't have it
from easydict import EasyDict as edict

__C = edict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.MODE = 'TRAIN'
__C.ITERS = 0
__C.NO_LOG = False
__C.NO_CACHE = False
__C.GPUNUM = 1

__C.TRAIN = edict()
__C.TRAIN.FREQUENCY = 20

__C.TRAIN.USING = edict()

# learning rate
__C.TRAIN.DET = edict()
# learning rate
__C.TRAIN.REF = edict()
__C.TRAIN.REF.OUTPUT_DIR = ""
__C.TRAIN.REF.SOLVER = 'SGD' # 'SGD' 'ADAM'
__C.TRAIN.REF.LR_SCHEDULER = 'WARMUP_MULTISTEP' # "WARMUP_MULTISTEP", "COSINE"
__C.TRAIN.REF.LEARNING_RATE = 0.0001
__C.TRAIN.REF.MILESTONES_IMG = (-1, -1)
__C.TRAIN.REF.MILESTONES_EPOCH = (-1, -1)
__C.TRAIN.REF.MOMENTUM = 0.9
__C.TRAIN.REF.BETA = 0.999
__C.TRAIN.REF.GAMMA = 0.1
__C.TRAIN.REF.IMGS_EACH_EPOCH = -1
__C.TRAIN.REF.NUM_EPOCH = 1
__C.TRAIN.REF.BIAS_LR_FACTOR = 1.0
__C.TRAIN.REF.WARMUP_ITERS_IMG = -1
__C.TRAIN.REF.IMS_PER_BATCH = 16
__C.TRAIN.REF.WEIGHT_DECAY = 0.0001

__C.TRAIN.REF.FG_RATIO = 1.
__C.TRAIN.REF.ITERNUM = 4
__C.TRAIN.REF.HEATUP = 1
__C.TRAIN.REF.HEATUP_IMG = -1
__C.TRAIN.REF.HEATUP_STRATEGY = 'NONE'

__C.TRAIN.REF.ALL_SYN = False
__C.TRAIN.REF.SYN_RATIO = 1.0

## Iterations between snapshots
__C.TRAIN.REF.SNAPSHOT_ITERS_IMG = -1
__C.TRAIN.REF.SNAPSHOT_PREFIX = 'caffenet_fast_rcnn'
__C.TRAIN.REF.SNAPSHOT_INFIX = ''

__C.TRAIN.REF.CLASSES = [0,]

# generate initial pose
__C.TRAIN.REF.INIT_POSE_TYPE = 'close' # 'close', 'random', 'fixed' or 'det'
__C.TRAIN.REF.INIT_STD_ROTATION = 15 # for all except 'det'
__C.TRAIN.REF.INIT_SYN_STD_TRANSLATION = [0.01, 0.01, 0.05]# for all except 'det'
__C.TRAIN.REF.INIT_FIXED_POSE = [0, 0, 0] # for 'fixed' only

__C.TRAIN.REF.CONSIST_THRESHOLD = 0.25
__C.TRAIN.REF.MASK_DILATE = False # When feed DeepIM with mask
__C.TRAIN.REF.ZOOM_INCLUDE_GT = True

__C.TRAIN.REF.PRED_MASK = False
__C.TRAIN.REF.PRED_FLOW = True

__C.TRAIN.REF.EARLY_STOP = False

# for building the datasets/dataloader
__C.TRAIN.CURTAIL_RATIO = 10

__C.TRAIN.VISUALIZE = False

# Scales to compute real features
__C.TRAIN.SCALES_BASE = (1.0,)

__C.TRAIN.DISPLAY = 20

__C.TRAIN.EARLY_STOP = False

# determined by the SYN_RATIO and ALL_SYN
__C.TRAIN.SYNTHESIZE = False

# settings for synthetic data
__C.SYN = edict()
__C.SYN.HEIGHT = 480
__C.SYN.WIDTH = 640
__C.SYN.ONLINE = False
__C.SYN.POSE_UNIFORM_RATIO = 0.0
__C.SYN.MIN_FG_OBJECT = 1
__C.SYN.MAX_FG_OBJECT = 5
__C.SYN.NUM_OTHER_OBJECT = 1
__C.SYN.OCCLUDER_FACTOR = [0.5, 0.7]
__C.SYN.MAX_REND_TRY = 1
__C.SYN.BACKGROUND = 'pascal' # 'pascal' or 'ycb-video' or 'ikea'

__C.SYN.TNEAR = 0.5
__C.SYN.TFAR = 2.0
__C.SYN.BOUND = 0.13
__C.SYN.MARGIN = 50 # the distance between syn object center to border
__C.SYN.STD_ROTATION = 15


# Loss
__C.LOSS = edict()
__C.LOSS.USE_SYM_INFO = 'all' #'all', 'bowl' 'none'

__C.LOSS.REF = edict()
__C.LOSS.REF.LW_PML = 1.0
__C.LOSS.REF.LW_ROT = 0.0
__C.LOSS.REF.LW_TRANS = 0.0
__C.LOSS.REF.PML_NORMALIZE_METHOD = 'extents' # extents, diameter, constant
__C.LOSS.REF.PML_NORMALIZE_FACTOR = 10.0
__C.LOSS.REF.LW_FLOW = 0.0
__C.LOSS.REF.LW_ANGLE = 0.0
__C.LOSS.REF.LW_MASK = 0.0
__C.LOSS.REF.MASK_LOSS = 'sigmoid' # 'sigmoid' or 'softmax'
__C.LOSS.REF.LW_REG = 0.0
__C.LOSS.REF.LW_DCN_LOSS = 0.0
__C.LOSS.REF.LW_EMBEDDING = 0.0
__C.LOSS.REF.EMBEDDING = edict()
__C.LOSS.REF.EMBEDDING.LOSS_TYPE = 'contrastive' # triplet, contrastive
__C.LOSS.REF.EMBEDDING.DISTANCE_TYPE = 'cosine' # cosine, l2
__C.LOSS.REF.EMBEDDING.NUM_NEGATIVE_EACH_ANCHOR = 3
__C.LOSS.REF.EMBEDDING.NEGATIVE_STRATEGY = 'nonzero' # random, nonzero, hard-negative
__C.LOSS.REF.EMBEDDING.M_NEGATIVE = 0.5
__C.LOSS.REF.EMBEDDING.LW_NEGATIVE = 1.

__C.LOSS.DET = edict()

__C.DATA = edict()
# ims per batch for dataloader, same to TRAIN.DET.IPB if DET
# same to TEST.DET.IPB if REF and 'det'
__C.DATA.MIN_VIS_AREA = -1
__C.DATA.MIN_VIS_RATIO = 0.25
__C.DATA.ADD_NOISE = False
__C.DATA.CHROMATIC = True
__C.DATA.CHROM_NOISE = [0.02, 0.2, 0.05] # H, L, S
__C.DATA.MOTION_BLUR_RATIO = 0.2
__C.DATA.DEPTH_NOISE_STD = 0.0
__C.DATA.ADD_HOLES = 0.
__C.DATA.INPUT_HEIGHT = 480
__C.DATA.INPUT_WIDTH = 640
__C.DATA.DISCRET_COLOR_STEP = 2
__C.DATA.HIGH_RES_REND = False
__C.DATA.NUM_WORKERS = 0
__C.DATA.IMAGE_SETS = []
__C.DATA.DATASET = 'ycbv_bop_dcm'
__C.DATA.REPLACE_REAL_BG_RATIO = 0.0
__C.DATA.ANCHOR_IMG_HEIGHT = 224
__C.DATA.ANCHOR_IMG_WIDTH = 224
__C.DATA.ANCHOR_STEPSIZE = 6

__C.DATA.LARGE_SCALE = edict()
__C.DATA.LARGE_SCALE.ENABLE = False
__C.DATA.LARGE_SCALE.NUM_MODELS_EACH_ROUND = 0
__C.DATA.LARGE_SCALE.MODEL_SAMPLE_STRATEGY = 'sequential' # sequential, random or manual
__C.DATA.LARGE_SCALE.MANUAL = []

# Network
__C.NETWORK = edict()

__C.NETWORK.DET = edict()

__C.NETWORK.REF = edict()
__C.NETWORK.REF.NAME = 'flownets'
# __C.NETWORK.REF.PRETRAINED = 'data/checkpoints/flownets_EPE1.951.pth.tar'
__C.NETWORK.REF.PRETRAINED = ''
__C.NETWORK.REF.RELU_TYPE = 'relu'
__C.NETWORK.REF.BASENET_TYPE = 'resnet50'
__C.NETWORK.REF.FIX_RESNET_BN = False
__C.NETWORK.REF.USE_GN = False
__C.NETWORK.REF.GN_CHANNEL_PER_GROUP = 32
__C.NETWORK.REF.EMBEDDING_BIAS = False

__C.NETWORK.REF.SHARE_ENCODER = True
__C.NETWORK.REF.FCN_STRIDE = 16
__C.NETWORK.REF.OUT_FCN = 16
__C.NETWORK.REF.DESCRIPTOR_SIZE = 16
__C.NETWORK.REF.ENCODER_DETACH = False

__C.NETWORK.REF.CONCAT_CORR = False
__C.NETWORK.REF.XCONVS_NUM = 1
__C.NETWORK.REF.XCONVS_CHANNELS = 512
__C.NETWORK.REF.FC_SIZE = 1024
__C.NETWORK.REF.IDENTITY_INIT = False
__C.NETWORK.REF.LATE_FUSION_METHOD = 'concat'
## inputs
__C.NETWORK.REF.INPUT_MASK = False
__C.NETWORK.REF.INPUT_OBS_DEPTH = False
__C.NETWORK.REF.INPUT_OBS_DEPTH_MASK = False
__C.NETWORK.REF.INPUT_REND_DEPTH = False
__C.NETWORK.REF.INPUT_REND_DEPTH_MASK = False
__C.NETWORK.REF.INPUT_RGB_DIFF = False
__C.NETWORK.REF.INPUT_DEPTH_DIFF = False
__C.NETWORK.REF.NORM_INPUT_DEPTH = 'none' # 'z' or 'center' or 'halfhalf'
__C.NETWORK.REF.DEPTH_TYPE = 'z' # 'z' or 'pcloud'
__C.NETWORK.REF.ZOOM_AT_RENDERER = False

## outputs
__C.NETWORK.REF.CLASS_AWARE = True
__C.NETWORK.REF.REPRESENTATION = 'UNTANGLED' # 'UNTANGLED' or 'IMAGE'
__C.NETWORK.REF.ROT_TYPE = 'quaternion' # 'quaternion' or 'axisangle'
__C.NETWORK.REF.ORIENTATION_TYPE = 'ego' # 'ego' or 'allo'
__C.NETWORK.REF.OBJECT_IN_IMG = 0.
__C.NETWORK.REF.MULTI_REGRESSOR = False
__C.NETWORK.REF.REG_THRESHOLD = 30 # when using regressor alpha/beta
__C.NETWORK.REF.TRANS_SCALE_FACTOR = 1.


# Pixel mean values (BGR order) as a (1, 1, 3) array
# These are the values originally used for training VGG16
# __C.NETWORK.REF.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])
__C.NETWORK.REF.PIXEL_MEANS = np.array([0.5573105812072754, 0.37420374155044556, 0.37020164728164673])
__C.NETWORK.REF.PIXEL_STD = np.array([0.24336038529872894, 0.2987397611141205, 0.31875079870224])

__C.INPUT = edict()
__C.INPUT.MASK_IM_SRC = False
__C.INPUT.ENLARGE_RATIO = 1.4
#
# Testing options
#
__C.VAL = edict()
__C.VAL.CLASSES = []
__C.VAL.INIT_POSE_TYPE = 'cdpn' # 'close', 'random', 'fixed', 'posecnn', or 'det'
__C.VAL.CURTAIL_RATIO = 2
__C.VAL.FREQUENCY = -1
__C.VAL.FREQUENCY_EPOCH = -1
__C.VAL.ALL_SYN = False
__C.VAL.SYN_LENGTH = 200

__C.TEST = edict()
__C.TEST.SINGLE_FRAME = False
__C.TEST.VISUALIZE = False
__C.TEST.VIS_TYPE = 'edge' # or 'shadow'
__C.TEST.SYNTHETIC = False
__C.TEST.CURTAIL_RATIO = 1
__C.TEST.ALL_SYN = False

__C.TEST.DET = edict()

__C.TEST.REF = edict()
__C.TEST.REF.IMS_PER_BATCH = 1
__C.TEST.REF.ITERNUM = 4
__C.TEST.REF.INIT_POSE_TYPE = 'cdpn' # 'close', 'random', 'fixed', 'posecnn', or 'det'
__C.TEST.REF.INIT_FIXED_POSE = [0, 0, 0]
__C.TEST.REF.INIT_STD_ROTATION = 15 # for all except 'det'
__C.TEST.REF.INIT_SYN_STD_TRANSLATION = [0.01, 0.01, 0.05]# for all except 'det'
__C.TEST.REF.SYN_LENGTH = 200
__C.TEST.REF.TRACKING = False

__C.TEST.CLASSES = ()

__C.TEST.TEST_EPOCH = -1

# Scales to compute real features
__C.TEST.SCALES_BASE = (1.0,)

# for live demo
__C.TEST.TRACKING = False
__C.TEST.SAVE_VIDEO = False
__C.TEST.SAVE_IMAGE = False
__C.TEST.VIDEO_NAME = 'temp'

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Place outputs under an experiments directory
__C.DET_EXP_DIR = 'default'
__C.REF_EXP_DIR = 'default'

__C.PHASE = 'DET' # DET or REF

# Default GPU device id
__C.GPU_ID = [0]
__C.ABS_GPU_ID = [-1]

def get_exp_dir(cfg_name):
    path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', __C.EXP_DIR, cfg_name))
    return path

def get_output_dir(cfg_name, image_set, net_type):
    """Return the directory where experimental artifacts are placed.

    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    exp_dir = __C.REF_EXP_DIR if net_type == 'ref' else __C.DET_EXP_DIR
    config_output_path = osp.abspath(osp.join(__C.ROOT_DIR, 'output', exp_dir, cfg_name))
    image_sets = [iset for iset in image_set.split('+')]
    final_output_path = os.path.join(config_output_path, '_'.join(image_sets))
    if not os.path.exists(final_output_path):
        os.makedirs(final_output_path)
    return final_output_path

def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    if type(a) is not edict:
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if not k in b:
            raise KeyError('{} is not a valid config key'.format(k))

        # the types must match, too
        if type(b[k]) is not type(v):
            raise ValueError(('Type mismatch ({} vs. {}) '
                              'for config key: {}').format(type(b[k]),
                                                           type(v), k))

        # recursively merge dicts
        if type(v) is edict:
            try:
                _merge_a_into_b(a[k], b[k])
            except:
                print('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v

def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = edict(yaml.load(f))

    _merge_a_into_b(yaml_cfg, __C)


def write_selected_class_file(filename, index):
    # read file
    with open(filename) as f:
        lines = [x for x in f.readlines()]
    lines_selected = [lines[i] for i in index]

    # write new file
    filename_new = filename + '.selected'
    f = open(filename_new, 'w')
    for i in range(len(lines_selected)):
        f.write(lines_selected[i])
    f.close()
    return filename_new
