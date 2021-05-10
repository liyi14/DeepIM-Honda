# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""

from lib.datasets.curiosity_base import Curiosity_Base
import os.path as osp
ROOT_DIR = osp.join(osp.dirname(__file__), '..', '..')
curiosity_path = osp.join(ROOT_DIR, 'data', 'curiosity')
def get_dataset(dataset_class, image_set, mode, **kwargs):
    param = {'image_set': image_set,
             'mode': mode}
    if dataset_class == 'curiosity':
        param['dataset_root_path'] = curiosity_path
        param.update(kwargs)
        dataset = Curiosity_Base(**param)
        return dataset