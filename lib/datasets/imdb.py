# --------------------------------------------------------
# FCN
# Copyright (c) 2016
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import os
import os.path as osp
import numpy as np
import pickle
# from lib.fcn.config import cfg

class imdb(object):
    """Image database."""

    def __init__(self):
        self._name = ''
        self.num_classes = 0
        self._class_names_used = []

    @property
    def name(self):
        return self._name

    @property
    def num_classes(self):
        return len(self._class_names_used)

    @property
    def classes(self):
        return self._class_names_used

    @property
    def cache_path(self):
        cache_path = osp.abspath(osp.join(self.cfg.ROOT_DIR, 'data', 'cache'))
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path


    def _build_uniform_poses(self):

        self.eulers = []
        for roll in range(-180, 180, self.cfg.TRAIN.SYN_UNIFORM_POSE_INTERVAL):
            for pitch in range(-90, 90, self.cfg.TRAIN.SYN_UNIFORM_POSE_INTERVAL):
                for yaw in range(-180, 180, self.cfg.TRAIN.SYN_UNIFORM_POSE_INTERVAL):
                    self.eulers.append([roll, pitch, yaw])

        # sample indexes
        num_poses = len(self.eulers)
        num_classes = len(self._class_names_all)
        self.pose_indexes = np.zeros((num_classes, ), dtype=np.int32)
        self.pose_lists = []
        for i in range(num_classes):
            self.pose_lists.append(np.random.permutation(np.arange(num_poses)))


    def _build_background_images(self):

        color_cache_file = os.path.join(self.cache_path, self.cfg.SYN.BACKGROUND+'_backgrounds_color.pkl')
        depth_cache_file = os.path.join(self.cache_path, self.cfg.SYN.BACKGROUND+'_backgrounds_depth.pkl')

        if not self.cfg.NO_CACHE:
            if os.path.exists(color_cache_file) and os.path.exists(depth_cache_file):
                with open(color_cache_file, 'rb') as fid:
                    self._backgrounds_color = pickle.load(fid)
                print('{} backgrounds loaded from {}'.format(self._name, color_cache_file))
                with open(depth_cache_file, 'rb') as fid:
                    self._backgrounds_depth = pickle.load(fid)
                print('{} backgrounds loaded from {}'.format(self._name, depth_cache_file))
                if self.cfg.SYN.BACKGROUND == 'ikea' or self.cfg.SYN.BACKGROUND == 'pascal+ikea':
                    self._background_depth_metric = 0.001
                return

        backgrounds_color = []
        backgrounds_depth = []
        '''
        # SUN 2012
        root = os.path.join(self.cache_path, '../SUN2012/data/Images')
        subdirs = os.listdir(root)

        for i in xrange(len(subdirs)):
            subdir = subdirs[i]
            names = os.listdir(os.path.join(root, subdir))

            for j in xrange(len(names)):
                name = names[j]
                if os.path.isdir(os.path.join(root, subdir, name)):
                    files = os.listdir(os.path.join(root, subdir, name))
                    for k in range(len(files)):
                        if os.path.isdir(os.path.join(root, subdir, name, files[k])):
                            filenames = os.listdir(os.path.join(root, subdir, name, files[k]))
                            for l in range(len(filenames)):
                               filename = os.path.join(root, subdir, name, files[k], filenames[l])
                               backgrounds.append(filename)
                        else:
                            filename = os.path.join(root, subdir, name, files[k])
                            backgrounds.append(filename)
                else:
                    filename = os.path.join(root, subdir, name)
                    backgrounds.append(filename)

        # ObjectNet3D
        objectnet3d = os.path.join(self.cache_path, '../ObjectNet3D/data')
        files = os.listdir(objectnet3d)
        for i in range(len(files)):
            filename = os.path.join(objectnet3d, files[i])
            backgrounds.append(filename)
        '''

        if self.cfg.SYN.BACKGROUND == 'pascal':
            # PASCAL 2012
            pascal = os.path.join(self.cache_path, '../VOCdevkit/VOC2012/JPEGImages')
            files = os.listdir(pascal)
            for i in range(len(files)):
                filename = os.path.join(pascal, files[i])
                backgrounds_color.append(filename)

            for i in range(len(backgrounds_color)):
                if not os.path.isfile(backgrounds_color[i]):
                    print('file not exist {}'.format(backgrounds_color[i]))
        elif self.cfg.SYN.BACKGROUND == 'ikea':
            ikea_root = os.path.join(self.cache_path, '../ikea_data')
            folders = os.listdir(ikea_root)
            for folder in folders:
                ikea_dir = os.path.join(ikea_root, folder)
                if not os.path.isdir(ikea_dir):
                    continue
                files = os.listdir(ikea_dir)
                for file in files:
                    if file.endswith('_color.png'):
                        backgrounds_color.append(os.path.join(ikea_dir, file))
                    elif file.endswith('_depth.png'):
                        backgrounds_depth.append(os.path.join(ikea_dir, file))
                    else:
                        pass
            self._background_depth_metric = 0.001
        elif self.cfg.SYN.BACKGROUND == 'pascal+ikea':
            pascal = os.path.join(self.cache_path, '../VOCdevkit/VOC2012/JPEGImages')
            files = os.listdir(pascal)
            for i in range(len(files)):
                filename = os.path.join(pascal, files[i])
                backgrounds_color.append(filename)

            for i in range(len(backgrounds_color)):
                if not os.path.isfile(backgrounds_color[i]):
                    print('file not exist {}'.format(backgrounds_color[i]))

            ikea_root = os.path.join(self.cache_path, '../ikea_data')
            folders = os.listdir(ikea_root)
            for folder in folders:
                ikea_dir = os.path.join(ikea_root, folder)
                if not os.path.isdir(ikea_dir):
                    continue
                files = os.listdir(ikea_dir)
                for file in files:
                    if file.endswith('_depth.png'):
                        backgrounds_depth.append(os.path.join(ikea_dir, file))
                    else:
                        pass
            num_color_bg = len(backgrounds_color)
            num_depth_bg = len(backgrounds_depth)
            color_depth_match = np.random.randint(0, num_depth_bg, num_color_bg)
            backgrounds_depth = [backgrounds_depth[idx] for idx in color_depth_match]
            self._background_depth_metric = 0.001

        backgrounds_color.sort()
        backgrounds_depth.sort()
        self._backgrounds_color = backgrounds_color
        self._backgrounds_depth = backgrounds_depth
        print("build background images finished")

        if not self.cfg.NO_CACHE:
            with open(color_cache_file, 'wb') as fid:
                pickle.dump(backgrounds_color, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote backgrounds to {}'.format(color_cache_file))
        if not self.cfg.NO_CACHE:
            with open(depth_cache_file, 'wb') as fid:
                pickle.dump(backgrounds_depth, fid, pickle.HIGHEST_PROTOCOL)
            print('wrote backgrounds to {}'.format(depth_cache_file))
