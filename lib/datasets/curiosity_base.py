import torch
import torch.utils.data as data

import os, math, sys
import os.path as osp
from os.path import *
import numpy as np
import random
import numpy.random as npr
import cv2
import pickle
import scipy.io
import json
import csv

# from lib.datasets import ROOT_DIR
from lib.datasets.imdb import imdb
from lib.fcn.config import cfg
from lib.utils.blob import pad_im, add_noise_torch
from transforms3d.euler import euler2quat
from lib.utils.pose_error import *
from lib.utils.se3 import *
from lib.utils.print_and_log import print_and_log

from lib.utils.image_preprocess import color_jittering_torch
from lib.utils.get_closest_pose import get_closest_pose

from pyassimp import load
from lib.utils.back_projection import backproject_camera_tensor

class Curiosity_Base(data.Dataset, imdb):
    def __init__(self, image_set, dataset_root_path, mode, class_names=None):
        self.cfg = cfg
        self.renderer = None
        self.mode = mode

        self._name = 'curiosity_base_' + image_set
        self._image_set = image_set
        self._dataset_root_path = dataset_root_path
        self._model_list = self.load_model_list()
        self._model_list_len = len(self._model_list)
        self.model_info_file = os.path.join(self._dataset_root_path, 'models', 'model_info.json')
        self.ROOT_DIR = self.cfg.ROOT_DIR
        self.model_scale = 1 # the model points and translation is defined on mm

        # for all classes
        """ example:
        class_name: 002_master_chef_can
        class_id: 1
        class_idx: unknown
        class_color: [10, 10, 10]
        symmetry: 0
        sym_info: ndarray([0, 0, 0, 0])
        diameters: xxx
        extents: ndarray([x, x, x])
        """
        self.round=0
        self.num_models_each_round = min(self.cfg.DATA.LARGE_SCALE.NUM_MODELS_EACH_ROUND, self._model_list_len)
        self._class_names_all = ['__background__']
        self._sample_class_names_all(class_names)
        self._num_classes_all = len(self._class_names_all)
        self._class_ids_all = list(range(self._num_classes_all))
        assert(cfg.DATA.DISCRET_COLOR_STEP*self._num_classes_all<255)
        self._class_colors_all = [(x*self.cfg.DATA.DISCRET_COLOR_STEP,)*3 for x in range(len(self._class_names_all))]
        self._symmetry_all = np.array([0]*self._num_classes_all)
        self._sym_info_all = np.ones([len(self._class_names_all), 10, 4]) * -1 # use 10 as the max number of gt candidates
        if self.cfg.LOSS.USE_SYM_INFO != 'none':
            for i, cls_name in enumerate(self._class_names_all):
                sym_info = self._get_sym_info(cls_name)
                self._sym_info_all[self._cls2idx(cls_name), :sym_info.shape[0], :sym_info.shape[1]] = sym_info
        self._load_model_info()

        self.setup_synthetic()
        self.setup_classes()
        self.load_dataset()
        self.setup_others()

    def get_class_names_all(self):
        return self._class_names_all.copy()

    def _sample_class_names_all(self, class_names=None):
        if class_names is not None:
            self._class_names_all = [x for x in class_names]
        elif self.cfg.DATA.LARGE_SCALE.MODEL_SAMPLE_STRATEGY == 'sequential':
            start = self.round*self.num_models_each_round % self._model_list_len
            end = (self.round+1)*self.num_models_each_round % self._model_list_len
            if end<start:
                self._class_names_all += self._model_list[start:-1]+self._model_list[:end]
            else:
                self._class_names_all += self._model_list[start:end]
        elif self.cfg.DATA.LARGE_SCALE.MODEL_SAMPLE_STRATEGY == 'random':
            sample_idx = random.sample(range(self._model_list_len), self.num_models_each_round)
            sample_idx.sort()
            self._class_names_all += [self._model_list[i] for i in sample_idx]
        else:
            raise Exception("NOT IMPLEMENTED")
        print_and_log(self._class_names_all)

    def load_model_list(self):
        model_list_path = os.path.join(self._dataset_root_path, 'models', 'model_list.txt')
        with open(model_list_path, 'r') as f:
            model_list = [x.rstrip() for x in f.readlines()]
        return model_list

    def setup_classes(self):
        # for train/test classes (a subset of all classes)
        # self.class_ids = self.cfg.TRAIN.USING.CLASSES \
        #     if self.mode.lower() in ['train', 'val'] \
        #     else self.cfg.TEST.CLASSES
        self.class_ids = self._class_ids_all
        assert 0 in self.class_ids
        self._sym_info = np.array([self._sym_info_all[i] for i in self.class_ids])
        self._class_names_used = [self._class_names_all[i] for i in self.class_ids]
        self._class_colors = [self._class_colors_all[i] for i in self.class_ids]
        self._diameters = self._diameters_all[self.class_ids]
        self._symmetry = self._symmetry_all[self.class_ids]
        # print_and_log("symmetry for {}: {}".format(self.class_ids, [self._get_sym_info(n) for n in self._class_names_used]))
        self._extents = self._extents_all[self.class_ids]
        self._points_ori, self._points_neat, self._points_std = self._load_object_points()

        self.class_ids_occluder = self._class_ids_all[1:]
        self._num_classes_occluder = len(self.class_ids_occluder)

        self._class_id_to_index = dict(zip(self.class_ids, range(self.num_classes)))
        self._class_name_to_index = dict(zip(self._class_names_used, range(self.num_classes)))
        self._class_name_to_id = dict(zip(self._class_names_all, range(self.num_classes)))

    def get_next_ind(self):
        batch_size = self.cfg.TRAIN.USING.IMS_PER_BATCH
        if self.mode == 'TRAIN':
            if self._cur >= len(self._roidb)-batch_size:
                self.perm_if_needed()
            db_ind = self._perm[self._cur]
        else:
            if self._cur >= len(self._roidb):
                self._cur = 0
            db_ind = self._cur
        self._cur += 1
        return db_ind

    def setup_synthetic(self):
        # 3D model paths
        self._intrinsic_matrix = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02], \
                                           [0.000000e+00, 1.067487e+03, 2.413109e+02], \
                                           [0.000000e+00, 0.000000e+00, 1.000000e+00]])
        self._width = self.cfg.SYN.WIDTH
        self._height = self.cfg.SYN.HEIGHT
        # __background__ has a very simple model that will never be rendered
        self.model_mesh_paths = [os.path.join(self.ROOT_DIR, 'tools/default', '__background__', 'textured_simple.obj')]
        self.model_texture_paths = [os.path.join(self.ROOT_DIR, 'tools/default', '__background__', 'texture_map.png')]
        self.model_mesh_paths += ['{}/models/{}/model.obj'.format(self._dataset_root_path, n)
                                  for n in self._class_names_all if n != '__background__']
        self.model_texture_paths += ['{}/models/{}/texture_map.jpg'.format(self._dataset_root_path, n)
                                  for n in self._class_names_all if n != '__background__']
        #TODO support 255 objects at most currently
        self.model_colors = [np.array(self._class_colors_all[i]) / 255.0 for i in range(len(self._class_names_all))]

        margin = self.cfg.SYN.MARGIN
        self._valid_range = np.array([[margin, self._width - margin], [margin, self._height - margin], [self.cfg.SYN.TNEAR, self.cfg.SYN.TFAR]])

    def load_dataset(self):
        self._image_ext = 'jpg'

        self._roidb = self.gt_roidb()

        self.perm_if_needed()

        self._build_background_images()

    def perm_if_needed(self):
        if self.mode == 'TRAIN' or self.mode == 'VAL' or self.cfg.TEST.VISUALIZE:
            self._perm = np.random.permutation(np.arange(len(self._roidb)))
        else:
            self._perm = np.arange(len(self._roidb))
        self._cur = 0

    def initialize_evaluation(self):
        self._distance = [None for i in range(cfg.TEST.REF.ITERNUM+2)] # [ADD, ADD-S]
        self._correct_poses = np.zeros((self.num_classes, cfg.TEST.REF.ITERNUM + 2, 8), dtype=np.float32)
        self._total_poses = [0 for i in self.class_ids]
        self.acc = np.zeros((self.num_classes, cfg.TEST.REF.ITERNUM + 2, 10))

    def setup_others(self):
        self._width = self.cfg.SYN.WIDTH
        self._height = self.cfg.SYN.HEIGHT
        self.x2d_flatten = self._get_x2d_flatten(self._width, self._height)

        def generate_pose_anchors3(step_size):
            interval = step_size
            quats = []
            for yaw in range(-180, 180, interval):
                for pitch in range(-90, 90, interval):
                    for roll in range(-180, 180, interval):
                        quats.append(euler2quat(yaw, pitch, roll))
            anchors = np.array(quats)
            return anchors
        self.anchors_quat = generate_pose_anchors3(self.cfg.DATA.ANCHOR_STEPSIZE)
        self.anchors_trans = self._calculate_anchor_trans()
        self.anchors_mat = [quat2mat(q) for q in self.anchors_quat]

    def _calculate_anchor_trans(self):
        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        anchors_trans = np.ndarray([self.num_classes, 3])
        for class_idx, d in enumerate(self._diameters):
            cz_x = fx*d/self.cfg.DATA.ANCHOR_IMG_WIDTH
            cz_y = fy*d/self.cfg.DATA.ANCHOR_IMG_HEIGHT
            cz = max(cz_x, cz_y)
            cx = 0
            cy = 0
            anchors_trans[class_idx] = [cx, cy, cz]
        return anchors_trans

    def _get_x2d_flatten(self, width, height):
        # construct the 2D points matrix
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        ones = np.ones((height, width), dtype=np.float32)
        x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)
        return x2d.transpose()

    def depth2pcloud(self, depth, intrinsic_matrix):
        K = np.array(intrinsic_matrix)
        Kinv = np.linalg.inv(K)
        R = Kinv * self.x2d_flatten

        height = depth.shape[0]
        width = depth.shape[1]

        x3d = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R)
        pcloud = np.array(x3d).reshape([3, height, width])
        pcloud = pcloud.transpose([1,2,0])
        return pcloud

    def _get_sym_info(self, cls_name):
        return np.array([[0, 0, 1., -1]])

    def _cls2idx(self, cls_name):
        return self._class_names_all.index(cls_name)

    def _set_renderer_light(self):
        # sample lighting
        # light pose
        theta = np.random.uniform(-np.pi / 2, np.pi / 2)
        phi = np.random.uniform(0, np.pi / 2)
        r = np.random.uniform(0.25, 3.0)
        light_pos = [r * np.sin(theta) * np.sin(phi), r * np.cos(phi) + np.random.uniform(-2, 2),
                     r * np.cos(theta) * np.sin(phi)]
        self.renderer.set_light_pos(light_pos)

        # light color
        intensity = np.random.uniform(0.5, 2.0)
        # light_color = intensity * np.random.uniform(0.5, 1.5, 3)
        light_color = intensity * np.random.uniform(0.8, 1.2, 3)
        self.renderer.set_light_color(light_color)

    def _get_pose_anchor_label(self, gt_rot, sym_info):
        # TODO implement max vis area anchor
        dist2anchors = np.array([re(est_rot, get_closest_pose(est_rot, gt_rot, sym_info))
                                 for est_rot in self.anchors_mat])
        anchor_labels = (dist2anchors <= 45).astype(np.float32)
        if np.sum(anchor_labels) == 0:
            anchor_labels = (dist2anchors==np.min(dist2anchors)).astype(np.float32)

        return anchor_labels

    def _render_objects(self, num_targets, num_occluder, fg_class_ids):
        """
        give the class_id of foreground objects and background objects,
        return a rendered image
        :param num_targets: int, num of foreground objects
        :param num_occluder: int, num of background objects
        :param fg_class_ids: list of int, the class ids of these objects
        :return:
            fg_im: tensor HxWx3 float32, the image of rendered objects
            masks: tensor 1xHxW, float32, the foreground mask for each class, ordered by class_ids
            rend_poses: list of poses [(7,)]
        """
        height = self.cfg.SYN.HEIGHT
        width = self.cfg.SYN.WIDTH
        aspect_ratio = height/width

        fx = self._intrinsic_matrix[0, 0]
        fy = self._intrinsic_matrix[1, 1]
        px = self._intrinsic_matrix[0, 2]
        py = self._intrinsic_matrix[1, 2]
        zfar = 6.0
        znear = 0.01

        min_fg_ratio = 0
        count = 0
        bound = self.cfg.SYN.BOUND
        while min_fg_ratio < self.cfg.DATA.MIN_VIS_RATIO:
            # if count > 1:
            #     print_and_log("{}, {}, {}".format(count, min_fg_ratio, [self._class_names_all[class_id] for class_id in rend_class_ids]))
            poses_rend = []
            # sample other objects as distracts
            valid_occluder_ids = [i for i in self.class_ids_occluder if i not in fg_class_ids]
            occluder_ids = np.random.choice(valid_occluder_ids, size=num_occluder)
            rend_class_ids = np.concatenate([fg_class_ids, occluder_ids])
            num = num_targets + num_occluder
            for i in range(num):
                pose_rend = np.zeros((7,), dtype=np.float32)
                # rotation
                class_id = int(rend_class_ids[i])

                if i < num_targets:
                    # foreground object
                    if np.random.rand(1) >= self.cfg.SYN.POSE_UNIFORM_RATIO:
                        class_idx = self._class_id_to_index[class_id]
                        pose = self._get_pose(class_idx)

                        euler = pose[:3] + (self.cfg.SYN.STD_ROTATION * math.pi / 180.0) * np.random.randn(3)
                        pose_rend[3:] = euler2quat(euler[0], euler[1], euler[2])
                        self._pose_indexes[class_idx] += 1
                    else:
                        pose_rend[3:] = get_random_rotation()

                    m = self.cfg.SYN.MARGIN
                    center_range = np.array([[m, width-m], [m, height-m], [self.cfg.SYN.TNEAR, self.cfg.SYN.TFAR]])
                    pose_rend[:3] = get_valid_location(self._intrinsic_matrix, center_range)
                    # pose_rend[0] = np.random.uniform(-bound * aspect_ratio, bound * aspect_ratio)
                    # pose_rend[1] = np.random.uniform(-bound, bound)
                    # pose_rend[2] = np.random.uniform(self.cfg.SYN.TNEAR, self.cfg.SYN.TFAR)
                else:
                    # occluder
                    pose_rend[3:] = get_random_rotation()

                    # sample an object as occluder nearby
                    object_id = np.random.randint(0, num_targets)
                    extent = (self._diameters_all[object_id]+self._diameters_all[class_id])/2

                    fx = self._intrinsic_matrix[0, 0]
                    fy = self._intrinsic_matrix[1, 1]
                    # sample z
                    pose_rend[2] = poses_rend[object_id][2] - extent * np.random.uniform(1.0, 2.0)

                    oc_min = cfg.SYN.OCCLUDER_FACTOR[0]
                    oc_max = cfg.SYN.OCCLUDER_FACTOR[1]
                    # sample x
                    sign = np.random.choice([-1,1], 1)[0]
                    x = poses_rend[object_id][0]/poses_rend[object_id][2]*pose_rend[2]
                    offset = extent * (np.random.uniform(oc_min, oc_max))
                    pose_rend[0] = x + sign * offset

                    # sample y
                    sign = np.random.choice([-1,1], 1)[0]
                    y = poses_rend[object_id][1]/poses_rend[object_id][2]*pose_rend[2]
                    # offset = extent * (np.random.uniform(0.5, 0.7))
                    offset = extent * (np.random.uniform(oc_min, oc_max))
                    pose_rend[1] = y + sign * offset

                    #
                    if pose_rend[2] < self.cfg.SYN.TNEAR or count>cfg.SYN.MAX_REND_TRY:
                        pose_rend[2] = poses_rend[object_id][2] + extent * np.random.uniform(1.0, 2.0)

                poses_rend.append(pose_rend)

            self.renderer.set_poses(poses_rend)
            self._set_renderer_light()
            self.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)

            # rendering
            image_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
            seg_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
            pc2_tensor = torch.cuda.FloatTensor(height, width, 4).detach()
            self.renderer.render(rend_class_ids, image_tensor, seg_tensor, pc2_tensor=pc2_tensor)
            fg_im = image_tensor.flip(0)[:, :, [2,1,0]] # RGB2BGR
            seg_tensor = seg_tensor.flip(0)
            depth_tensor = pc2_tensor.flip(0)[:,:,2]

            # RGB to BGR order
            fg_im = torch.clamp(fg_im, 0., 1.)
            fg_im = fg_im * 255.

            im_label = torch.round(seg_tensor[:, :, 0] * 255 / self.cfg.DATA.DISCRET_COLOR_STEP)
            masks = torch.stack([im_label == class_id for class_id in rend_class_ids[:num_targets]], dim=0).float()

            self.renderer.set_poses(poses_rend[:num_targets])

            visiable_area = [torch.sum(masks[i]!=0).item() for i in range(num_targets)]
            full_area = []
            full_mask = []
            fg_poses = []
            for idx_in_img in range(num_targets):
                fg_poses.append(poses_rend[idx_in_img][[3,4,5,6,0,1,2]])
                self.renderer.render([rend_class_ids[idx_in_img]], image_tensor, seg_tensor, pc2_tensor=pc2_tensor)
                full_area.append(torch.sum(seg_tensor[:,:,0]!=0).item())
                full_mask.append((seg_tensor[:, :, 0]!=0).float())
            fg_poses = np.stack(fg_poses)
            fg_ratios = [visiable_area[i]/full_area[i] for i in range(num_targets)]
            full_masks = torch.stack(full_mask, dim=0)
            min_fg_ratio = min(fg_ratios)
            if count >= 10:
                break
            count += 1

        return fg_im, masks, fg_poses, depth_tensor, full_masks

    def _paste_bg_img(self, fg_im, depth_cuda, bg_mask):
        """

        :param fg_im: tensor, HxWx3, float32
        :param bg_mask: tensor, HxWx1, bool, mask of each instance
        :return:
        """
        # add background to the image
        ind = np.random.randint(len(self._backgrounds_color), size=1)[0]
        filename = self._backgrounds_color[ind]
        background_color = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

        try:
            # randomly crop a region as background
            bw = background_color.shape[1]
            bh = background_color.shape[0]
            x1 = npr.randint(0, int(bw / 3))
            y1 = npr.randint(0, int(bh / 3))
            x2 = npr.randint(int(2 * bw / 3), bw)
            y2 = npr.randint(int(2 * bh / 3), bh)
            background_color = background_color[y1:y2, x1:x2]
            background_color = cv2.resize(background_color, (self._width, self._height), interpolation=cv2.INTER_LINEAR)
        except:
            background_color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            print('bad background image')

        if len(background_color.shape) != 3:
            background_color = np.zeros((self._height, self._width, 3), dtype=np.uint8)
            print('bad background image')

        # paste objects on background
        background_cuda = torch.from_numpy(background_color.astype(np.float32)).detach().cuda()
        im_full = fg_im * (1-bg_mask) + background_cuda * bg_mask

        if len(self._backgrounds_depth) > 0:
            filename = self._backgrounds_depth[ind]
            background_depth = cv2.imread(filename, cv2.IMREAD_UNCHANGED)*self._background_depth_metric
            background_depth_cuda = torch.from_numpy(background_depth.astype(np.float32)).detach().cuda()
            bg_mask = bg_mask.squeeze()
            depth_full = depth_cuda * (1-bg_mask)+background_depth_cuda*bg_mask
        else:
            depth_full = depth_cuda

        return im_full, depth_full


    def _render_item(self):
        """

        :return:
            im_cuda: tensor HxWx3, cuda, float32 before adding noise
            masks: tensor 1xHxW, cuda, float32
            K: tensor, 3x3, cpu, float32, intrinsic_matrix
            class_id: int, the target object class_id
            pose: ndarray, 1x7, float32
        """
        # sample target objects
        min_fg_num = self.cfg.SYN.MIN_FG_OBJECT
        max_fg_num = np.minimum(self.num_classes - 1, self.cfg.SYN.MAX_FG_OBJECT)
        num_target = np.random.randint(min_fg_num, max_fg_num+1)
        cls_indexes = np.random.choice([i for i in range(1, self.num_classes)], size=num_target)
        fg_class_ids = [self.class_ids[i] for i in cls_indexes]

        num_occluder = min(self.cfg.SYN.NUM_OTHER_OBJECT,
                           self._num_classes_occluder)
        fg_im, masks, fg_poses, depth_cuda, full_masks = self._render_objects(num_target, num_occluder, fg_class_ids)

        # paste background image
        bg_mask = (fg_im.sum(dim=2) == 0).float().unsqueeze(2) # HxWx1
        im_cuda, depth_cuda = self._paste_bg_img(fg_im, depth_cuda, bg_mask)
        im_cuda /= 255.

        K = torch.FloatTensor(self._intrinsic_matrix)

        return im_cuda, masks, K, fg_class_ids, fg_poses, depth_cuda

    def read_roidb(self, roidb):
        """
        :param:
            image_path: image_path,
            depth_path: depth_path,
            mask_path:mask_path,
            bbox: bbox,
            px_count_visib: px_count_visib,
            visib_fract: visib_fract,
            pose: pose,
            class_id: class_id,
            K: K
        :return:
            im: tensor HxWx3, cpu, float32 before adding noise, BGR format
            masks: tensor 1xHxW, cpu, float32, mask for each object
            K: tensor, 3x3, intrinsic_matrix, float32
            class_id: int, the target object class_id
            pose: ndarray, 1x7, float32
        """
        # rgba
        rgba = pad_im(cv2.imread(roidb['image_path'], cv2.IMREAD_UNCHANGED), 16)
        if rgba.shape[2] == 4:
            im = np.copy(rgba[:, :, :3])
            alpha = rgba[:, :, 3]
            I = np.where(alpha == 0)
            im[I[0], I[1], :] = 0
        else:
            im = rgba
        im = torch.FloatTensor(im)

        # replace the background of some images
        if self.mode=='TRAIN' and np.random.rand()<self.cfg.DATA.REPLACE_REAL_BG_RATIO:
            mask_dir = os.path.join(os.path.dirname(roidb['mask_path']), '../mask')
            mask_name = os.path.basename(roidb['mask_path'])
            mask_prefix = mask_name.split('_')[0]
            all_mask_names = os.listdir(mask_dir)
            matched_mask_name = [n for n in all_mask_names if n.startswith(mask_prefix)]
            matched_mask_path = [os.path.join(mask_dir, n) for n in matched_mask_name]
            all_masks = [np.expand_dims(cv2.imread(p, cv2.IMREAD_UNCHANGED), 2) for p in matched_mask_path]
            all_masks = np.concatenate(all_masks, axis=2).astype(np.float32)
            bg_mask = torch.FloatTensor(np.sum(all_masks, axis=2)==0).unsqueeze(2)
            im = self._paste_bg_img(im, bg_mask)

        im /= 255.

        # im_label
        masks = torch.FloatTensor(cv2.imread(roidb['mask_path'], cv2.IMREAD_UNCHANGED)/255).unsqueeze(0)

        # quat_poses
        quat_pose = np.expand_dims(matPose2quatPose(roidb['pose']), 0)

        # K
        K = torch.FloatTensor(roidb['K'])

        # rgba
        depth = pad_im(cv2.imread(roidb['depth_path'], cv2.IMREAD_UNCHANGED), 16)
        depth = depth / self._depth_factor
        depth = torch.from_numpy(depth).float()

        return im, masks, K, quat_pose, depth

    def post_process_image(self, im_torch):
        """
        :param im_torch: tensor HxWx3, cuda, float32
        :return:
        """
        # chromatic transform
        if self.cfg.DATA.CHROMATIC and self.mode == 'TRAIN' and np.random.rand(1) > 0.1:

            im_torch = color_jittering_torch(im_torch)
        # motion blur
        if self.cfg.DATA.ADD_NOISE and self.mode == 'TRAIN' and np.random.rand(1) > 0.1:
            im_torch = add_noise_torch(im_torch)
        return im_torch

    def post_process_depth(self, depth_torch, im_label):
        """
        :param depth_torch: tensor HxW, cuda, float32
        :return:
        """
        if self.mode == 'TRAIN' and self.cfg.DATA.DEPTH_NOISE_STD > 0.:
            depth_torch = torch.normal(depth_torch, self.cfg.DATA.DEPTH_NOISE_STD)
        if self.mode == 'TRAIN' and self.cfg.DATA.ADD_HOLES > 0:
            mask_area = torch.sum(im_label)*self.cfg.DATA.ADD_HOLES
            patch_h = patch_w = int(torch.sqrt(mask_area).item())
            fg_y, fg_x = torch.nonzero(im_label[0], as_tuple=True)
            assert (len(fg_x)>0)
            x1 = torch.min(fg_x).item()
            y1 = torch.min(fg_y).item()
            x2 = torch.max(fg_x).item()
            y2 = torch.max(fg_y).item()
            if x2-patch_w<=x1 or y2-patch_h<=y1:
                # skip masking this sample
                print_and_log('skip masking this sample')
                patch_w = patch_h = 0
            mask_start_x = np.random.randint(x1, x2-patch_w)
            mask_start_y = np.random.randint(y1, y2-patch_h)
            mask = depth_torch.detach().clone().zero_()+1.
            mask[mask_start_y:mask_start_y+patch_h, mask_start_x:mask_start_x+patch_w] = 0.
            depth_torch *= mask.float()
        return depth_torch

    def process_targets(self, masks, fg_class_ids, fg_poses):
        """

        :param masks: tensor 1xHxW, masks with class_ids
        :param fg_class_id: int
        :param fg_poses: ndarray(1, 7)
        :return:
            targets: list of dict for each image
                    boxes: tensor 1x4, cuda, float32
                    class_id: tensor (1,) cuda, int32
                    class_idx: tensor (1,) cuda, int32
                    masks: tensor 1xHxW, cuda, float32
                    poses: tensor 1x7, cuda, float32
                    complete_boxes: tensor 1x4, cuda, float32
        """
        # PROCESS TARGETS
        # boxes, class labels and binary masks

        #TODO deal with situation when more than one fg object in one image

        # box
        boxes = []
        cur_binary_mask = masks[0]
        fg_y, fg_x = torch.nonzero(cur_binary_mask, as_tuple=True)
        assert(fg_x.shape)
        x1 = torch.min(fg_x)
        y1 = torch.min(fg_y)
        x2 = torch.max(fg_x)
        y2 = torch.max(fg_y)
        boxes.append([x1, y1, x2, y2])

        # class_idx
        class_idx = []
        fg_class_id = fg_class_ids[0]
        class_index = self._class_id_to_index[fg_class_id]
        class_idx.append(class_index)

        # complete box
        # including the invisible part, can out of image
        complete_boxes = []
        pixels = pts2pixels(self.get_point_clouds('std')[class_idx[0]], quat2mat(fg_poses[0, :4]), fg_poses[0, 4:], self._intrinsic_matrix)
        fg_x = pixels[:, 0]
        fg_y = pixels[:, 1]
        x1 = np.min(fg_x)
        y1 = np.min(fg_y)
        x2 = np.max(fg_x)
        y2 = np.max(fg_y)
        complete_boxes.append([x1, y1, x2, y2])

        # gather
        boxes = torch.FloatTensor(boxes).reshape(-1, 4)  # guard against no boxes
        class_id = torch.IntTensor([fg_class_id])
        class_idx = torch.IntTensor(class_idx)
        fg_pose_tensor = torch.FloatTensor(fg_poses)
        complete_boxes = torch.FloatTensor(complete_boxes)

        target = {
            'boxes': boxes,
            'class_id': class_id,
            'class_idx': class_idx,
            'masks': masks,
            'poses': fg_pose_tensor,
            'complete_boxes': complete_boxes
        }

        return target

    def update_external_result(self, roidb, targets):
        """

        :param
            roidb:image: str
                  box: str
                  depth: str
                  label: str
                  meta_data: str
                  video_id: str
                  image_id: str
                  posecnn: str
                  flipped: False
        :param
            targets: list of dict for each image
                    boxes: tensor 1x4, cuda, float32
                    class_id: tensor (1,) cuda, int32
                    class_idx: tensor (1,) cuda, int32
                    masks: tensor 1xHxW, cuda, float32
                    poses: tensor 1x7, cuda, float32
                    posecnn_results: tensor 1x7, cuda, float32
                    icp_results: tensor 1x7, cuda, float32
        :return:
        """
        if 'posecnn_result' in roidb:
            targets['posecnn_results'] = torch.FloatTensor(roidb['posecnn_result'])
            targets['icp_results'] = torch.FloatTensor(roidb['icp_result'])
        if 'cdpn_result' in roidb:
            targets['cdpn_results'] = torch.FloatTensor(roidb['cdpn_result'])
            targets['icp_results'] = torch.FloatTensor(roidb['icp_result'])
        else:
            targets['icp_results'] = targets['poses']

    def __getitem__(self, index):
        """

        :param index:
        :return:
            data_cuda: list of dict for each image
                    image: tensor HxWx3, cuda, float32
                    depth: tensor HxW, cuda, float32
                    pcloud: tensor HxWx3, cuda, float32
                    K: tensor 3x3, cuda, float32
            targets: list of dict for each image
                    boxes: tensor 1x4, cuda, float32
                    class_id: tensor (1,) cuda, int32
                    class_idx: tensor (1,) cuda, int32
                    masks: tensor 1xHxW, cuda, float32
                    poses: tensor 1x7, cuda, float32
                    # ---- optional ----
                    posecnn_results: tensor 1x7, cuda, float32
                    icp_results: tensor 1x7, cuda, float32
        """
        # import threading
        # thread_name = threading.current_thread().name
        # import os
        # pid = str(os.getpid())
        # print('thread_name: {}, pid: {}'.format(thread_name, pid))
        # print self.cfg.pid_renderer, pid in self.cfg.pid_renderer.keys()
        cpu_only=True
        # cpu_only = True if self.cfg.DATA.NUM_WORKERS>0 else False
        is_syn = 1

        # a=np.random.randint(1000000)
        # print(a)

        if is_syn:
            im, im_label, K, fg_class_ids, fg_poses, depth = self._render_item()
        else:
            # TODO: predetermined index order for multiprocessing
            db_ind = self.get_next_ind()
            roidb = self._roidb[db_ind]

            # Get the input image blob
            fg_class_ids = [roidb['class_id']]
            im, im_label, K, fg_poses, depth = self.read_roidb(roidb)

        im = self.post_process_image(im)
        if is_syn:
            depth = self.post_process_depth(depth, im_label)
        data = {'image': im,
                'K': K,
                'depth': depth,
                'pcloud': backproject_camera_tensor(depth, K)}

        target = self.process_targets(im_label, fg_class_ids, fg_poses)
        target['icp_results'] = target['poses']

        if not cpu_only:
            data = {k: v.cuda() for k, v in data.items()}
            target = {k: v.cuda() for k, v in target.items()}

        return data, target

    def get_anchors(self, type='both'):
        if type == 'quat':
            return self.anchors_quat
        elif type == 'mat':
            return self.anchors_mat
        elif type == 'both':
            return {'qt': self.anchors_quat,
                    'mat': self.anchors_mat,
                    'trans': self.anchors_trans}
        elif type == 'trans':
            return self.anchors_trans
        else:
            raise Exception("anchor type quat, mat or both")

    def get_point_clouds(self, type='neat'):
        if type=='ori':
            return self._points_ori
        elif type == 'neat':
            return self._points_neat
        elif type == 'std':
            return self._points_std
        else:
            raise Exception("point cloud type ori, neat or rescaled")

    def pose_anchor2full_pose(self, det, K=None):
        num_objects = det['labels'].shape[0]
        pose_quats = np.zeros((num_objects, 7), dtype=np.float32)
        pose_anchor_idx = np.zeros((num_objects,))
        pose_anchor_score = np.zeros((num_objects,), dtype=np.float32)
        box_ref = np.zeros((num_objects, 4), dtype=np.float32)
        for i in range(num_objects):
            cur_cls_index = det['labels'][i]
            pose_probs = det['pose_anchor_probs'][i]
            box = det['boxes'][i]
            box_cx = (box[0]+box[2])/2
            box_cy = (box[1]+box[3])/2
            box_w = box[2]-box[0]
            box_h = box[3]-box[1]
            if 'centers' in det:
                box_cx = det['centers'][i]
                box_cy = det['centers'][i]

            if K==None:
                K=self._intrinsic_matrix
            fx = K[0,0]
            fy = K[1,1]
            cx = K[0,2]
            cy = K[1,2]

            t_z0 = 1.0 # default
            t_x0 = (box_cx-cx)/fx
            t_y0 = (box_cy-cy)/fy

            cur_pose_anchor_idx = np.argmax(pose_probs)
            rot_mat = self.anchors_mat[cur_pose_anchor_idx]
            x3d = np.ones((4, self._points_std.shape[1]), dtype=np.float32)
            x3d[0, :] = self._points_std[cur_cls_index, :, 0]
            x3d[1, :] = self._points_std[cur_cls_index, :, 1]
            x3d[2, :] = self._points_std[cur_cls_index, :, 2]
            RT = np.zeros((3, 4), dtype=np.float32)
            RT[:3, :3] = rot_mat
            RT[:, 3] = [t_x0,t_y0,t_z0]
            x2d = np.matmul(self._intrinsic_matrix, np.matmul(RT, x3d))
            x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
            x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

            x1 = np.min(x2d[0, :])
            y1 = np.min(x2d[1, :])
            x2 = np.max(x2d[0, :])
            y2 = np.max(x2d[1, :])

            box_w0 = x2-x1
            box_h0 = y2-y1

            # FIXME: fix the bug here
            # FIXME: it will be a problem if the object is occluded
            t_zw = box_w0/box_w*t_z0
            t_zh = box_h0/box_h*t_z0
            t_z = (t_zw+t_zh)/2
            t_x = t_x0/t_z0*t_z
            t_y = t_y0/t_z0*t_z

            pose_quats[i, :4] = self.anchors_quat[cur_pose_anchor_idx]
            pose_quats[i, 4:] = [t_x, t_y, t_z]

            x2d[0, :] = x2d[0, :] * x2d[2, :] / (x2d[2, :] - t_z0 + t_z)
            x2d[1, :] = x2d[1, :] * x2d[2, :] / (x2d[2, :] - t_z0 + t_z)

            x1 = np.min(x2d[0, :])
            y1 = np.min(x2d[1, :])
            x2 = np.max(x2d[0, :])
            y2 = np.max(x2d[1, :])

            box_ref[i] = [x1, y1, x2, y2]

            pose_anchor_idx[i] = cur_pose_anchor_idx
            pose_anchor_score[i] = pose_probs[cur_pose_anchor_idx]
        return pose_quats, pose_anchor_idx, pose_anchor_score, box_ref

    def __len__(self):
        if self.mode == 'TRAIN':
            self._size = self.cfg.TRAIN.USING.IMGS_EACH_EPOCH
        elif self.mode == 'VAL':
            self._size = self.cfg.VAL.SYN_LENGTH
        elif self.mode == 'TEST':
            self._size = self.cfg.TEST.REF.SYN_LENGTH

        return self._size

    def get_roidb_name(self, class_id):
        roidb_name = os.path.join(self.cache_path, self._name + '_class_{:03d}_roidb.pkl'.format(class_id))
        return roidb_name

    def try_save_roidb(self, data, class_id):
        cache_file = self.get_roidb_name(class_id)
        with open(cache_file, 'wb') as fid:
            pickle.dump(data, fid, pickle.HIGHEST_PROTOCOL)
        print('cache {} saved successful!'.format(cache_file))

    def try_load_roidb(self, class_id):
        cache_file = self.get_roidb_name(class_id)
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                data = pickle.load(fid)
            print('cache {} loaded successful!'.format(cache_file))
            return True, data
        else:
            return False, None

    def _load_object_points(self):
        """

        :return:
        points_ori: raw points
        points_neat: points that are cut to same size
        points_std: rescale them to adjust the loss
        """
        points_ori = [[] for _ in range(len(self._class_names_used))]
        # num = np.inf
        num = 100000
        for i in range(1, self.num_classes):
            class_id = self.class_ids[i]
            model_path = self.model_mesh_paths[class_id]
            assert os.path.exists(model_path), 'Path does not exist: {}'.format(model_path)
            model = load(model_path)
            assert (len(model.meshes) == 1)
            vertex = model.meshes[0].vertices
            points = np.unique(vertex, axis=0)
            points_ori[i] = points
            if self._class_names_used[i] != '__background__':
                num = min(num, points_ori[i].shape[0])

        print("cut points cloud to {}".format(num))
        points_neat = np.zeros((self.num_classes, num, 3), dtype=np.float32)
        for i in range(1, self.num_classes):
            sample_points_idx = np.arange(points_ori[i].shape[0])
            np.random.shuffle(sample_points_idx)
            sample_points_idx = sample_points_idx[:num]
            points_neat[i, :, :] = points_ori[i][sample_points_idx, :]

        points_std = points_neat.copy()*self.model_scale
        if self.mode=='VAL':
            # accerlerate validation
            model_downsample_factor = num//2600+1
            points_std = points_std[:, ::model_downsample_factor, :]

        return points_ori, points_neat, points_std

    def _load_model_info(self):
        with open(self.model_info_file) as f:
            model_info = json.load(f)
        extents = [[0,0,0]]
        diameters = [0,]
        for class_name in self._class_names_all:
            # add the extents of the background
            if class_name == '__background__':
                continue
            d = model_info[class_name]
            extents.append([d['size_x'], d['size_y'], d['size_z']])
            diameters.append(d['diameter'])
        extents *= self.model_scale
        diameters *= self.model_scale
        self._extents_all = np.asarray(extents)
        self._diameters_all = np.asarray(diameters)

    # image
    def image_path_from_index(self, scene_name, image_index):
        """
        Construct an image path from the image's "index" identifier.
        """

        image_path = os.path.join(self._data_path, scene_name, 'rgb',
                                  '{:06d}.{}'.format(image_index, self._image_ext))
        assert os.path.exists(image_path), \
            'Path does not exist: {}'.format(image_path)
        return image_path

    # depth
    def depth_path_from_index(self, scene_name, image_index):
        """
        Construct an depth path from the image's "index" identifier.
        """
        depth_path = os.path.join(self._data_path, scene_name, 'depth',
                                  '{:06d}.png'.format(image_index))
        assert os.path.exists(depth_path), \
            'Path does not exist: {}'.format(depth_path)
        return depth_path

    def label_path_from_index(self, scene_name, image_index, idx_in_img):
        """
        Construct an metadata path from the image's "index" identifier.
        """
        label_path = os.path.join(self._data_path, scene_name, 'mask_visib',
                                  '{:06d}_{:06d}.png'.format(image_index, idx_in_img))
        assert os.path.exists(label_path), \
            'Path does not exist: {}'.format(label_path)
        return label_path

    @staticmethod
    def load_and_process_bop_result(bop_result_path):
        assert os.path.exists(bop_result_path), "{} not found".format(bop_result_path)
        bop_raw_results = load_bop_results(bop_result_path) # list of dict with correct dtype
        for idx, dict in enumerate(bop_raw_results):
            scene_image_id = "{:06d}-{:06d}".format(int(dict['scene_id']), int(dict['im_id']))
            bop_raw_results[idx]['scene_image_id'] = scene_image_id
        bop_result = {}
        for idx, dict in enumerate(bop_raw_results):
            scene_image_id = dict['scene_image_id']
            bop_result[scene_image_id] = {d['obj_id']: d for d in bop_raw_results if
                                            d['scene_image_id'] == scene_image_id}
        return bop_result

    def gt_roidb(self):
        """
        Return the database of ground-truth regions of interest.

        This function loads/saves from/to a cache file to speed up future calls.
        """
        gt_roidb_all = []

        return gt_roidb_all

    def evaluate_pose(self, pose):
        """
        evaluate pose estimation
        """
        poses_init = pose['poses_init']
        poses_gt = pose['poses_gt']
        if 'poses_icp' in pose:
            poses_icp = pose['poses_icp']
        if 'poses_ref' in pose:
            poses_ref = pose['poses_ref']
        intrinsic_matrix = pose['intrinsic_matrix']
        class_idxs = pose['class_idxs']

        self.initialize_evaluation()
        self._evaluate(0, poses_init, poses_gt, intrinsic_matrix, class_idxs)

        num_iterations = poses_ref.shape[1]
        for i in range(num_iterations):
            self._evaluate(i+1, poses_ref[:, i, :], poses_gt, intrinsic_matrix, class_idxs)

        self._evaluate(num_iterations+1, poses_icp, poses_gt, intrinsic_matrix, class_idxs)

    def measure_distance(self, pose_1, pose_2):
        dist = np.zeros([pose_1.shape[0], 2])
        for i in range(pose_1.shape[0]):
            if pose_1[i, -1]>=990:
                continue

            RT_1 = np.zeros((3, 4), dtype=np.float32)
            RT_1[:3, :3] = quat2mat(pose_1[i, :4])
            RT_1[:, 3] = pose_1[i, 4:]

            RT_2 = np.zeros((3, 4), dtype=np.float32)
            RT_2[:3, :3] = quat2mat(pose_2[i, :4])
            RT_2[:, 3] = pose_2[i, 4:]

            dist_rot = re(RT_1[:3, :3], RT_2[:3, :3])
            dist_tran = te(RT_1[:, 3], RT_2[:, 3])

            dist[i, 0] = dist_rot
            dist[i, 1] = dist_tran

        return dist

    def _evaluate(self, ind, pose_est, pose_tgt, intrinsic_matrix, class_idxs):
        # TODO: implement dict based result
        # if self.cfg.TEST.VISUALIZE:
        #     print_and_log('iteration %d' % (ind))

        # for each object
        self._distance[ind] = np.ones((self.num_classes, pose_est.shape[0], 2), dtype=np.float32) * -1 # [ADD, ADD-S]
        for i in range(pose_est.shape[0]):
            class_idx = int(class_idxs[i])
            if ind == 0:
                self._total_poses[class_idx] += 1
            # if all(pose_est[i, 2:]==0):
            if pose_est[i, -1]==999:
                self._distance[ind][class_idx, i, :] = np.inf
                continue
            RT_est = np.zeros((3, 4), dtype=np.float32)
            RT_est[:3, :3] = quat2mat(pose_est[i, :4])
            RT_est[:, 3] = pose_est[i, 4:]

            RT_tgt = np.zeros((3, 4), dtype=np.float32)
            RT_tgt[:3, :3] = quat2mat(pose_tgt[i, :4])
            RT_tgt[:, 3] = pose_tgt[i, 4:]

            error_rot = re(RT_est[:3, :3], RT_tgt[:3, :3])
            error_tran = te(RT_est[:, 3], RT_tgt[:, 3])

            if error_rot < 5.0 and error_tran < 0.05:
                self._correct_poses[class_idx, ind, 0] += 1
            error = add(RT_est[:3, :3], RT_est[:, 3], RT_tgt[:3, :3], RT_tgt[:, 3], self._points_std[class_idx])
            self._distance[ind][class_idx, i, 0] = error
            if error < 0.02:#self._diameters[cls] * 0.1:
                self._correct_poses[class_idx, ind, 1] += 1
            error = adi(RT_est[:3, :3], RT_est[:, 3], RT_tgt[:3, :3], RT_tgt[:, 3], self._points_std[class_idx])
            self._distance[ind][class_idx, i, 1] = error
            if error < 0.02:#self._diameters[cls] * 0.1:
                self._correct_poses[class_idx, ind, 2] += 1
            # if all(RT_est[:, 3]==0):
            if RT_est[2, 3]==999:
                self._distance[ind][class_idx, i, :] = np.inf

            # if self.cfg.TEST.VISUALIZE:
            #     print_and_log( 'average distance error: {}'.format(error))

            # reprojection error
            error_reprojection = reproj(intrinsic_matrix[i], RT_est[:3, :3], RT_est[:, 3], RT_tgt[:3, :3], RT_tgt[:, 3], self._points_std[class_idx])
            if error_reprojection < 5.0:
                self._correct_poses[class_idx, ind, 3] += 1
            # if self.cfg.TEST.VISUALIZE:
            #     print_and_log( 'reprojection error: {}'.format(error_reprojection))

            error_xy = np.linalg.norm(RT_est[:2, 3]-RT_tgt[:2, 3])
            error_z = np.linalg.norm(RT_est[2, 3]-RT_tgt[2, 3])
            if error_xy < 0.03:
                self._correct_poses[class_idx, ind, 4] += 1
            if error_z < 0.04:
                self._correct_poses[class_idx, ind, 5] += 1

            if error_rot < 5:
                self._correct_poses[class_idx, ind, 6] += 1
            if error_tran < 0.05:
                self._correct_poses[class_idx, ind, 7] += 1

        # np.save('{}/{}_iter{}.npy'.format('/home/yili/PoseEst/deepim-pytorch/output/ycb_video/results_for_plot', self.cfg.CFG_NAME, ind), self._distance[ind])

    def voc_ap(self, rec, prec, use_07_metric=False):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:False).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))

            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]

            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def VOCap(self, rec, prec):
        index = np.where(np.isfinite(rec))[0]
        rec = rec[index]
        prec = prec[index]
        if len(rec) == 0 or len(prec) == 0:
            ap = 0
        else:
            mrec = np.insert(rec, 0, 0)
            mrec = np.append(mrec, 0.1)
            mpre = np.insert(prec, 0, 0)
            mpre = np.append(mpre, prec[-1])
            for i in range(1, len(mpre)):
                mpre[i] = max(mpre[i], mpre[i - 1])
            i = np.where(mrec[1:] != mrec[:-1])[0] + 1
            ap = np.sum(np.multiply(mrec[i] - mrec[i - 1], mpre[i])) * 10
        return ap

    def print_pose_accuracy(self, mute=False):
        for cls_ind, abs_ind in enumerate(self.class_ids):
            if cls_ind == 0:
                continue
            num_poses_cur_class = self._total_poses[cls_ind]
            if not mute:
                print_and_log('============================================')
                print_and_log('%d objects in %s' % (num_poses_cur_class, self._class_names_all[abs_ind]))
            for i in range(cfg.TEST.REF.ITERNUM + 2):
                if not mute:
                    if i == 0:
                        prefix = 'initial pose'
                    elif i==cfg.TEST.REF.ITERNUM+1:
                        prefix = 'icp result'
                    else:
                        prefix = 'iteration %d' % (i)
                    print_and_log(prefix)
                self.acc[cls_ind, i, 0] = self._correct_poses[cls_ind, i, 0] / num_poses_cur_class
                self.acc[cls_ind, i, 1] = self._correct_poses[cls_ind, i, 1] / num_poses_cur_class
                self.acc[cls_ind, i, 2] = self._correct_poses[cls_ind, i, 2] / num_poses_cur_class
                self.acc[cls_ind, i, 3] = self._correct_poses[cls_ind, i, 3] / num_poses_cur_class

                dist_add = self._distance[i][cls_ind, :, 0].copy()
                valid_ind = np.where(dist_add >= 0)[0]
                if not mute:
                    print("instance number check {} vs. {}".format(len(valid_ind), num_poses_cur_class))
                dist_add = dist_add[valid_ind]
                max_distance = 0.1
                outlier_ind = np.where(dist_add > max_distance)[0]
                if len(outlier_ind) > 0:
                    dist_add[outlier_ind] = np.inf
                d = np.sort(dist_add)
                n = len(d)
                accuracy = np.cumsum(np.ones((n,), np.float32)) / n
                aps = self.VOCap(d, accuracy)
                self.acc[cls_ind, i, 4] = aps

                dist_adi = self._distance[i][cls_ind, :, 1].copy()
                valid_ind = np.where(dist_adi >= 0)[0]
                dist_adi = dist_adi[valid_ind]
                max_distance = 0.1
                outlier_ind = np.where(dist_adi > max_distance)[0]
                if len(outlier_ind) > 0:
                    dist_adi[outlier_ind] = np.inf
                d = np.sort(dist_adi)
                n = len(d)
                accuracy = np.cumsum(np.ones((n,), np.float32)) / n
                aps = self.VOCap(d, accuracy)
                self.acc[cls_ind, i, 5] = aps

                self.acc[cls_ind, i, 6] = self._correct_poses[cls_ind, i, 4] / num_poses_cur_class
                self.acc[cls_ind, i, 7] = self._correct_poses[cls_ind, i, 5] / num_poses_cur_class
                self.acc[cls_ind, i, 8] = self._correct_poses[cls_ind, i, 6] / num_poses_cur_class
                self.acc[cls_ind, i, 9] = self._correct_poses[cls_ind, i, 7] / num_poses_cur_class
                if not mute:
                    print_and_log('5 degree, 5cm accuracy:   %.2f (%.2f, %.2f)' % (self.acc[cls_ind, i, 0]*100., self.acc[cls_ind, i, 8]*100., self.acc[cls_ind, i, 9]*100.))
                    print_and_log('6D pose accuracy ADD:     %.2f' % (self.acc[cls_ind, i, 1]*100.))
                    print_and_log('6D pose accuracy ADI:     %.2f' % (self.acc[cls_ind, i, 2]*100.))
                    print_and_log('Reprojection 2D accuracy: %.2f' % (self.acc[cls_ind, i, 3]*100.))
                    print_and_log('VOCap ADD:                %.2f' % (self.acc[cls_ind, i, 4]*100.))
                    print_and_log('VOCap ADD-S:              %.2f' % (self.acc[cls_ind, i, 5]*100.))

            if i == cfg.TEST.REF.ITERNUM - 1 or not mute:
                print_and_log('============================================')

        if not mute:
            if len(self.class_ids) >= 1:

                for i in [cfg.TEST.REF.ITERNUM]:
                    print_and_log('============================================')
                    if i == 0:
                        prefix = 'initial pose'
                    elif i == cfg.TEST.REF.ITERNUM + 1:
                        prefix = 'icp result'
                    else:
                        prefix = 'iteration %d' % (i)
                    print_and_log(prefix)
                    for metric_id in range(6):
                        metric_list = ['5d 5cm', '6D Pose ADD <0.2cm', '6D Pose ADI <0.2cm', 'ReProj 2D Acc', 'VOCap ADD', 'VOCap ADI']
                        metric = metric_list[metric_id]
                        print_and_log('-------------       ---------------')
                        print_and_log(metric)
                        for cls_ind, abs_ind in enumerate(self.class_ids):
                            if abs_ind == 0:
                                continue
                            print_and_log('{: <25} {:.2f}'.format(self._class_names_all[abs_ind], self.acc[cls_ind, i, metric_id] * 100.))

                add_and_adds = np.zeros(self.acc[:, :, :2].shape)
                for n in range(add_and_adds.shape[0]):
                    if self._symmetry[n] == 0:
                        add_and_adds[n, :, 0] = self.acc[n, :, 1]
                        add_and_adds[n, :, 1] = self.acc[n, :, 4]
                    else:
                        add_and_adds[n, :, 0] = self.acc[n, :, 2]
                        add_and_adds[n, :, 1] = self.acc[n, :, 5]

                for i in [0, cfg.TEST.REF.ITERNUM, cfg.TEST.REF.ITERNUM+1]:
                    print_and_log('============================================')
                    if i == 0:
                        prefix = 'averaged initial pose'
                    elif i==cfg.TEST.REF.ITERNUM+1:
                        prefix = 'averaged icp result'
                    else:
                        prefix = 'averaged iteration %d' % (i)
                    print_and_log(prefix)

                    print_and_log('5 degree, 5cm accuracy:   %.1f' % (np.average(self.acc[1:, i, 0])*100.))
                    print_and_log('6D pose accuracy ADD:     %.1f' % (np.average(self.acc[1:, i, 1])*100.))
                    print_and_log('6D pose accuracy ADI:     %.1f' % (np.average(self.acc[1:, i, 2])*100.))
                    print_and_log('Reprojection 2D accuracy: %.1f' % (np.average(self.acc[1:, i, 3])*100.))
                    print_and_log('VOCap ADD:                %.1f' % (np.average(self.acc[1:, i, 4])*100.))
                    print_and_log('VOCap ADD-S:              %.1f' % (np.average(self.acc[1:, i, 5])*100.))

                    print_and_log('6D Pose Acc ADD & ADD-S   %.1f' % (np.average(add_and_adds[1:, i, 0])*100.))
                    print_and_log('VOCap ADD & ADD-S         %.1f' % (np.average(add_and_adds[1:, i, 1])*100.))

        return self.acc

    def __trackitem__(self, image):
        #TODO not updated
        # Get the input image blob
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        im_blob = (image - self.cfg.PIXEL_MEANS)/ 255.0
        height, width, channel = image.shape
        im_scale = 1
        depth_blob = np.zeros([height, width, 3])

        test_rotate = True
        K = self._intrinsic_matrix
        Kinv = np.linalg.pinv(K)
        meta_data_blob = np.zeros(18, dtype=np.float32)
        meta_data_blob[0:9] = K.flatten()
        meta_data_blob[9:18] = Kinv.flatten()

        # 1 by 1 rotate pose experiment
        im_info = np.array([image.shape[0], image.shape[1], im_scale], dtype=np.float32)
        im_blob = torch.Tensor(im_blob).unsqueeze(0)
        depth_blob = torch.Tensor(depth_blob).unsqueeze(0)
        im_info = torch.Tensor(im_info).unsqueeze(0)
        meta_data_blob = torch.Tensor(meta_data_blob).unsqueeze(0)
        pose_blob = torch.Tensor(np.zeros((1, 9))).unsqueeze(0)
        pose_blob[:, :, -1] = 1
        gt_boxes = torch.Tensor(np.zeros((1, 5))).unsqueeze(0)
        pose_result = torch.Tensor(np.zeros((1, 9))).unsqueeze(0)
        _extents = torch.Tensor(self._extents).unsqueeze(0)
        _point_blob = torch.Tensor(self._points_std).unsqueeze(0)
        # label_blob = torch.Tensor(np.zeros((image.shape[0], image.shape[1]))).unsqueeze(0)
        sym_info = torch.Tensor(self._sym_info).unsqueeze(0)
        sample = {'image': im_blob,
                  'depth': depth_blob,
                  'meta_data': meta_data_blob,
                  'poses': pose_blob,
                  'extents': _extents,  # self._extents
                  'points': _point_blob,
                  'gt_boxes': gt_boxes,
                  'poses_result': pose_result,
                  'im_info': im_info,
                  'sym_info': sym_info}
        # print(pose_result.shape, pose_blob.shape, self._image_index[index])
        return sample


        # sample = {'image': im_cuda,
        #           'depth': depth,
        #           'meta_data': meta_data_blob,
        #           'label_blob': label_blob,
        #           'poses': pose_blob,
        #           'extents': self._extents[[int(pose_blob[0,1])]],
        #           'points': self._points_rescaled[[int(pose_blob[0,1])]],
        #           'gt_boxes': gt_boxes,
        #           'poses_result': pose_result,
        #           'im_info': im_info,
        #           'sym_info': self._sym_info}