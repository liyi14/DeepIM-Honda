# --------------------------------------------------------
# FCN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import torch
# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
import torch.nn.functional as F

import time

from lib.utils.deepim_utils import *
from transforms3d.euler import euler2quat, quat2euler
from lib.utils.lr_scheduler import WarmUpMultiStepLR, FlatCosineAnnealingLR

from ycb_render.ycb_renderer import YCBRenderer
import lib.networks as networks
import lib.utils.vision_utils as utils
from lib.utils.se3 import *
from lib.utils.get_closest_pose import get_closest_pose
from lib.utils.pose_error import re_quat
import cv2
import os
from math import pi
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from tqdm import tqdm
from lib.utils.model_utils import load_model_points

class DeepCodeMatchingTraining(object):
    def __init__(self, config, args, dataset_train, dataset_test, mode='TRAIN', phase='REF'):
        self.cfg = config
        self._args = args
        # self._start_iter_train = args.startiter+1
        self._mode = mode
        self._phase = phase
        self._dataset_train = dataset_train
        self._dataset_test = dataset_test
        self._dataset = self._dataset_train if self.at_train() else self._dataset_test
        self._optimizer_cfg = self.cfg.TRAIN.REF if self._phase=='REF' else self.cfg.TRAIN.DET
        self._network_cfg = self.cfg.NETWORK.REF if self._phase=='REF' else self.cfg.NETWORK.DET
        self._loss_cfg = self.cfg.LOSS.REF if self._phase=='REF' else self.cfg.LOSS.DET
        self._test_cfg = self.cfg.TEST.REF if self._phase=='REF' else self.cfg.TEST.DET
        self.num_gpus = len(self.cfg.GPU_ID)
        self.round = 0
        self._dcm_det = None
        self._dcm_refine = None
        self._optimizer = None
        self._scheduler = None
        self.iter_train = 0
        self.tracking = False

        # set points, extents, pixel_mean and pixel_std
        self.setup_constants()
        self.build_tf_log()

    def at_train(self):
        return self._mode=='TRAIN'
    def at_val(self):
        return self._mode=='VAL'
    def at_test(self):
        return self._mode=='TEST'

    def set_train(self):
        if self._phase=='REF':
            self._dcm_refine.train()
        else:
            self._dcm_det.train()
        self._mode = 'TRAIN'

    def set_val(self):
        if self._phase=='REF':
            self._dcm_refine.eval()
        else:
            self._dcm_det.eval()
        self._mode = 'VAL'

    def set_test(self):
        if self._phase=='REF':
            self._dcm_refine.eval()
        else:
            self._dcm_det.eval()
        self._mode = 'TEST'

    def build_tf_log(self):
        if self.at_train():
            train_output_dir = self.cfg.TRAIN.USING.OUTPUT_DIR
            final_model_path = os.path.join(train_output_dir, self._phase.lower()+'_model_final.pth')
            log_dir = os.path.join(
                train_output_dir, 'tf_log')
            if not os.path.exists(final_model_path) and os.path.exists(log_dir):
                import shutil
                shutil.rmtree(log_dir)
            self.tf_logger = SummaryWriter(log_dir)

    def setup_constants(self):
        """

        :return:
            points: tensor Cx4x2600, float32
            sym_info: ndarray CxTx
            extents:  tensor Cx3, float32
            pixel_mean: tensor 1x3x1x1, float32
            pixel_std: tensor 1x3x1x1, float32
        """
        points = torch.FloatTensor(self._dataset._points_std).cuda()
        points = torch.cat([points,
                            torch.ones([points.shape[0], points.shape[1], 1],
                                       dtype=points.dtype).cuda()], dim=2)
        self._points = points.permute((0,2,1)) # Nx4x2600
        self._sym_info = self._dataset._sym_info
        self._extents = torch.FloatTensor(self._dataset._extents).cuda()
        self._diameters = torch.FloatTensor(self._dataset._diameters).cuda()
        self._anchors_quat = self._dataset.get_anchors('quat')
        self._anchors_trans = self._dataset.get_anchors('trans')
        self._model_xyz, self._model_normal, self._model_rgb = self.load_xyzrgb()

        # self._pixel_mean_cpu = [0.485, 0.456, 0.406] # RGB order
        # self._pixel_std_cpu = [0.229, 0.224, 0.225]
        basenet_type = self.cfg.NETWORK.REF.BASENET_TYPE if self._phase=='REF' else self.cfg.NETWORK.DET.BASENET_OBS
        if basenet_type.startswith('flownets'):
            self._pixel_mean = torch.reshape(torch.cuda.FloatTensor([0.45, 0.432, 0.411]), (1, 3, 1, 1))  # BGR order
            self._pixel_std = torch.reshape(torch.cuda.FloatTensor([1, 1, 1]), (1, 3, 1, 1))
        else:
            self._pixel_mean = torch.reshape(torch.cuda.FloatTensor([0.406, 0.456, 0.485]), (1,3,1,1)) #BGR order
            self._pixel_std = torch.reshape(torch.cuda.FloatTensor([0.225, 0.224, 0.229]), (1,3,1,1))

        self.model_scale = self._dataset.model_scale

    def update_constants_for_val(self):
        points = torch.FloatTensor(self._dataset_test._points_std).cuda()
        points = torch.cat([points,
                            torch.ones([points.shape[0], points.shape[1], 1],
                                       dtype=points.dtype).cuda()], dim=2)
        self._points = points.permute((0,2,1)) # Nx4x2600
        self._model_xyz, self._model_normal, self._model_rgb = self.load_xyzrgb('test')
        self._sym_info = self._dataset_test._sym_info
        self._extents = torch.FloatTensor(self._dataset_test._extents).cuda()
        self._diameters = torch.FloatTensor(self._dataset_test._diameters).cuda()
        self._anchors_quat = self._dataset_test.get_anchors('quat')
        self._anchors_trans = self._dataset_test.get_anchors('trans')

    def get_num_iters_refine(self):
        if self.at_train():
            if self._optimizer_cfg.HEATUP_STRATEGY == 'SMOOTH':
                num_iters_refine = min(self.iter_train / self._optimizer_cfg.HEATUP_IMG,
                                       self._optimizer_cfg.ITERNUM)
                num_iters_refine = max(num_iters_refine, 1)
            elif self._optimizer_cfg.HEATUP_STRATEGY == 'DRASTIC':
                num_iters_refine = 1 if self.iter_train < self._optimizer_cfg.HEATUP_IMG \
                    else self._optimizer_cfg.ITERNUM
            else:
                num_iters_refine = self._optimizer_cfg.ITERNUM
        else:
            num_iters_refine = self.cfg.TEST.REF.ITERNUM
        return int(num_iters_refine)

    @staticmethod
    def get_angle_distance_batch(pose_src_blob, pose_tgt_blob, sym_info):
        num = pose_src_blob.shape[0]
        poses_src = pose_src_blob.cpu().numpy()
        poses_tgt = pose_tgt_blob.cpu().numpy()
        errors_rot = torch.cuda.FloatTensor(num, 1).detach()
        closest_gt = get_closest_pose_batch_cpu(poses_src, poses_tgt, sym_info)
        for i in range(num):
            errors_rot[i, 0] = np.arccos(2 * np.power(np.dot(poses_src[i, 2:6], closest_gt[i, 2:6]), 2) - 1)
        return errors_rot

    @staticmethod
    def get_angle_distance(quat1, quat2):
        return np.arccos(2 * np.power(np.dot(quat1, quat2), 2) - 1)/3.14159*180.

    def get_regressor_type(self, angular_distance):
        # 0 for over 30, 1 for below 30
        return (angular_distance < self.cfg.NETWORK.REG_THRESHOLD/180*3.14).type(torch.long).squeeze()

    def round_step(self, dataset_train, dataset_test):
        self.round += 1
        self.set_dataset(dataset_train, dataset_test)
        self.clean_renderer()
        self.build_renderer()
        self.build_dataloader()
        self.setup_constants()

    def set_dataset(self, dataset_train, dataset_test):
        self._dataset_train = dataset_train
        self._dataset_test = dataset_test
        self._dataset = self._dataset_train if self.at_train() else self._dataset_test

    def setup(self, network_data=None):
        self.build_renderer()
        self.build_dataloader()
        self.load_network(network_data)
        if self.at_train():
            self.build_optimizer_and_scheduler()

    def build_renderer(self, type='default'):
        print_and_log('=> loading 3D models')
        if type == 'default':
            d = self._dataset
        elif type == 'train':
            d = self._dataset_train
        elif type == 'test':
            d = self._dataset_test

        renderer = YCBRenderer(width=self.cfg.SYN.WIDTH,
                               height=self.cfg.SYN.HEIGHT,
                               gpu_id=self.cfg.ABS_GPU_ID[0],
                               render_marker=False,
                               model_scale=d.model_scale)
        renderer.load_objects(d.model_mesh_paths,
                              d.model_texture_paths,
                              d.model_colors)
        renderer.set_camera_default()
        # if type == 'default':
        self.renderer = renderer
        d.renderer = renderer
        if type=='default' and self._dataset_test._class_names_all == self._dataset._class_names_all:
            self._dataset_test.renderer = renderer

    def load_xyzrgb(self, type='default'):
        if 'BASENET_REND' in self._network_cfg and self._network_cfg.BASENET_REND.startswith('pointnet'):
            print_and_log('=> loading 3D point cloud')
            if type == 'default':
                d = self._dataset
            elif type == 'train':
                d = self._dataset_train
            elif type == 'test':
                d = self._dataset_test
            model_xyz_list = []
            model_normal_list = []
            model_rgb_list = []
            num_models = len(d.model_mesh_paths)
            for i in range(num_models):
                points, normals, point_color = load_model_points(d.model_mesh_paths[i],
                                                                 d.model_scale,
                                                                 max_num_points=self.cfg.DATA.POINTNET.REND.MAX_NUM_POINTS,
                                                                 model_texture_path=d.model_texture_paths[i])
                model_xyz_list.append(torch.FloatTensor(points))
                model_normal_list.append(torch.FloatTensor(normals))
                model_rgb_list.append(torch.FloatTensor(point_color)[:, [2,1,0]]) # RGB to BGR

            num_points_keep = min([pc.shape[0] for pc in model_xyz_list[1:]])
            for i in range(1, num_models):
                keep_idx = np.arange(model_xyz_list[i].shape[0])
                np.random.shuffle(keep_idx)
                keep_idx = keep_idx[:num_points_keep]
                model_xyz_list[i] = model_xyz_list[i][keep_idx]
                model_normal_list[i] = model_normal_list[i][keep_idx]
                model_rgb_list[i] = model_rgb_list[i][keep_idx]

            model_xyz_list = [p.cuda() for p in model_xyz_list]
            model_normal_list = [p.cuda() for p in model_normal_list]
            model_rgb_list = [p.cuda() for p in model_rgb_list]

            return model_xyz_list, model_normal_list, model_rgb_list
        else:
            return [], [], []

    def clean_renderer(self, type='default'):
        if type == 'default':
            d = self._dataset
        elif type == 'train':
            d = self._dataset_train
        elif type == 'test':
            d = self._dataset_test
        self.renderer.clean()
        self.renderer = None
        d.renderer = None

    def build_dataloader(self):
        """
        build self.train_loader and self.val_loader based on dataset and dataset_test
        :return:
        """

        def collate_fn(batch):
            return tuple(zip(*batch))

        optimizer_cfg = self._optimizer_cfg
        if self.at_train():
            num_workers = self.cfg.DATA.NUM_WORKERS
            self.train_loader = torch.utils.data.DataLoader(self._dataset, batch_size=optimizer_cfg.IMS_PER_BATCH,
                                                     num_workers=num_workers, collate_fn=collate_fn, drop_last=True)

        if self._phase=='REF' and self._test_cfg.TRACKING:
            self.test_loader = torch.utils.data.DataLoader(self._dataset_test, batch_size=1,
                                                           num_workers=0, collate_fn=collate_fn, drop_last=True)
        else:
            self.test_loader = torch.utils.data.DataLoader(self._dataset_test, batch_size=self._test_cfg.IMS_PER_BATCH,
                                                           num_workers=0, collate_fn=collate_fn, drop_last=True)

    def load_network(self, network_data=None):
        """
        load networks and pretrained weights into self._dcm_refine
        :return:
        """
        network_cfg = self._network_cfg
        if network_data is None:
            if network_cfg.PRETRAINED:
                network_data = torch.load(network_cfg.PRETRAINED, map_location=torch.device('cpu'))
                print_and_log("=> using pre-trained network_ref '{}'".format(network_cfg.PRETRAINED))
            else:
                print_and_log("=> creating network_ref '{}'".format(network_cfg.NAME))
        else:
            print_and_log("=> using given network_ref")
        network = networks.__dict__[network_cfg.NAME](self._dataset.num_classes, network_data).cuda()
        network_data = None  # free it

        if self._phase == 'REF':
            self._dcm_refine = torch.nn.DataParallel(network, device_ids=self.cfg.GPU_ID).cuda()
            self._dcm = self._dcm_refine
        else:
            self._dcm_det = torch.nn.DataParallel(network, device_ids=self.cfg.GPU_ID).cuda()
            if 'grid_centers' in network.__dict__:
                self._grid_centers = network.grid_centers
                self._grid_boxes = network.grid_boxes
            self._dcm = self._dcm_det

    def build_optimizer_and_scheduler(self):
        """
        build self._optimizer and self._scheduler
        :return:
        """
        # build optimizer and lr_scheduler
        optimizer_config = self._optimizer_cfg
        param_groups = [p for p in self._dcm.parameters() if p.requires_grad]
        # param_groups = self._dcm_refine.parameters()
        if optimizer_config.SOLVER == 'ADAM':
            optimizer = torch.optim.Adam(param_groups, optimizer_config.LEARNING_RATE,
                                         betas=(optimizer_config.MOMENTUM, optimizer_config.BETA),
                                         weight_decay=optimizer_config.WEIGHT_DECAY)
        elif optimizer_config.SOLVER == 'SGD':
            optimizer = torch.optim.SGD(param_groups, optimizer_config.LEARNING_RATE,
                                        momentum=optimizer_config.MOMENTUM,
                                        weight_decay=optimizer_config.WEIGHT_DECAY)
        elif optimizer_config.SOLVER == 'Ranger':
            from ranger import Ranger
            optimizer = Ranger(param_groups, optimizer_config.LEARNING_RATE,
                               betas=(optimizer_config.MOMENTUM, optimizer_config.BETA),
                               weight_decay=optimizer_config.WEIGHT_DECAY)
        self._optimizer = optimizer

        print('start iter: ', self._args.startiter)
        #TODO check startiter
        if optimizer_config.MILESTONES_IMG[0]!=-1:
            milestones_img =[int(x) for x in optimizer_config.MILESTONES_IMG]
        else:
            milestones_img = [int(x * self._dataset_train.__len__()) for x in optimizer_config.MILESTONES_EPOCH]
        milestones = [x // optimizer_config.IMS_PER_BATCH for x in milestones_img]
        if optimizer_config.LR_SCHEDULER == 'WARMUP_MULTISTEP':
            warmup_iters = optimizer_config.WARMUP_ITERS_IMG // optimizer_config.IMS_PER_BATCH
            warmup_lr = optimizer_config.LEARNING_RATE / 10
            self._scheduler = WarmUpMultiStepLR(self._optimizer, milestones=milestones, gamma=optimizer_config.GAMMA,
                                          last_epoch=self._args.startiter, warmup_lr=warmup_lr,
                                          warmup_epoch=warmup_iters)
        elif optimizer_config.LR_SCHEDULER == 'FlatCosine':
            milestones = [x // optimizer_config.IMS_PER_BATCH for x in milestones_img]
            assert len(milestones)<=1
            warmup_iters = optimizer_config.WARMUP_ITERS_IMG // optimizer_config.IMS_PER_BATCH
            warmup_lr = optimizer_config.LEARNING_RATE / 10
            T_max = optimizer_config.NUM_EPOCH*optimizer_config.IMGS_EACH_EPOCH // optimizer_config.IMS_PER_BATCH
            self._scheduler = FlatCosineAnnealingLR(self._optimizer,
                                                    T_max=T_max,
                                                    last_epoch=self._args.startiter,
                                                    flat_epochs=milestones[0],
                                                    warmup_lr=warmup_lr,
                                                    warmup_epoch=warmup_iters)

    def sample_poses_refine(self, instance_target):
        """
        generate a pose_src based on the method INIT_POSE_TYPE defined
        :param pose_tgt: tensor 1x7 float32 cuda [quat, trans]
        :return: pose_src: tensor 1x7 float32 cuda [quat, trans]
        """
        pose_tgt = instance_target['pose']
        pose_src = torch.zeros_like(pose_tgt)
        #TODO multiple objects
        assert(pose_tgt.shape[0]==1)
        i = 0
        if self.at_train():
            init_pose_type = self.cfg.TRAIN.REF.INIT_POSE_TYPE
        elif self.at_val():
            init_pose_type = self.cfg.VAL.INIT_POSE_TYPE
        elif self.at_test():
            init_pose_type = self.cfg.TEST.REF.INIT_POSE_TYPE
        else:
            raise Exception("Unknown mode {}".format(self._mode))

        if self.tracking:
            pose_src[i] = pose_tgt[i]
            return pose_src

        # rot
        if init_pose_type == 'fixed':
            euler = self.cfg.TRAIN.REF.INIT_FIXED_POSE if (self.at_train() or self.at_val()) else self.cfg.TEST.REF.INIT_FIXED_POSE
        elif init_pose_type == 'random':
            euler = quat2euler(get_random_rotation())
        elif init_pose_type == 'posecnn' and 'pose_posecnn' in instance_target:
            pose_external = instance_target['pose_posecnn'][i].cpu().numpy()
            euler = quat2euler(pose_external[:4])
        elif init_pose_type == 'cdpn' and 'pose_cdpn' in instance_target:
            pose_external = instance_target['pose_cdpn'][i].cpu().numpy()
            euler = quat2euler(pose_external[:4])
        else:
            #TODO notice the user when use close by default
            euler = quat2euler(pose_tgt[i, :4])
            if cfg.MODE == 'TRAIN':
                euler += self.cfg.TRAIN.REF.INIT_STD_ROTATION * np.random.randn(3) * pi / 180.0
            else:
                euler += self.cfg.TEST.REF.INIT_STD_ROTATION * np.random.randn(3) * pi / 180.0
        pose_src[i, :4] = torch.tensor(euler2quat(euler[0], euler[1], euler[2]))

        # trans

        if init_pose_type == 'posecnn' and 'pose_posecnn' in instance_target:
            pose_src[i, 4:] = instance_target['pose_posecnn'][i, 4:]
        elif init_pose_type == 'cdpn' and 'pose_cdpn' in instance_target:
            pose_src[i, 4:] = instance_target['pose_cdpn'][i, 4:]
        else:
            if cfg.MODE == 'TRAIN':
                std_trans = self.cfg.TRAIN.REF.INIT_SYN_STD_TRANSLATION
            else:
                std_trans = self.cfg.TEST.REF.INIT_SYN_STD_TRANSLATION
            pose_src[i, 4] = pose_tgt[i, 4] + std_trans[0] * np.random.randn()
            pose_src[i, 5] = pose_tgt[i, 5] + std_trans[1] * np.random.randn()
            pose_src[i, 6] = pose_tgt[i, 6] + std_trans[2] * np.random.randn()

        return pose_src

    # def sample_poses_det(self, instance_target):

    def prepare_data_for_refine(self, data_cuda, targets):
        """
        prepare the training data for REF, using heuristic method in DeepIM (train) or PoseCNN result (test) to generate init pose
        :param
            data_cuda: list of dict for each image
                    image: tensor HxWx3, cuda, float32
                    depth: tensor HxW, cuda, float32
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
        :return
            data_inputs: list of dict for each instance
                    image: tensor HxWx3, cuda, float32
                    depth: tensor HxW, cuda, float32
                    K: tensor 3x3, cuda, float32
                    class_idx: int
                    pose_rend: tensor 1x7, cuda, float32
            targets_det: list of dict for each instance
                    box: tensor 1x4, cuda, float32
                    class_idx: int
                    class_id: int
                    mask: tensor 1xHxW, cuda, float32
                    pose: tensor 1x7, cuda, float32
                    # ---- optional ----
                    pose_external: tensor 1x7, cuda, float32
                    pose_icp: tensor 1x7, cuda, float32
        """
        data_ref = []
        targets_ref = []
        for image_idx in range(len(data_cuda)):
            image_data_cuda = data_cuda[image_idx]
            image_targets = targets[image_idx]
            num_objects = image_targets['class_idx'].shape[0]
            for instance_idx in range(num_objects):
                #TODO deal with situation when the number of instance in the data_cuda is not static
                instance_data_cuda = {}
                instance_data_cuda['image'] = image_data_cuda['image']
                instance_data_cuda['depth'] = image_data_cuda['depth']
                if 'pcloud' in image_data_cuda:
                    instance_data_cuda['pcloud'] = image_data_cuda['pcloud']
                instance_data_cuda['K'] = image_data_cuda['K']
                # use gt label, can be replaced with
                instance_class_idx = image_targets['class_idx'][instance_idx].item()
                instance_data_cuda['class_idx'] = instance_class_idx

                instance_target = {}
                instance_target['box'] = image_targets['boxes'][[instance_idx]]
                instance_target['class_id'] = image_targets['class_id'][instance_idx].item()
                instance_target['class_idx'] = instance_class_idx  # int
                instance_target['mask'] = image_targets['masks'][[instance_idx]]  # tensor 1xHxW
                instance_target['pose'] = image_targets['poses'][[instance_idx]]  # tensor 1x7
                instance_target['complete_box'] = image_targets['complete_boxes'][[instance_idx]]

                if 'posecnn_results' in image_targets:
                    instance_target['pose_posecnn'] = image_targets['posecnn_results'][[instance_idx]]
                if 'cdpn_results' in image_targets:
                    instance_target['pose_cdpn'] = image_targets['cdpn_results'][[instance_idx]]
                if 'icp_results' in image_targets:
                    instance_target['pose_icp'] = image_targets['icp_results'][[instance_idx]]

                # print(instance_target['pose'].shape, image_targets['poses'].shape)
                if self._phase == 'DET' or self._phase == 'E2E':
                    pose_init = self.sample_poses_det(instance_target)
                elif self._phase == 'REF':
                    pose_init = self.sample_poses_refine(instance_target) # tensor 1x7
                else:
                    raise KeyError
                instance_data_cuda['pose_rend'] = pose_init
                instance_target['pose_init'] = pose_init

                data_ref.append(instance_data_cuda)
                targets_ref.append(instance_target)

        return data_ref, targets_ref

    def get_closest_pose(self, pose_src, class_idx, pose_gt):
        """

        :param pose_src: tensor cuda, [7,]
        :param class_idx: int
        :param pose_gt: tensor cuda, [7,]
        :return:
            pose_target: tensor cuda, [7,]
        """
        pose_target = pose_gt.clone()
        pose_src = pose_src.cpu().numpy()
        pose_gt = pose_gt.cpu().numpy()

        rot_src = quat2mat(pose_src[:4])
        rot_gt = quat2mat(pose_gt[:4])
        sym_info = self._sym_info[class_idx]
        rot_target = get_closest_pose(rot_src, rot_gt, sym_info)

        pose_target[:4] = torch.tensor(mat2quat(rot_target))
        return pose_target

    def _dynamic_zoom_in(self, data, target, max_num_points):
        height, width = data['depth'].shape
        x1, y1, x2, y2 = target['box'][0]
        cx = (x1+x2)/2
        cy = (y1+y2)/2
        w_obj = x2-x1
        h_obj = y2-y1
        scale_ratio = self.cfg.DATA.POINTNET.OBS.DZI_SCALE_RATIO
        shift_ratio = self.cfg.DATA.POINTNET.OBS.DZI_SHIFT_RATIO
        basic_ratio = self.cfg.DATA.POINTNET.OBS.DZI_BASIC_RATIO
        if self.cfg.DATA.POINTNET.OBS.DZI:
            cx_dzi = cx+w_obj*shift_ratio*np.random.randn()
            cy_dzi = cy+h_obj*shift_ratio*np.random.randn()
            w_dzi = torch.clamp(w_obj*(basic_ratio+scale_ratio*np.random.randn()), min=200)
            h_dzi = torch.clamp(h_obj*(basic_ratio+scale_ratio*np.random.randn()), min=200)
            x1_dzi = torch.clamp(cx_dzi-w_dzi/2, min=0, max=width).round().int().item()
            y1_dzi = torch.clamp(cy_dzi-h_dzi/2, min=0, max=height).round().int().item()
            x2_dzi = torch.clamp(cx_dzi+w_dzi/2, min=0, max=width).round().int().item()
            y2_dzi = torch.clamp(cy_dzi+h_dzi/2, min=0, max=height).round().int().item()
            if x2_dzi-x1_dzi < 5 or y2_dzi-y1_dzi < 5:
                plt.imshow(data['image'].cpu().numpy())
                plt.show()
                print_and_log('old: {}, new: {}'.format([x1.item(), y1.item(), x2.item(), y2.item()],
                                                [x1_dzi, y1_dzi, x2_dzi, y2_dzi]))
                x1_dzi = 0
                y1_dzi = 0
                x2_dzi = width
                y2_dzi = height
        else:
            x1_dzi = 0
            y1_dzi = 0
            x2_dzi = width
            y2_dzi = height

        zoom_map = torch.zeros_like(data['pcloud'][:, :, 0])
        zoom_map[y1_dzi:y2_dzi, x1_dzi:x2_dzi] = 1
        valid_map = data['pcloud'][:, :, -1] > 0.1
        select_map = ((valid_map*zoom_map)>0).flatten()
        if torch.sum(select_map) < max_num_points and cfg.TRAIN.DET.IMS_PER_BATCH > 1:
            select_map = valid_map.flatten()
        select_index = torch.nonzero(select_map)[:, 0]
        if len(select_index) > max_num_points and max_num_points>0:
            shuffle_index = torch.randperm(len(select_index))[:max_num_points]
            shuffle_index = shuffle_index.sort()[0]
            select_index = select_index[shuffle_index]
            select_map = torch.zeros_like(select_map)
            select_map[select_index] = 1
            select_map = select_map>0.1

        data['points_valid_map'] = select_map
        valid_index = torch.nonzero(select_map)
        data['points_valid_location'] = torch.cat([valid_index%width, valid_index/width], dim=1) # x,y
        data['points_xyz_obs'] = data['pcloud'].reshape([-1, 3])[select_map, :].permute(
            [1, 0]).contiguous()
        data['points_rgb_obs'] = data['image'].reshape([-1, 3])[select_map, :].permute(
            [1, 0]).contiguous()
        target['label_same_object'] = target['mask'].reshape([-1, 1])[select_map, :]
        target['3d_center_offset'] = target['pose'][0, 4:].reshape([3, 1]) - data['points_xyz_obs']
        target['2d_center_offset'] = target['3d_center_offset'][:2, :] / data['points_xyz_obs'][[2], :] * 100
        target['3d_direction'] = F.normalize(target['3d_center_offset'], dim=0)
        target['2d_direction'] = F.normalize(target['2d_center_offset'], dim=0)

    def update_data_refine(self, data_ref, targets_ref, preds):
        """
        update rend images and zoom the images
        :param
            data_inputs: list of dict for each instance
                    image: tensor HxWx3, cuda, float32
                    depth: tensor HxW, cuda, float32
                    K: tensor 3x3, cuda, float32
                    class_idx: int
                    pose_rend: tensor 1x7, cuda, float32, q4t3
            targets_det: list of dict for each instance
                    box: tensor 1x4, cuda, float32
                    class_idx: int
                    class_id: int
                    mask: tensor 1xHxW, cuda, float32
                    pose: tensor 1x7, cuda, float32
                    # ---- optional ----
                    pose_external: tensor 1x7, cuda, float32
                    pose_icp: tensor 1x7, cuda, float32
        :return:
            data_inputs: list of dict for each instance
                    image torch.Size([480, 640, 3]) cuda:0
                    depth torch.Size([480, 640]) cuda:0
                    pcloud torch.Size([480, 640, 3]) cuda:0
                    K torch.Size([3, 3]) cuda:0
                    class_idx <class 'int'>
                    pose_rend torch.Size([1, 7]) cuda:0
                    image_rend torch.Size([480, 640, 3]) cuda:0
                    mask_rend torch.Size([480, 640]) cuda:0
                    depth_rend torch.Size([480, 640]) cuda:0
                    pcloud_rend torch.Size([480, 640, 3]) cuda:0
                    box_rend torch.Size([4]) cpu
                    zoom_factor torch.Size([1, 4]) cpu
                    zoom_K <class 'numpy.ndarray'>
                    zoom_image_obs torch.Size([1, 3, 480, 640]) cuda:0
                    zoom_depth_obs torch.Size([1, 1, 480, 640]) cuda:0
                    zoom_pcloud_obs torch.Size([1, 3, 480, 640]) cuda:0
                    zoom_image_rend torch.Size([1, 3, 480, 640]) cuda:0
                    zoom_depth_rend torch.Size([1, 1, 480, 640]) cuda:0
                    zoom_pcloud_rend torch.Size([1, 3, 480, 640]) cuda:0
            targets_det: list of dict for each instance
                    box torch.Size([1, 4]) cuda:0
                    class_id <class 'int'>
                    class_idx <class 'int'>
                    mask torch.Size([1, 480, 640]) cuda:0
                    pose torch.Size([1, 7]) cuda:0
                    complete_box torch.Size([1, 4]) cuda:0
                    pose_icp torch.Size([1, 7]) cuda:0
                    pose_init torch.Size([1, 7]) cuda:0
                    zoom_mask torch.Size([1, 1, 480, 640]) cuda:0
                    pose_target torch.Size([1, 7]) cuda:0

        """
        num_samples = len(data_ref)
        remove_idx = []
        for sample_idx in range(num_samples):
            data = data_ref[sample_idx]
            target = targets_ref[sample_idx]

            # render reference image and depth
            if preds:
                pose_rend = matPose2quatPose(
                    preds['poses_pred'][sample_idx].detach().cpu().numpy())
                data_ref[sample_idx]['pose_rend'] = torch.FloatTensor(pose_rend).cuda().unsqueeze(0)
            data_rend = self.rend_one_image([data['pose_rend'].cpu().numpy().flatten()],
                                            [target['class_id']],
                                            data['K'].cpu().numpy())
            data_ref[sample_idx].update(data_rend)

            # zoom images
            affine_matrix, bbox = self.get_zoom_factor(data)
            data_ref[sample_idx]['box_rend'] = torch.FloatTensor(bbox)
            image_obs = data['image'].permute(2,0,1).unsqueeze(0)
            image_rend = data['image_rend'].permute(2,0,1).unsqueeze(0)

            data_ref[sample_idx]['zoom_factor'] = torch.tensor([[affine_matrix[0, 0],
                                                                 affine_matrix[1, 1],
                                                                 affine_matrix[0, 2],
                                                                 affine_matrix[1, 2]]], dtype=torch.float32)

            data_ref[sample_idx]['zoom_K'] = self.get_zoomed_K(data['K'],
                                                               data['zoom_factor'],
                                                               [data['image'].size(0), data['image'].size(1)],
                                                               [self.cfg.DATA.INPUT_HEIGHT, self.cfg.DATA.INPUT_WIDTH])

            #TODO: zoom mask
            theta = torch.FloatTensor(affine_matrix).cuda().unsqueeze(0)
            zoomed_size = list(image_obs.size())
            zoomed_size[2] = self.cfg.DATA.INPUT_HEIGHT
            zoomed_size[3] = self.cfg.DATA.INPUT_WIDTH
            grids = nn.functional.affine_grid(theta, zoomed_size)

            zoom_image_obs = F.grid_sample(image_obs, grids)
            zoom_image_obs -= self._pixel_mean
            zoom_image_obs /= self._pixel_std
            data_ref[sample_idx]['zoom_image_obs'] = zoom_image_obs
            zoom_depth_obs = data_ref[sample_idx]['depth'].unsqueeze(0).unsqueeze(0)
            data_ref[sample_idx]['zoom_depth_obs'] = F.grid_sample(zoom_depth_obs, grids)
            zoom_pcloud_obs = data_ref[sample_idx]['pcloud'].permute([2,0,1]).unsqueeze(0)
            data_ref[sample_idx]['zoom_pcloud_obs'] = F.grid_sample(zoom_pcloud_obs, grids)
            mask_obs = target['mask'].unsqueeze(0)
            targets_ref[sample_idx]['zoom_mask'] = (F.grid_sample(mask_obs, grids)>0.1).type(torch.cuda.FloatTensor)

            if self.cfg.DATA.HIGH_RES_REND:
                zoom_size = [self.cfg.DATA.INPUT_HEIGHT, self.cfg.DATA.INPUT_WIDTH]
                zoom_data_rend = self.rend_one_image([data['pose_rend'].cpu().numpy().flatten()],
                                                     [target['class_id']],
                                                     data['zoom_K'].cpu().numpy(),
                                                     size=zoom_size)
                zoom_image_rend = zoom_data_rend['image_rend'].permute(2,0,1).unsqueeze(0)
                zoom_image_rend = F.interpolate(zoom_image_rend, zoom_size)
                zoom_depth_rend = zoom_data_rend['depth_rend'].unsqueeze(0).unsqueeze(0)
                zoom_depth_rend = F.interpolate(zoom_depth_rend, zoom_size)
                zoom_pcloud_rend = zoom_data_rend['pcloud_rend'].unsqueeze(0).unsqueeze(0)
                zoom_pcloud_rend = F.interpolate(zoom_pcloud_rend, zoom_size)
            else:
                zoom_image_rend = F.grid_sample(image_rend, grids)
                zoom_depth_rend = data_ref[sample_idx]['depth_rend'].unsqueeze(0).unsqueeze(0)
                zoom_depth_rend = F.grid_sample(zoom_depth_rend, grids)
                zoom_pcloud_rend = data_ref[sample_idx]['pcloud_rend'].permute(2,0,1).unsqueeze(0)
                zoom_pcloud_rend = F.grid_sample(zoom_pcloud_rend, grids)
            zoom_image_rend -= self._pixel_mean
            zoom_image_rend /= self._pixel_std
            data_ref[sample_idx]['zoom_image_rend'] = zoom_image_rend
            data_ref[sample_idx]['zoom_depth_rend'] = zoom_depth_rend
            data_ref[sample_idx]['zoom_pcloud_rend'] = zoom_pcloud_rend

            if self._network_cfg.NORM_INPUT_DEPTH == 'z':
                cz = data_ref[sample_idx]['pose_rend'][0, -1]
                mask_obs = (data_ref[sample_idx]['zoom_depth_obs']>0.1).float()
                mask_rend = (data_ref[sample_idx]['zoom_depth_rend']>0.1).float()
                data_ref[sample_idx]['zoom_depth_obs'] -= cz*mask_obs
                data_ref[sample_idx]['zoom_depth_rend'] -= cz*mask_rend
                data_ref[sample_idx]['zoom_pcloud_obs'][:,:,-1] -= cz * mask_obs
                data_ref[sample_idx]['zoom_pcloud_rend'][:,:,-1] -= cz * mask_rend
            elif self._network_cfg.NORM_INPUT_DEPTH == 'pcloud':
                cz = data_ref[sample_idx]['pose_rend'][0, -3:].reshape([3,1,1])
                mask_obs = (data_ref[sample_idx]['zoom_depth_obs']>0.1).float()
                mask_rend = (data_ref[sample_idx]['zoom_depth_rend']>0.1).float()
                data_ref[sample_idx]['zoom_pcloud_obs'] -= cz * mask_obs
                data_ref[sample_idx]['zoom_pcloud_rend'] -= cz * mask_rend

            targets_ref[sample_idx]['pose_target'] = self.get_closest_pose(data['pose_rend'][0],
                                                                           data['class_idx'],
                                                                           target['pose'][0]).unsqueeze(0)

            if self.cfg.LOSS.REF.LW_DCN_LOSS>0  and self.at_train():
                targets_ref[sample_idx]['DCN_data'], targets_ref[sample_idx]['DCN_debug_data']\
                    = self.dcn_module.get_pixel_correspondences(data_ref[sample_idx],
                                                                targets_ref[sample_idx])

                if targets_ref[sample_idx]['DCN_data']['matches_a'] is None:
                    print_and_log("remove a no-matches sample from training")
                    remove_idx.append(sample_idx)

                if self.cfg.DCN.debug:
                    self.visualize_dcn(data_ref[sample_idx], targets_ref[sample_idx])

        for sample_idx in remove_idx[::-1]:
            data_ref.pop(sample_idx)
            targets_ref.pop(sample_idx)

    @staticmethod
    def get_zoomed_K(K, zoom_factor, input_size, output_size):
        """

            data_inputs[sample_idx]['zoom_factor'] = torch.tensor([[affine_matrix[0, 0],
                                                                 affine_matrix[1, 1],
                                                                 affine_matrix[0, 2],
                                                                 affine_matrix[1, 2]]])
        :param K:
        :param zoom_factor:
        :param height:
        :param width:
        :return:
        """
        height_i, width_i = input_size
        height_o, width_o = output_size
        K_np = K.cpu().numpy()
        zoom_factor_np = zoom_factor.cpu().numpy()
        fx = K_np[0, 0]
        fy = K_np[1, 1]
        cx = K_np[0, 2]
        cy = K_np[1, 2]
        zoom_scale = zoom_factor_np[0, 0]
        zoom_cx = (zoom_factor_np[0, 2] + 1) / 2. * width_i
        zoom_cy = (zoom_factor_np[0, 3] + 1) / 2. * height_i
        fx_zoomed = fx / zoom_scale * width_o / width_i
        fy_zoomed = fy / zoom_scale * height_o / height_i
        cx_zoomed = ((cx - zoom_cx) / zoom_scale + width_i / 2) * width_o / width_i
        cy_zoomed = ((cy - zoom_cy) / zoom_scale + height_i / 2) * height_o / height_i
        # print('zoom_cx', zoom_cx, 'zoom_cy', zoom_cy)
        K_zoomed = np.array([[fx_zoomed, 0, cx_zoomed], [0, fy_zoomed, cy_zoomed], [0, 0, 1.]])
        K_zoomed = torch.FloatTensor(K_zoomed).to(K.device)
        return K_zoomed

    def rend_one_image(self, poses, class_ids, K, size=None):
        """
        :param pose: ndarray, [7, ]
        :param class_id: [int]
        :param K: ndarray, 3x3
        :return:
            data_rend: dict for the rendered image
                image_rend: tensor, HxWx3, cuda, float32
                depth: tensor, HxW, cuda, float32
                mask_cuda: tensor, HxW, cuda, float32
        """
        if size is None:
            height = self.cfg.SYN.HEIGHT
            width = self.cfg.SYN.WIDTH
        else:
            height, width = size
        syn_width = self.cfg.SYN.WIDTH
        syn_height = self.cfg.SYN.HEIGHT
        zfar = 6.0
        znear = 0.01

        self.renderer.set_light_pos([0, 0, 0])
        self.renderer.set_light_color([1, 1, 1])
        rend_pose = [pose[[4,5,6,0,1,2,3]] for pose in poses]
        self.renderer.set_poses(rend_pose)
        self.renderer.set_projection_matrix(width, height, K[0, 0], K[1, 1],
                                           K[0, 2], K[1, 2], znear, zfar)

        image_tensor = torch.cuda.FloatTensor(syn_height, syn_width, 4).detach()
        seg_tensor = torch.cuda.FloatTensor(syn_height, syn_width, 4).detach()
        pc_tensor = torch.cuda.FloatTensor(syn_height, syn_width, 4).detach()
        self.renderer.render(class_ids, image_tensor, seg_tensor, pc2_tensor=pc_tensor)
        im_cuda = image_tensor.flip(0)[:, :, [2,1,0]].cuda() # RGB2BGR
        mask_cuda = (seg_tensor.flip(0)[:, :, 0] != 0).float().cuda()
        pcloud_cuda = pc_tensor.flip(0)[:,:,:3].cuda()
        depth_cuda = pcloud_cuda[:,:,2]
        data_rend = {'image_rend': im_cuda,
                     'mask_rend': mask_cuda,
                     'depth_rend': depth_cuda,
                     'pcloud_rend': pcloud_cuda}
        return data_rend

    def get_zoom_factor(self, data):
        """
        get zoom_factor, assuming only one object (G=1)
        :param data:
            data_inputs: list of dict for each instance
                    image: tensor HxWx3, cuda, float32
                    K: tensor 3x3, cuda, float32
                    class_idx: list (1,)
                    pose_rend: tensor 1x7, cuda, float32
        :return:
            affine_matrix: np.ndarray, 2x3, float64
            bbox: list, (4,), float32
        """
        # pose_mat: tensor 3x4, cuda, float32
        pose_mat = qtPose2matPose_tensor(data['pose_rend'][0])
        # points: tensor 4x2600, cuda, float32
        points = self._points[data['class_idx']]
        # intrinsic_matrix: tensor 3x3, cuda, float32
        intrinsic_matrix = data['K']
        # TODO: not consider situation when cfg.DATA.INPUT_HEIGHT/WIDTH not consist with cfg.SYN
        height = data['image'].size(0)
        width = data['image'].size(1)
        zoom_height = self.cfg.DATA.INPUT_HEIGHT
        zoom_width = self.cfg.DATA.INPUT_WIDTH

        ratio = zoom_height / zoom_width

        # center
        obj_imgn_c = torch.matmul(intrinsic_matrix, pose_mat[:, [-1]])
        zoom_c_x = (obj_imgn_c[0] / obj_imgn_c[2]).item()
        zoom_c_y = (obj_imgn_c[1] / obj_imgn_c[2]).item()

        # x2d: tensor 2x2600
        x2d = torch.matmul(intrinsic_matrix, torch.matmul(pose_mat, points))
        x2d[0, :] /= x2d[2, :]
        x2d[1, :] /= x2d[2, :]
        obj_start_x = torch.min(x2d[0, :]).item()
        obj_start_y = torch.min(x2d[1, :]).item()
        obj_end_x = torch.max(x2d[0, :]).item()
        obj_end_y = torch.max(x2d[1, :]).item()

        left_dist = zoom_c_x - obj_start_x
        right_dist = obj_end_x - zoom_c_x
        up_dist = zoom_c_y - obj_start_y
        down_dist = obj_end_y - zoom_c_y

        crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2.8
        crop_width = crop_height / ratio

        # affine transformation for PyTorch
        x1 = (zoom_c_x - crop_width / 2) * 2 / width - 1
        x2 = (zoom_c_x + crop_width / 2) * 2 / width - 1
        y1 = (zoom_c_y - crop_height / 2) * 2 / height - 1
        y2 = (zoom_c_y + crop_height / 2) * 2 / height - 1

        pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
        pts2 = np.float32([[-1, -1], [-1, 1], [1, -1]])
        affine_matrix = cv2.getAffineTransform(pts2, pts1)
        # scale_factor = (x2-x1+1
        bbox = [obj_start_x, obj_start_y, obj_end_x, obj_end_y]

        if not np.isfinite(bbox).all():
            raise Exception('nan or inf found at pose: ', bbox)

        return affine_matrix, bbox

    @staticmethod
    def get_pose_mat(pose_src, trans, rot_delta_quat=None, rot_delta_mats=None):
        poses_mat = torch.zeros([pose_src.shape[0], 3, 4]).cuda()
        for batch_ind in range(pose_src.shape[0]):
            if rot_delta_quat is not None:
                if np.isfinite(rot_delta_quat[batch_ind].detach().cpu().numpy()).all():
                    rot_src_mat = quat2mat_tensor(pose_src[batch_ind, :4])
                    rot_delta_mat = quat2mat_tensor(rot_delta_quat[batch_ind])
                    poses_mat[batch_ind, :3, :3] = torch.matmul(rot_delta_mat, rot_src_mat)
                else:
                    poses_mat[batch_ind, :3, :3] = torch.eye(3)
            elif rot_delta_mats is not None:
                if np.isfinite(rot_delta_mats[batch_ind].detach().cpu().numpy()).all():
                    rot_src_mat = quat2mat_tensor(pose_src[batch_ind, :4])
                    rot_delta_mat = rot_delta_mats[batch_ind]
                    poses_mat[batch_ind, :3, :3] = torch.matmul(rot_delta_mat, rot_src_mat)
                else:
                    poses_mat[batch_ind, :3, :3] = torch.eye(3)
            else:
                poses_mat[batch_ind, :3, :3] = quat2mat_tensor(pose_src[batch_ind])

            if np.isfinite(trans[batch_ind].detach().cpu().numpy()).all():
                poses_mat[batch_ind, :, 3] = trans[batch_ind]
            else:
                print_and_log("invalid trans: "+str(trans[batch_ind]))
                # fail to converge, can also set to another valid number
                poses_mat[batch_ind, :, 3] = 999
        return poses_mat

    def pml(self, preds, data_ref, targets_ref):
        """

        :param preds: dict of tensor
            quat: tensor Nx4, cuda, float32
            trans, tensor Nx3, cuda, float32
            poses_pred: tensor, Nx3x4, float32, cuda
        :param data_ref:
            omit
        :param targets_ref:
            targets_det: list of dict for each instance
                    box: tensor 1x4, cuda, float32
                    class_idx: int
                    class_id: int
                    mask: tensor 1xHxW, cuda, float32
                    pose: tensor 1x7, cuda, float32
                    pose_target: tensor 1x7, cuda, float32
        :return:
        """
        poses_gt_qt = torch.cat([target['pose_target'] for target in targets_ref])
        points = torch.stack([self._points[data['class_idx']] for data in data_ref], dim=0)

        if self._loss_cfg.PML_NORMALIZE_METHOD == 'extents':
            extents = torch.stack([self._extents[data['class_idx']] for data in data_ref], dim=0)
            norm_term = torch.max(extents, dim=1)[0]
            weights = (self.cfg.LOSS.REF.PML_NORMALIZE_FACTOR / norm_term) \
                .reshape([-1, 1, 1, 1]).detach()
        elif self._loss_cfg.PML_NORMALIZE_METHOD == 'diameter':
            norm_term = torch.stack([self._diameters[data['class_idx']] for data in data_ref], dim=0)
            weights = (self.cfg.LOSS.REF.PML_NORMALIZE_FACTOR / norm_term) \
                .reshape([-1, 1, 1, 1]).detach()
        elif self._loss_cfg.PML_NORMALIZE_METHOD == 'constant':
            norm_term = 0.1
            weights = (self.cfg.LOSS.REF.PML_NORMALIZE_FACTOR / norm_term)

        poses_est = preds['poses_pred']
        poses_gt = self.get_pose_mat(poses_gt_qt[:, :4],
                                     poses_gt_qt[:, 4:])

        # points: Nx4x2600
        # points_est/gt: Nx3x2600
        points_est = torch.matmul(poses_est, points)
        points_gt = torch.matmul(poses_gt, points)

        num_points = points_est.shape[0]*points_est.shape[2]
        loss_pose = F.smooth_l1_loss(points_est * weights, points_gt * weights)*3
        return loss_pose

    def process_dcn_output(self, image_pred):
        """
        Processes the network output into a new shape

        :param image_pred: output of the network img.shape = [N,descriptor_dim, H , W]
        :type image_pred: torch.Tensor
        :param N: batch size
        :type N: int
        :return: same as input, new shape is [N, W*H, descriptor_dim]
        :rtype:
        """
        N = image_pred.size(0)
        descriptor_dimension = image_pred.size(1)
        H = image_pred.size(2)
        W = image_pred.size(3)
        image_pred = image_pred.view(N, descriptor_dimension, W * H)
        image_pred = image_pred.permute(0, 2, 1)
        return image_pred

    def loss_angular_distance(self, quat_est, quat_gt):
        errors_rot = torch.clamp(torch.acos(torch.clamp(2 * (torch.sum(quat_est * quat_gt, axis=1) ** 2) - 1, min=0, max=1))-0.05,min=0.)
        return errors_rot

    def make_loss_refine(self, preds, data_ref, targets_ref):
        """

        :param preds: dict
            quat torch.Size([8, 4]) torch.float32 cuda:0
            trans torch.Size([8, 3]) torch.float32 cuda:0
            mat <class 'NoneType'>
            poses_pred torch.Size([8, 3, 4]) torch.float32 cuda:0
        :param data_ref: list of dict
            image torch.Size([480, 640, 3]) torch.float32 cuda:0
            depth torch.Size([480, 640]) torch.float32 cuda:0
            pcloud torch.Size([480, 640, 3]) torch.float32 cuda:0
            K torch.Size([3, 3]) torch.float32 cuda:0
            class_idx <class 'int'>
            pose_rend torch.Size([1, 7]) torch.float32 cuda:0
            image_rend torch.Size([480, 640, 3]) torch.float32 cuda:0
            mask_rend torch.Size([480, 640]) torch.float32 cuda:0
            depth_rend torch.Size([480, 640]) torch.float32 cuda:0
            pcloud_rend torch.Size([480, 640, 3]) torch.float32 cuda:0
            box_rend torch.Size([4]) torch.float32 cpu
            zoom_factor torch.Size([1, 4]) torch.float32 cpu
            zoom_K <class 'numpy.ndarray'>
            zoom_image_obs torch.Size([1, 3, 480, 640]) torch.float32 cuda:0
            zoom_depth_obs torch.Size([1, 1, 480, 640]) torch.float32 cuda:0
            zoom_pcloud_obs torch.Size([1, 3, 480, 640]) torch.float32 cuda:0
            zoom_image_rend torch.Size([1, 3, 480, 640]) torch.float32 cuda:0
            zoom_depth_rend torch.Size([1, 1, 480, 640]) torch.float32 cuda:0
            zoom_pcloud_rend torch.Size([1, 3, 480, 640]) torch.float32 cuda:0
        :param targets_ref: list of dict
            box torch.Size([1, 4]) torch.float32 cuda:0
            class_id <class 'int'>
            class_idx <class 'int'>
            mask torch.Size([1, 480, 640]) torch.float32 cuda:0
            pose torch.Size([1, 7]) torch.float32 cuda:0
            complete_box torch.Size([1, 4]) torch.float32 cuda:0
            pose_icp torch.Size([1, 7]) torch.float32 cuda:0
            pose_init torch.Size([1, 7]) torch.float32 cuda:0
            zoom_mask torch.Size([1, 1, 480, 640]) torch.float32 cuda:0
            pose_target torch.Size([1, 7]) torch.float32 cuda:0
        :return:
        """
        loss = 0
        loss_dict = {}
        loss_weights = self.cfg.LOSS.REF

        num_samples = len(data_ref)
        # the smoothl1 used in pml already average the loss with the number of samples
        if loss_weights.LW_PML>0:
            loss_pose = self.pml(preds, data_ref, targets_ref)
            loss += loss_pose*loss_weights.LW_PML
            loss_dict['pml'] = loss_pose.item()

        if loss_weights.LW_ROT>0:
            assert preds['quat'] is not None
            poses_gt_qt = torch.cat([target['pose_target'] for target in targets_ref], axis=0)
            loss_rot = self.loss_angular_distance(preds['quat'], poses_gt_qt[:, :4]).sum()/num_samples
            loss += loss_rot * loss_weights.LW_ROT
            loss_dict['loss_rot'] = loss_rot.item()

        if loss_weights.LW_TRANS>0:
            assert preds['trans'] is not None
            poses_gt_qt = torch.cat([target['pose_target'] for target in targets_ref], axis=0)
            loss_trans = F.smooth_l1_loss(preds['trans'], poses_gt_qt[:, 4:], reduction='sum')*1000/num_samples
            loss+=loss_trans*loss_weights.LW_TRANS
            loss_dict['loss_trans'] = loss_trans.item()

        if loss_weights.LW_MASK>0:
            mask_predict = preds['mask']
            mask_gt = torch.cat([x['zoom_mask'] for x in targets_ref]) # Nx1xHxW
            if cfg.LOSS.REF.MASK_LOSS == 'sigmoid':
                loss_mask = F.binary_cross_entropy_with_logits(mask_predict, mask_gt)
            elif cfg.LOSS.REF.MASK_LOSS == 'softmax':
                mask_gt = torch.squeeze(mask_gt, 1)
                mask_gt = mask_gt.type(torch.cuda.LongTensor)
                loss_mask = F.cross_entropy(mask_predict, mask_gt)
            loss_dict['mask'] = loss_mask.item()
            loss += loss_mask*loss_weights.LW_MASK

        return loss, loss_dict

    def merge_pose_preds(self, pose_preds_all, batch_preds):
        for preds in batch_preds:
            if len(pose_preds_all)==0:
                for k in preds.keys():
                    if k != 'x_obs':
                        pose_preds_all[k] = [preds[k].cpu()]
            else:
                for k in pose_preds_all.keys():
                    pose_preds_all[k].append(preds[k].cpu())

    def post_process_refine_preds(self, data_ref, preds):
        """

        :param data_ref:
        :param preds:
        :return:
            poses_pred: tensor, Nx3x4, float32, cuda
        """
        poses_rend_qt = torch.cat([data['pose_rend'] for data in data_ref])
        poses_est = self.get_pose_mat(poses_rend_qt,
                                      preds['trans'],
                                      rot_delta_quat=preds['quat'],
                                      rot_delta_mats=preds['mat'])
        preds['poses_pred'] = poses_est

    def merge_data(self, data_ref, selected_keys=[]):
        """

        :param data_ref:
            data_inputs: list of dict for each instance
                    image torch.Size([480, 640, 3]) cuda:0
                    depth torch.Size([480, 640]) cuda:0
                    pcloud torch.Size([480, 640, 3]) cuda:0
                    K torch.Size([3, 3]) cuda:0
                    class_idx <class 'int'>
                    pose_rend torch.Size([1, 7]) cuda:0
                    image_rend torch.Size([480, 640, 3]) cuda:0
                    mask_rend torch.Size([480, 640]) cuda:0
                    depth_rend torch.Size([480, 640]) cuda:0
                    pcloud_rend torch.Size([480, 640, 3]) cuda:0
                    box_rend torch.Size([4]) cpu
                    zoom_factor torch.Size([1, 4]) cpu
                    zoom_K <class 'numpy.ndarray'>
                    zoom_image_obs torch.Size([1, 3, 480, 640]) cuda:0
                    zoom_depth_obs torch.Size([1, 1, 480, 640]) cuda:0
                    zoom_pcloud_obs torch.Size([1, 3, 480, 640]) cuda:0
                    zoom_image_rend torch.Size([1, 3, 480, 640]) cuda:0
                    zoom_depth_rend torch.Size([1, 1, 480, 640]) cuda:0
                    zoom_pcloud_rend torch.Size([1, 3, 480, 640]) cuda:0
        :return:
        """
        inputs = {}
        if len(selected_keys)==0:
            selected_keys = data_ref[0].keys()
        for k in selected_keys:
            if k not in data_ref[0]:
                raise KeyError
            if k in ['box_rend',
                     'K',
                     'points_valid_map',
                     'points_xyz_obs',
                     'points_rgb_obs',
                     'points_xyz_rend',
                     'points_normal_rend',
                     'points_rgb_rend',
                     'points_label']:
                inputs[k] = torch.stack([data[k] for data in data_ref], dim=0).detach()
            elif k == 'class_idx':
                inputs[k] = torch.IntTensor([data[k] for data in data_ref]).detach()
            elif k in ['zoom_image_obs',
                       'zoom_image_rend',
                       'zoom_depth_obs',
                       'zoom_depth_rend',
                       'zoom_pcloud_obs',
                       'zoom_pcloud_rend',
                       'zoom_factor',
                       'pose_rend',
                       'x_obs',
                       'pointnet']:
                inputs[k] = torch.cat([data[k] for data in data_ref], dim=0).detach()
        return inputs

    def train_refine(self, i_epoch=0):
        """

        batch_preds: dict of tensor
            quat: tensor Nx4, cuda, float32
            trans, tensor Nx3, cuda, float32
            poses_est: tensor, Nx3x4, float32, cuda
        :return:
        """
        # torch.cuda.get_device_properties(0).total_memory
        self._dcm_refine.train()
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter('loss', utils.SmoothedValue(window_size=self.cfg.TRAIN.FREQUENCY, fmt='{value:.6f}'))

        optimizer_config = self.cfg.TRAIN.USING
        header = 'Epoch: [{}]'.format(i_epoch)
        for data_cuda, targets in metric_logger.log_every(self.train_loader,
                                                          self.cfg.TRAIN.FREQUENCY,
                                                          header):
            self.iter_train += optimizer_config.IMS_PER_BATCH
            metric_logger.update(lr=self._optimizer.param_groups[0]["lr"])

            data_cuda = [{k: v.cuda() for k, v in t.items()} for t in data_cuda]
            targets = [{k: v.cuda() for k, v in t.items()} for t in targets]

            # train REF
            data_ref, targets_ref = self.prepare_data_for_refine(data_cuda, targets)
            num_iter_refine = self.get_num_iters_refine()
            preds = None
            for i_refine in range(num_iter_refine):
                self.update_data_refine(data_ref, targets_ref, preds)
                if self.cfg.TRAIN.VISUALIZE:
                    for i in range(len(data_ref)):
                        self.visualize(data_ref[i], targets_ref[i])
                        plt.show()
                inputs = self.merge_data(data_ref)
                preds = self._dcm_refine(**inputs)
                self.post_process_refine_preds(data_ref, preds)
                loss, loss_dict = self.make_loss_refine(preds, data_ref, targets_ref)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

                for k in loss_dict:
                    self.tf_logger.add_scalar("logs_s/losses/{}".format(k),
                                             loss_dict[k],
                                             self.iter_train)

            metric_logger.update(loss=loss.item(), lr=self._scheduler.get_lr()[0])
            self._scheduler.step()

            if self.iter_train % optimizer_config.SNAPSHOT_ITERS_IMG \
                < optimizer_config.IMS_PER_BATCH:
                filename = 'checkpoint_{:08}'.format(self.iter_train)
                checkpoint_path = os.path.join(optimizer_config.OUTPUT_DIR, filename)
                torch.save(self._dcm_refine.module.state_dict(), checkpoint_path)

    def clean_buffer(self):
        del self._dataset

    def finish_train(self):
        filename = self._phase.lower()+'_model_final.pth'
        final_path = os.path.join(self.cfg.TRAIN.USING.OUTPUT_DIR, filename)
        if self._phase == 'REF':
            torch.save(self._dcm_refine.module.state_dict(), final_path)
        else:
            torch.save(self._dcm_det.module.state_dict(), final_path)
        print_and_log("final model saved at "+final_path)

    def close(self):
        self.tf_logger.close()

    def is_pointnet_based(self):
        return self._phase == 'DET' and cfg.NETWORK.DET.NAME.startswith('dpn')

    def test(self):
        if self._phase == 'REF':
            if self.cfg.TEST.REF.TRACKING:
                self.tracking = True
                return self.track_refine()
            else:
                return self.test_refine()
        elif self._phase == 'DET':
            if self.is_pointnet_based():
                self.test_det_pointnet()
            elif self.cfg.TEST.DET.QUICK_TEST:
                return self.test_det_quick()
            else:
                return self.test_det()
        else:
            raise KeyError

    def test_refine(self):
        """

        data_inputs: list of dict for each instance
                image: tensor HxWx3, cuda, float32, BGR
                K: tensor 3x3, cpu, float32
                class_idx: int
                pose_rend: tensor 1x7, cuda, float32
                image_rend: tensor HxWx3, cuda, float32
                # depth_rend:
                zoom_image_obs: tensor 1x3xHxW, cuda, float32, BGR
                zoom_image_rend: tensor 1x3xHxW, cuda, float32, BGR
                zoom_factor: tensor 1x4
        targets_det: list of dict for each instance
                box: tensor 1x4, cuda, float32
                class_idx: int
                class_id: int
                mask: tensor 1xHxW, cuda, float32
                pose: tensor 1x7, cuda, float32
                pose_target: tensor 1x7, cuda, float32
                # ---- optional ----
                pose_external: tensor 1x7, cuda, float32
                pose_icp: tensor 1x7, cuda, float32
        batch_preds: dict of tensor
            quat: tensor Nx4, cuda, float32
            trans, tensor Nx3, cuda, float32
            poses_est: tensor, Nx3x4, float32, cuda
        :return:
        """
        self._dcm_refine.eval()
        val_start = time.time()
        num_iter_refine = self.get_num_iters_refine()
        pose_results = {'poses_init': np.zeros((0, 7)),
                        'poses_ref': np.zeros((0, num_iter_refine, 7)),
                        'poses_gt': np.zeros((0, 7)),
                        'intrinsic_matrix': np.zeros((0, 3, 3)),
                        'poses_icp': np.zeros((0, 7)),
                        'class_idxs': np.zeros((0, ))}
        for i_batch, (data_cuda, targets) in enumerate(tqdm(self.test_loader)):
            # print(i)

            data_cuda = [{k: v.cuda() for k, v in t.items()} for t in data_cuda]
            # avoid `RuntimeError: received 0 items of ancdata`
            # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/4
            targets_cuda = [{k: v.cuda() for k, v in t.items()} for t in targets]

            data_ref, targets_ref = self.prepare_data_for_refine(data_cuda, targets_cuda)
            num_instance = len(data_ref)
            poses_pred = np.zeros((num_instance, num_iter_refine, 7))
            preds = None
            for i_refine in range(num_iter_refine):
                self.update_data_refine(data_ref, targets_ref, preds)
                # self.visualize(data_inputs[0], targets_det[0])
                inputs = self.merge_data(data_ref)
                preds = self._dcm_refine(**inputs)
                self.post_process_refine_preds(data_ref, preds)
                for i_instance in range(num_instance):
                    pose_est_mat = preds['poses_pred'][i_instance].cpu().numpy()
                    poses_pred[i_instance, i_refine] = matPose2quatPose(pose_est_mat)
            if cfg.TEST.VISUALIZE:
                self.update_data_refine(data_ref, targets_ref, preds)
                for i in range(len(data_ref)):
                    self.visualize(data_ref[i], targets_ref[i])
                    # plt.show()
                    from lib.utils.pose_error import add, adi, re, te
                    from lib.utils.se3 import quatPose2matPose
                    pose_init = quatPose2matPose(targets_ref[i]['pose_init'].cpu().numpy())
                    pose_gt = quatPose2matPose(targets_ref[i]['pose'].cpu().numpy())
                    pose_ours = quatPose2matPose(data_ref[i]['pose_rend'].cpu().numpy())
                    points = self._dataset_test._points_std[data_ref[i]['class_idx']]
                    add_init = add(pose_init[:3,:3], pose_init[:,3], pose_gt[:3,:3], pose_gt[:,3], points)
                    adi_init = adi(pose_init[:3,:3], pose_init[:,3], pose_gt[:3,:3], pose_gt[:,3], points)
                    add_ours = add(pose_ours[:3,:3], pose_ours[:,3], pose_gt[:3,:3], pose_gt[:,3], points)
                    adi_ours = adi(pose_ours[:3,:3], pose_ours[:,3], pose_gt[:3,:3], pose_gt[:,3], points)
                    re_init = re(pose_init[:3, :3], pose_gt[:3, :3])
                    te_init = te(pose_init[:, 3], pose_gt[:, 3])
                    re_ours = re(pose_ours[:3, :3], pose_gt[:3, :3])
                    te_ours = te(pose_ours[:, 3], pose_gt[:, 3])
                    print("add {:.3f}->{:.3f} adi {:.3f}->{:.3f}".format(add_init, add_ours, adi_init, adi_ours))
                    print("re {:.3f}->{:.3f} te {:.3f}->{:.3f}".format(re_init, re_ours, te_init, te_ours))
                    pass
                    class_name = self._dataset_test._class_names_used[data_ref[i]['class_idx']]
                    vis_dir = os.path.join(self.cfg.TRAIN.REF.OUTPUT_DIR, '../vis-{}'.format(class_name))
                    if not os.path.exists(vis_dir):
                        os.mkdir(vis_dir)
                    vis_fname = '{}-{}.jpg'.format(i_batch, i)
                    plt.savefig(os.path.join(vis_dir, vis_fname))

            poses_init = torch.cat([target['pose_init'] for target in targets_ref], dim=0).cpu().numpy()
            poses_gt = torch.cat([target['pose'] for target in targets_ref], dim=0).cpu().numpy()
            Ks = torch.stack([data['K'] for data in data_ref], dim=0).cpu().numpy()
            poses_icp = torch.cat([target['pose_icp'] for target in targets_ref], dim=0).cpu().numpy()
            class_idxs = np.array([data['class_idx'] for data in data_ref])

            pose_results['poses_init'] = np.concatenate((pose_results['poses_init'], poses_init), axis=0)
            pose_results['poses_ref'] = np.concatenate((pose_results['poses_ref'], poses_pred), axis=0)
            pose_results['poses_gt'] = np.concatenate((pose_results['poses_gt'], poses_gt), axis=0)
            pose_results['intrinsic_matrix'] = np.concatenate((pose_results['intrinsic_matrix'], Ks), axis=0)
            pose_results['poses_icp'] = np.concatenate((pose_results['poses_icp'], poses_icp), axis=0)
            pose_results['class_idxs'] = np.concatenate((pose_results['class_idxs'], class_idxs))

        self._dataset_test.evaluate_pose(pose_results)

        mute = self.at_val()
        acc = self._dataset_test.print_pose_accuracy(mute=mute)
        info = {
              '5d5cm': {},
              'ADD': {},
              'ADI': {},
              'VOCap_ADD': {},
              'VOCap_ADI': {},
              'xy_3cm': {},
              'z_4cm': {}}
        for j in [cfg.TEST.REF.ITERNUM-1]:
            info['5d5cm']['5d5cm_{}'.format(j)] = np.mean(acc[1:, j+1, 0])
            info['ADD']['ADD_{}'.format(j)] = np.mean(acc[1:, j+1, 1])
            info['ADI']['ADI_{}'.format(j)] = np.mean(acc[1:, j+1, 2])
            info['VOCap_ADD']['VOCap_ADD_{}'.format(j)] = np.mean(acc[1:, j+1, 4])
            info['VOCap_ADI']['VOCap_ADI_{}'.format(j)] = np.mean(acc[1:, j+1, 5])
            info['xy_3cm']['xy_3cm_{}'.format(j)] = np.mean(acc[1:, j+1, 6])
            info['z_4cm']['z_4cm_{}'.format(j)] = np.mean(acc[1:, j+1, 7])
        print_and_log("validation: 5d5cm: {:.4f}, ADD: {:.4f}, ADI: {:.4f}, using {:.2f} seconds"
                      .format(info['5d5cm']['5d5cm_{}'.format(cfg.TEST.REF.ITERNUM-1)],
                              info['ADD']['ADD_{}'.format(cfg.TEST.REF.ITERNUM-1)],
                              info['ADI']['ADI_{}'.format(cfg.TEST.REF.ITERNUM-1)],
                              time.time()-val_start))

        if self.at_val():
            if self.tf_logger is not None:
                for k in info:
                    self.tf_logger.add_scalars("logs_s/val/{}/".format(k), info[k], self.iter_train)

    def track_refine(self):
        """

        data_inputs: list of dict for each instance
                image: tensor HxWx3, cuda, float32, BGR
                K: tensor 3x3, cpu, float32
                class_idx: int
                pose_rend: tensor 1x7, cuda, float32
                image_rend: tensor HxWx3, cuda, float32
                # depth_rend:
                zoom_image_obs: tensor 1x3xHxW, cuda, float32, BGR
                zoom_image_rend: tensor 1x3xHxW, cuda, float32, BGR
                zoom_factor: tensor 1x4
        targets_det: list of dict for each instance
                box: tensor 1x4, cuda, float32
                class_idx: int
                class_id: int
                mask: tensor 1xHxW, cuda, float32
                pose: tensor 1x7, cuda, float32
                pose_target: tensor 1x7, cuda, float32
                # ---- optional ----
                pose_external: tensor 1x7, cuda, float32
                pose_icp: tensor 1x7, cuda, float32
        batch_preds: dict of tensor
            quat: tensor Nx4, cuda, float32
            trans, tensor Nx3, cuda, float32
            poses_est: tensor, Nx3x4, float32, cuda
        :return:
        """
        self._dcm_refine.eval()
        val_start = time.time()
        num_iter_refine = self.get_num_iters_refine()
        pose_results = {'poses_init': np.zeros((0, 7)),
                        'poses_ref': np.zeros((0, num_iter_refine, 7)),
                        'poses_gt': np.zeros((0, 7)),
                        'intrinsic_matrix': np.zeros((0, 3, 3)),
                        'poses_icp': np.zeros((0, 7)),
                        'class_idxs': np.zeros((0, ))}
        video_id_last_frame = ''
        for i_batch, (data_cuda, targets) in enumerate(tqdm(self.test_loader)):
            # print(i)
            assert(len(data_cuda)==1)
            video_id_cur_frame = data_cuda[0]['video_id']
            image_id_cur_frame = data_cuda[0]['image_id']
            del data_cuda[0]['video_id']
            del data_cuda[0]['image_id']
            if video_id_cur_frame != video_id_last_frame:
                preds = None
                print("{}->{}".format(video_id_last_frame, video_id_cur_frame))
                video_id_last_frame = video_id_cur_frame
            # avoid `RuntimeError: received 0 items of ancdata`
            # https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999/4
            data_cuda = [{k: v.cuda() for k, v in t.items()} for t in data_cuda]
            targets_cuda = [{k: v.cuda() for k, v in t.items()} for t in targets]
            data_ref, targets_ref = self.prepare_data_for_refine(data_cuda, targets_cuda)
            num_instance = len(data_ref)
            poses_pred = np.zeros((num_instance, num_iter_refine, 7))
            for i_refine in range(num_iter_refine):
                self.update_data_refine(data_ref, targets_ref, preds)
                if i_refine == 0:
                    poses_init = torch.cat([data['pose_rend'] for data in data_ref], dim=0).cpu().numpy()
                # self.visualize(data_inputs[0], targets_det[0])
                inputs = self.merge_data(data_ref)
                preds = self._dcm_refine(**inputs)
                self.post_process_refine_preds(data_ref, preds)
                for i_instance in range(num_instance):
                    pose_est_mat = preds['poses_pred'][i_instance].cpu().numpy()
                    poses_pred[i_instance, i_refine] = matPose2quatPose(pose_est_mat)
            # if cfg.TEST.VISUALIZE:
            self.update_data_refine(data_ref, targets_ref, preds)
            i = 0
            img = self.visualize_tracking(data_ref[i], targets_ref[i])
            class_name = self._dataset_test._class_names_used[data_ref[i]['class_idx']]
            vis_dir = os.path.join(self.cfg.TRAIN.REF.OUTPUT_DIR, '../track-{}-{}'.format(class_name, video_id_cur_frame))
            if not os.path.exists(vis_dir):
                os.mkdir(vis_dir)
            vis_fname = '{}.jpg'.format(image_id_cur_frame)
            cv2.imwrite(os.path.join(vis_dir, vis_fname), img*255)

            poses_gt = torch.cat([target['pose'] for target in targets_ref], dim=0).cpu().numpy()
            Ks = torch.stack([data['K'] for data in data_ref], dim=0).cpu().numpy()
            class_idxs = np.array([data['class_idx'] for data in data_ref])

            pose_results['poses_init'] = np.concatenate((pose_results['poses_init'], poses_init), axis=0)
            pose_results['poses_ref'] = np.concatenate((pose_results['poses_ref'], poses_pred), axis=0)
            pose_results['poses_gt'] = np.concatenate((pose_results['poses_gt'], poses_gt), axis=0)
            pose_results['intrinsic_matrix'] = np.concatenate((pose_results['intrinsic_matrix'], Ks), axis=0)
            pose_results['poses_icp'] = np.concatenate((pose_results['poses_icp'], poses_gt), axis=0)
            pose_results['class_idxs'] = np.concatenate((pose_results['class_idxs'], class_idxs))

        self._dataset_test.evaluate_pose(pose_results)

        mute = self.at_val()
        acc = self._dataset_test.print_pose_accuracy(mute=mute)
        info = {
              '5d5cm': {},
              'ADD': {},
              'ADI': {},
              'VOCap_ADD': {},
              'VOCap_ADI': {},
              'xy_3cm': {},
              'z_4cm': {}}
        for j in [cfg.TEST.REF.ITERNUM-1]:
            info['5d5cm']['5d5cm_{}'.format(j)] = np.mean(acc[1:, j+1, 0])
            info['ADD']['ADD_{}'.format(j)] = np.mean(acc[1:, j+1, 1])
            info['ADI']['ADI_{}'.format(j)] = np.mean(acc[1:, j+1, 2])
            info['VOCap_ADD']['VOCap_ADD_{}'.format(j)] = np.mean(acc[1:, j+1, 4])
            info['VOCap_ADI']['VOCap_ADI_{}'.format(j)] = np.mean(acc[1:, j+1, 5])
            info['xy_3cm']['xy_3cm_{}'.format(j)] = np.mean(acc[1:, j+1, 6])
            info['z_4cm']['z_4cm_{}'.format(j)] = np.mean(acc[1:, j+1, 7])
        print_and_log("validation: 5d5cm: {:.4f}, ADD: {:.4f}, ADI: {:.4f}, using {:.2f} seconds"
                      .format(info['5d5cm']['5d5cm_{}'.format(cfg.TEST.REF.ITERNUM-1)],
                              info['ADD']['ADD_{}'.format(cfg.TEST.REF.ITERNUM-1)],
                              info['ADI']['ADI_{}'.format(cfg.TEST.REF.ITERNUM-1)],
                              time.time()-val_start))

        if self.at_val():
            if self.tf_logger is not None:
                for k in info:
                    self.tf_logger.add_scalars("logs_s/val/{}/".format(k), info[k], self.iter_train)

    ### visualize
    def draw_box(self, x1, y1, x2, y2):
        color = [1., 0., 0.]
        plt.gca().add_patch(plt.Rectangle((x1, y1),
                                          x2-x1,
                                          y2-y1,
                                          fill=False,
                                          edgecolor=color,
                                          linewidth=3,
                                          clip_on=False))

    def retrive_image(self, image, normalized=False):
        if normalized:
            image = (image*self._pixel_std)+self._pixel_mean
            image = image[0].permute((1,2,0))
        image = image.cpu().numpy()
        image = image[:,:,[2,1,0]] # BGR2RGB
        image = np.clip(image, 0, 1)
        return image

    def visualize(self, data, target):
        """
            data: dict of one instance
                    image: tensor HxWx3, cuda, float32
                    K: tensor 3x3, cpu, float64
                    class_idx: int
                    pose_rend: tensor 1x7, cuda, float32
                    image_rend: tensor HxWx3, cuda, float32
                    depth_rend:
                    zoom_image_obs: tensor 1x3xHxW, cuda, float32
                    zoom_image_rend: tensor 1x3xHxW, cuda, float32
                    zoom_factor: tensor 1x4
            targets_det: list of dict for each instance
                    box: tensor 1x4, cuda, float32
                    class_idx: int
                    class_id: int
                    mask: tensor 1xHxW, cuda, float32
                    zoom_mask: tensor 1x1xHxW, cuda, float32
                    pose: tensor 1x7, cuda, float32
                    pose_target: tensor 1x7, cuda, float32
        """
        boxes = target['box'].cpu().numpy()
        # boxes_rend = data['box_rend']
        boxes_rend = boxes
        class_idx = target['class_idx']
        masks = target['mask'].cpu().numpy()
        poses_gt = target['pose_target'].cpu().numpy()
        poses_rend = data['pose_rend'].cpu().numpy()
        poses_init = target['pose_init'].cpu().numpy()
        # ratios = target.extra_fields['ratios'].numpy()

        m = 2
        n = 3

        fig = plt.figure(dpi=600)
        start = 1

        # show image_obs
        ax = fig.add_subplot(m, n, start)
        im_obs = self.retrive_image(data['image'])
        plt.imshow(im_obs)
        x1, y1, x2, y2 = boxes[0]
        self.draw_box(x1, y1, x2, y2)
        ax.set_title('image_obs')
        start+=1

        # fig_2 = 'im_rend'
        fig_2 = 'depth_obs'
        if fig_2 == 'im_rend':
            # show image_rend
            ax = fig.add_subplot(m, n, start)
            im_rend = self.retrive_image(data['image_rend'])
            plt.imshow(im_rend)
            x1, y1, x2, y2 = boxes_rend
            self.draw_box(x1, y1, x2, y2)
            ax.set_title('image_rend')
            start+=1
        elif fig_2 == 'depth_obs':
            # show depth_obs
            ax = fig.add_subplot(m, n, start)
            zoom_depth_obs = np.squeeze(data['zoom_depth_obs'].cpu().numpy())
            plt.imshow(zoom_depth_obs)
            ax.set_title('input_depth_obs')
            start+=1

        # fig_3 = 'gt_pose'
        fig_3 = 'depth_rend'
        if fig_3 == 'gt_pose':
            # show poses
            K = data['K'].cpu().numpy()
            fx = K[0, 0]
            fy = K[1, 1]
            px = K[0, 2]
            py = K[1, 2]
            zfar = 6.0
            znear = 0.01
            width = 640
            height = 480
            self.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
            self.renderer.set_light_pos([0, 0, 0])
            self.renderer.set_light_color([1, 1, 1])
            for i in range(poses_gt.shape[0]):
                qt = np.zeros((7, ), dtype=np.float32)
                qt[:3] = poses_gt[i][4:]
                qt[3:] = poses_gt[i][:4]
                self.renderer.set_poses([qt])
                class_ids = [target['class_id']]
                image, _, _, _ = self.renderer.render(class_ids, cpu_only=True)
                ax = fig.add_subplot(m, n, start)
                plt.imshow(image)
                ax.set_title('gt_pose')
                start += 1
            ax = fig.add_subplot(m, n, start)
            im_obs = self.retrive_image(data['image'])
            plt.imshow(im_obs)
            x1, y1, x2, y2 = boxes[0]
            self.draw_box(x1, y1, x2, y2)
            ax.set_title('image_obs')
            start+=1
        elif fig_3 == 'depth_rend':
            # show image_obs
            ax = fig.add_subplot(m, n, start)
            zoom_depth_rend = np.squeeze(data['zoom_depth_rend'].cpu().numpy())
            plt.imshow(zoom_depth_rend)
            ax.set_title('input_depth_rend')
            start+=1

        # show zoom_image_obs
        ax = fig.add_subplot(m, n, start)
        zoom_image_obs = self.retrive_image(data['zoom_image_obs'], normalized=True)
        plt.imshow(zoom_image_obs)
        ax.set_title('input_image_obs')
        start+=1

        # show zoom_image_rend
        ax = fig.add_subplot(m, n, start)
        zoom_image_rend = self.retrive_image(data['zoom_image_rend'], normalized=True)
        alpha = 1.
        plt.imshow(zoom_image_rend*alpha+zoom_image_obs*(1-alpha))
        ax.set_title('input_image_rend')
        start+=1

        if cfg.MODE == 'TRAIN':
            K = data['zoom_K'].cpu().numpy()
            fx = K[0, 0]
            fy = K[1, 1]
            px = K[0, 2]
            py = K[1, 2]
            zfar = 6.0
            znear = 0.01
            width = self.cfg.DATA.INPUT_WIDTH
            height = self.cfg.DATA.INPUT_HEIGHT
            self.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
            self.renderer.set_light_pos([0, 0, 0])
            self.renderer.set_light_color([1, 1, 1])
            for i in range(poses_rend.shape[0]):
                qt = np.zeros((7, ), dtype=np.float32)
                qt[:3] = poses_rend[i][4:]
                qt[3:] = poses_rend[i][:4]
                self.renderer.set_poses([qt])
                class_ids = [target['class_id']]
                zoom_image_rend_K, _, _, _ = self.renderer.render(class_ids, cpu_only=True)
                zoom_image_rend_K = zoom_image_rend_K[:,:,:3]
                if self.cfg.DATA.INPUT_WIDTH==320:
                    zoom_image_rend_K = zoom_image_rend_K[::2, ::2, :]
                ax = fig.add_subplot(m, n, start)
                plt.imshow(abs(zoom_image_rend_K-zoom_image_rend))
                ax.set_title('image_rend(zoom_K)')
                start += 1
        else:
            K = data['zoom_K'].cpu().numpy()
            fx = K[0, 0]
            fy = K[1, 1]
            px = K[0, 2]
            py = K[1, 2]
            zfar = 6.0
            znear = 0.01
            width = self.cfg.DATA.INPUT_WIDTH
            height = self.cfg.DATA.INPUT_HEIGHT
            self.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
            self.renderer.set_light_pos([0, 0, 0])
            self.renderer.set_light_color([1, 1, 1])
            for i in range(poses_init.shape[0]):
                qt = np.zeros((7, ), dtype=np.float32)
                qt[:3] = poses_init[i][4:]
                qt[3:] = poses_init[i][:4]
                self.renderer.set_poses([qt])
                class_ids = [target['class_id']]
                zoom_image_rend_init, _, _, _ = self.renderer.render(class_ids, cpu_only=True)
                zoom_image_rend_init = zoom_image_rend_init[:,:,:3]
                if self.cfg.DATA.INPUT_WIDTH==320:
                    zoom_image_rend_init = zoom_image_rend_init[::2, ::2, :]
                ax = fig.add_subplot(m, n, start)
                plt.imshow(zoom_image_rend_init*0.8+zoom_image_obs*0.2)
                ax.set_title('input_image_init')
                start += 1

    def visualize_tracking(self, data, target):
        """
            data: dict of one instance
                    image: tensor HxWx3, cuda, float32
                    K: tensor 3x3, cpu, float64
                    class_idx: int
                    pose_rend: tensor 1x7, cuda, float32
                    image_rend: tensor HxWx3, cuda, float32
                    depth_rend:
                    zoom_image_obs: tensor 1x3xHxW, cuda, float32
                    zoom_image_rend: tensor 1x3xHxW, cuda, float32
                    zoom_factor: tensor 1x4
            targets_det: list of dict for each instance
                    box: tensor 1x4, cuda, float32
                    class_idx: int
                    class_id: int
                    mask: tensor 1xHxW, cuda, float32
                    zoom_mask: tensor 1x1xHxW, cuda, float32
                    pose: tensor 1x7, cuda, float32
                    pose_target: tensor 1x7, cuda, float32
        """
        pose_gt = target['pose_target'].cpu().numpy()[0]
        pose_rend = data['pose_rend'].cpu().numpy()[0]

        def quick_rend(pose):
            K = data['K'].cpu().numpy()
            fx = K[0, 0]
            fy = K[1, 1]
            px = K[0, 2]
            py = K[1, 2]
            zfar = 6.0
            znear = 0.01
            width = 640
            height = 480
            self.renderer.set_projection_matrix(width, height, fx, fy, px, py, znear, zfar)
            self.renderer.set_light_pos([0, 0, 0])
            self.renderer.set_light_color([1, 1, 1])
            qt = np.zeros((7,), dtype=np.float32)
            qt[:3] = pose[4:]
            qt[3:] = pose[:4]
            self.renderer.set_poses([qt])
            class_ids = [target['class_id']]
            image, _, _, _ = self.renderer.render(class_ids, cpu_only=True)
            return image[:, :, :3]

        # show zoom_image_obs
        image_obs = data['image'].cpu().numpy()
        object_pred = quick_rend(pose_rend)[:, :, [2,1,0]]
        return image_obs*0.4+object_pred*0.6
