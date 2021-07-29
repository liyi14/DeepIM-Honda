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
import json
import sys
import os
import os.path as osp
import cv2

from lib.fcn.config import cfg, cfg_from_file
from lib.fcn.train_dcm import DeepCodeMatchingTraining
from lib.fcn.process_cfg import process_cfg_and_build_logger
from lib.datasets.factory import get_dataset
from ycb_render.ycb_renderer import YCBRenderer
from lib.utils.print_and_log import print_and_log
import lib.networks as networks
import pprint
import torch.nn as nn
import torch.nn.functional as F
from lib.utils.se3 import *
import matplotlib.pyplot as plt
from pyassimp import load
from lib.utils.pose_error import add, adi, re, te

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

    args = parser.parse_args()
    return args

class SimpleTester(object):
    def __init__(self, cfg, args, class_names, renderer, network):
        self.cfg = cfg
        self.args = args
        self.renderer = renderer
        self._dcm_refine = network
        self._pixel_mean = torch.reshape(torch.cuda.FloatTensor([0.45, 0.432, 0.411]), (1, 3, 1, 1))  # BGR order
        self._pixel_std = torch.reshape(torch.cuda.FloatTensor([1, 1, 1]), (1, 3, 1, 1))

        self._optimizer_cfg = self.cfg.TRAIN.REF
        self._network_cfg = self.cfg.NETWORK.REF
        self._loss_cfg = self.cfg.LOSS.REF
        self._test_cfg = self.cfg.TEST.REF
        self.num_gpus = 1
        self.vis_dir = 'vis'

    def load_points(self, model_mesh_paths, model_scale=1.):
        """

        :return:
        points_ori: raw points
        points_neat: points that are cut to same size
        points_std: rescale them to adjust the loss
        """
        num_model = len(model_mesh_paths)
        points_ori = []
        # num = np.inf
        num = 100000
        for model_path in model_mesh_paths:
            assert os.path.exists(model_path), 'Path does not exist: {}'.format(model_path)
            model = load(model_path)
            assert (len(model.meshes) == 1)
            vertex = model.meshes[0].vertices
            points = np.unique(vertex, axis=0)*model_scale
            points = np.concatenate([points, np.ones([points.shape[0], 1])], axis=1).T
            points_ori.append(torch.cuda.FloatTensor(points))
        self._points = points_ori

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

    def prepare_data_for_refine(self, data):
        data_ref = []
        #TODO deal with situation when the number of instance in the data_cuda is not static
        instance_data_cuda = {}
        instance_data_cuda['image'] = data['image']
        instance_data_cuda['depth'] = data['depth']
        if 'pcloud' in data:
            instance_data_cuda['pcloud'] = data['pcloud']
        instance_data_cuda['K'] = data['K']
        # use gt label, can be replaced with
        instance_class_idx = data['class_idx']
        instance_data_cuda['class_idx'] = instance_class_idx
        instance_data_cuda['pose_rend'] = data['init_pose']
        instance_data_cuda['pose_init'] = data['init_pose']
        data_ref.append(instance_data_cuda)
        return data_ref

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

    @staticmethod
    def get_zoomed_K(K, zoom_factor, input_size, output_size):
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

    def post_process_refine_preds(self, data_ref, preds):
        poses_rend_qt = torch.cat([data['pose_rend'] for data in data_ref])
        poses_est = self.get_pose_mat(poses_rend_qt,
                                      preds['trans'],
                                      rot_delta_quat=preds['quat'],
                                      rot_delta_mats=preds['mat'])
        preds['poses_pred'] = poses_est

    def update_data_refine(self, data_ref, preds):
        num_samples = len(data_ref)
        remove_idx = []
        for sample_idx in range(num_samples):
            data = data_ref[sample_idx]

            # render reference image and depth
            if preds:
                pose_rend = matPose2quatPose(
                    preds['poses_pred'][sample_idx].detach().cpu().numpy())
                data_ref[sample_idx]['pose_rend'] = torch.FloatTensor(pose_rend).cuda().unsqueeze(0)
            data_rend = self.rend_one_image([data['pose_rend'].cpu().numpy().flatten()],
                                            [data['class_idx']],
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
                                                                 affine_matrix[1, 2]]], dtype=torch.float32).cuda()

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

            zoom_image_rend = F.grid_sample(image_rend, grids)
            zoom_depth_rend = data_ref[sample_idx]['depth_rend'].unsqueeze(0).unsqueeze(0)
            zoom_depth_rend = F.grid_sample(zoom_depth_rend, grids)
            zoom_image_rend -= self._pixel_mean
            zoom_image_rend /= self._pixel_std
            data_ref[sample_idx]['zoom_image_rend'] = zoom_image_rend
            data_ref[sample_idx]['zoom_depth_rend'] = zoom_depth_rend

    def merge_data(self, data_ref, selected_keys=[]):
        inputs = {}
        if len(selected_keys)==0:
            selected_keys = data_ref[0].keys()
        for k in selected_keys:
            if k not in data_ref[0]:
                raise KeyError
            if k in ['box_rend',
                     'K']:
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
                       'pose_rend']:
                inputs[k] = torch.cat([data[k] for data in data_ref], dim=0).detach()
        return inputs

    def test_one_image(self, data, prefix=0):
        self._dcm_refine.eval()
        num_iter_refine = self.cfg.TEST.REF.ITERNUM
        data_cuda = {}
        for k,v in data.items():
            data_cuda[k] = torch.cuda.FloatTensor(v) if type(v) == np.ndarray else v
        data_ref = self.prepare_data_for_refine(data_cuda)
        num_instance = 1
        preds = None
        for i_refine in range(num_iter_refine):
            self.update_data_refine(data_ref, preds)
            # self.visualize_input(data_ref[0])
            inputs = self.merge_data(data_ref)
            preds = self._dcm_refine(**inputs)
            self.post_process_refine_preds(data_ref, preds)

        if cfg.TEST.VISUALIZE or 'gt_pose' in data:
            self.update_data_refine(data_ref, preds)

            if cfg.TEST.VISUALIZE:
                for i in range(len(data_ref)):
                    vis_fname = '{}-{}_init.jpg'.format(prefix, i)
                    vis = self.visualize(data_ref[i], 'init')
                    cv2.imwrite(os.path.join(self.vis_dir, vis_fname), (vis* 255).astype('uint8'))
                    vis = self.visualize(data_ref[i], 'rend')
                    vis_fname = '{}-{}_refine.jpg'.format(prefix, i)
                    cv2.imwrite(os.path.join(self.vis_dir, vis_fname), (vis* 255).astype('uint8'))

            if 'gt_pose' in data:
                pose_init = quatPose2matPose(data['init_pose'])
                pose_gt = quatPose2matPose(data['gt_pose'])
                pose_ours = quatPose2matPose(data_ref[0]['pose_rend'].cpu().numpy())
                re_init = re(pose_init[:3, :3], pose_gt[:3, :3])
                te_init = te(pose_init[:, 3], pose_gt[:, 3])
                re_ours = re(pose_ours[:3, :3], pose_gt[:3, :3])
                te_ours = te(pose_ours[:, 3], pose_gt[:, 3])
                print("re {:.3f}->{:.3f} te {:.3f}->{:.3f}".format(re_init, re_ours, te_init, te_ours))

        pose_est_mat = preds['poses_pred'][0].cpu().numpy()
        return pose_est_mat

    def retrive_image(self, image, normalized=False):
        if normalized:
            image = (image*self._pixel_std)+self._pixel_mean
            image = image[0].permute((1,2,0))
        image = image.cpu().numpy()
        image = image[:,:,[2,1,0]] # BGR2RGB
        image = np.clip(image, 0, 1)
        return image

    def visualize_input(self, data):
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
        m = 2
        n = 2

        fig = plt.figure(dpi=600)
        start = 1

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

        # show image_obs
        ax = fig.add_subplot(m, n, start)
        zoom_depth_obs = np.squeeze(data['zoom_depth_obs'].cpu().numpy())
        plt.imshow(zoom_depth_obs)
        ax.set_title('input_depth_obs')
        start += 1

        # show image_obs
        ax = fig.add_subplot(m, n, start)
        zoom_depth_rend = np.squeeze(data['zoom_depth_rend'].cpu().numpy())
        plt.imshow(zoom_depth_rend)
        ax.set_title('input_depth_rend')
        start += 1

        plt.show()


    def visualize(self, data, type='rend'):
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
        if type == 'rend':
            pose_rend = data['pose_rend'].cpu().numpy()[0]
        elif type == 'init':
            pose_rend = data['pose_init'].cpu().numpy()[0]
        else:
            raise KeyError

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
            class_ids = [data['class_idx']]
            image, _, _, _ = self.renderer.render(class_ids, cpu_only=True)
            return image[:, :, :3]

        # show zoom_image_obs
        image_obs = data['image'].cpu().numpy()
        object_pred = quick_rend(pose_rend)[:, :, [2, 1, 0]]
        return image_obs * 0.4 + object_pred * 0.6

def load_network(num_classes):
    output_dir_train = cfg.TRAIN.REF.OUTPUT_DIR
    pretrained_model = os.path.join(output_dir_train, "ref_model_final.pth")
    network_data = torch.load(pretrained_model, map_location=torch.device('cpu'))
    network = networks.__dict__['dcm_ot_v0'](num_classes, network_data).cuda()
    return network

def load_renderer(class_names):
    print_and_log('=> loading 3D models')
    model_scale = 1
    dataset_root_path = 'data/curiosity'
    model_mesh_paths = []
    model_texture_paths = []
    model_colors = []
    for name in class_names:
        if name == '__background__':
            model_mesh_paths.append('tools/default/__background__/textured_simple.obj')
            model_texture_paths.append('tools/default/__background__/texture_map.png')
        else:
            model_mesh_paths.append('{}/models/{}/model.obj'.format(dataset_root_path, name))
            model_texture_paths.append('{}/models/{}/texture_map.jpg'.format(dataset_root_path, name))
        model_colors.append((255,255,255))
    renderer = YCBRenderer(width=cfg.SYN.WIDTH,
                           height=cfg.SYN.HEIGHT,
                           gpu_id=cfg.ABS_GPU_ID[0],
                           render_marker=False,
                           model_scale=model_scale)
    renderer.load_objects(model_mesh_paths,
                          model_texture_paths,
                          model_colors)
    renderer.set_camera_default()
    print_and_log('=> 3D models Loaded')
    return renderer, model_mesh_paths

if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg_file)
    cfg.MODE = 'TEST'
    # device
    cfg.GPU_ID = [int(i) for i in args.gpu_id.split(',')]
    # all available GPUs
    if cfg.GPU_ID[0] == -1:
        import torch
        cfg.GPU_ID = list(range(torch.cuda.device_count()))

    cfg.ABS_GPU_ID = [int(i) for i in args.abs_gpu_id.split(',')]
    if args.abs_gpu_id[0] == -1:
        cfg.ABS_GPU_ID = cfg.GPU_ID
    process_cfg_and_build_logger(args, 'TEST')

    if args.vis:
        cfg.TEST.VISUALIZE = True
    torch.cuda.set_device(cfg.GPU_ID[0])
    print("Let's use", len(cfg.GPU_ID), "GPUs!")

    class_names = ['__background__', 'block_23', 'block_25']
    renderer, model_mesh_paths = load_renderer(class_names)
    networks = load_network(len(class_names))

    train = SimpleTester(cfg, args, class_names, renderer, networks)
    train.load_points(model_mesh_paths, model_scale=1.)
    for index in range(200):
        image_path = os.path.join('data/curiosity/data/{:06d}-color.jpg'.format(index))
        depth_path = os.path.join('data/curiosity/data/{:06d}-depth.png'.format(index))
        meta_path = os.path.join('data/curiosity/data/{:06d}-meta.json'.format(index))
        if not os.path.exists(image_path):
            break
        image = cv2.imread(image_path)/255
        depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)/1000
        with open(meta_path) as f:
            meta = json.load(f)
        data = {'image': image,
                'depth': depth,
                'K': np.array(meta['K']).reshape([3,3]),
                'class_idx': class_names.index(meta['class_name']),
                'init_pose': np.array([meta['init_pose']])}
        if 'gt_pose' in meta:
            data['gt_pose'] = np.array([meta['gt_pose']])
        with torch.no_grad():
            train.test_one_image(data, index)
    print(args.cfg_file)
