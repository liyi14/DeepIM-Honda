import numpy as np
import torch
from lib.fcn.config import cfg
from lib.utils.get_closest_pose import get_closest_pose_batch_cpu, RT_transform_batch_cpu
from transforms3d.quaternions import mat2quat, quat2mat, qmult, axangle2quat
from lib.utils.se3 import T_inv_transform, se3_mul, se3_inverse, angular_distance
from lib.utils.print_and_log import print_and_log

def initialize_poses(sample):
    pose_result = sample['poses_result'].numpy()

    # construct poses target
    pose_est = np.zeros((0, 9), dtype=np.float32)
    for i in range(pose_result.shape[0]):
        for j in range(pose_result.shape[1]):
            pose_result[i, j, 0] = i
            pose_est = np.concatenate((pose_est, pose_result[i, j, :].reshape(1, 9)), axis=0)

    return pose_est


def compute_pose_target(quaternion_delta, translation, poses_src, poses_gt, sym_info):
    poses_tgt = poses_src.copy()
    num = poses_src.shape[0]
    errors_rot = np.zeros((num, ), dtype=np.float32)
    errors_trans = np.zeros((num, ), dtype=np.float32)
    errors_trans_xy = np.zeros((num, ), dtype=np.float32)
    errors_trans_z = np.zeros((num, ), dtype=np.float32)
    poses_tgt = RT_transform_batch_cpu(quaternion_delta, translation, poses_src)
    closest_gt = get_closest_pose_batch_cpu(poses_tgt, poses_gt, sym_info)
    for i in range(poses_src.shape[0]):
        # compute pose errors
        errors_rot[i] = np.arccos(2 * np.power(np.dot(poses_tgt[i, 2:6], closest_gt[i, 2:6]), 2) - 1) * 180.0 / np.pi
        errors_trans[i] = np.linalg.norm(poses_tgt[i, 6:] - closest_gt[i, 6:]) * 100
        errors_trans_xy[i] = np.linalg.norm(poses_tgt[i, 6:8] - closest_gt[i, 6:8]) * 100
        errors_trans_z[i] = np.abs(poses_tgt[i, 8] - closest_gt[i, 8]) * 100

    return poses_tgt, np.mean(errors_rot), np.mean(errors_trans), np.mean(errors_trans_xy), np.mean(errors_trans_z)

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)

def squeeze_rot_trans(raw_quat, raw_trans, poses_src):
    batch_size = poses_src.shape[0]
    quat = np.zeros((batch_size, 4))
    trans = np.zeros((batch_size, 3))
    if cfg.NETWORK.CLASS_AWARE:
        for i in range(batch_size):
            cls = int(poses_src[i, 1])
            quat[i] = raw_quat[i, 4 * cls:(4 * cls + 4)]
            trans[i] = raw_trans[i, 3 * cls:(3 * cls + 3)]
    return raw_quat, raw_trans


def compute_delta_poses(pose_src_blob, pose_tgt_blob, zoom_factor):

    num = pose_src_blob.shape[0]
    pose_delta_blob = pose_src_blob.copy()

    for i in range(num):
        R_src = quat2mat(pose_src_blob[i, 2:6])
        T_src = pose_src_blob[i, 6:]

        R_tgt = quat2mat(pose_tgt_blob[i, 2:6])
        T_tgt = pose_tgt_blob[i, 6:]

        R_delta = np.dot(R_tgt, R_src.transpose())
        T_delta = T_inv_transform(T_src, T_tgt)

        pose_delta_blob[i, 2:6] = mat2quat(R_delta)
        pose_delta_blob[i, 6] = T_delta[0] / zoom_factor[i, 0]
        pose_delta_blob[i, 7] = T_delta[1] / zoom_factor[i, 1]
        pose_delta_blob[i, 8] = T_delta[2]

    return pose_delta_blob


def initialize_tracking(sample, pose_est):
    pose_result = sample['poses'].numpy()

    # construct poses target
    pose_est = np.zeros((0, 9), dtype=np.float32)
    for i in range(pose_result.shape[0]):
        for j in range(pose_result.shape[1]):
            pose_result[i, j, 0] = i
            pose_est = np.concatenate((pose_est, pose_result[i, j, :].reshape(1, 9)), axis=0)

    return pose_est

def check_tracking_valid(sample, train_data, pixel_thresh=900):
    mask_gt = train_data['mask_gt'][0,0,:,:].cpu().numpy()
    # mask_imgn = train_data['mask_imgn'][0,0,:,:].cpu().numpy()
    seg_gt =sample['label_blob'][0,0,:,:].cpu().numpy()
    num_mask_gt = np.sum(mask_gt)
    num_seg_gt = np.sum(seg_gt)
    # if np.float(num_seg_gt)/num_mask_gt < 0.2:
    #     print("Occlusion too heavy")
    # print("num_seg_gt: {}, num_mask_gt: {}".format(num_seg_gt, num_mask_gt))
    if num_seg_gt <= pixel_thresh or num_mask_gt <= pixel_thresh:
        print_and_log("Object at {}-{} not visible. {}, {}".format(sample['video_id'][0], sample['image_id'][0], num_seg_gt, num_mask_gt))
        return False
    # x = mask_imgn+mask_gt
    # if np.float(np.sum(x==2))/np.sum(x>0) < 0.3:
    #     print("tracking failed at {}-{}, use PoseCNN results to re-init".format(sample['video_id'][0], sample['image_id'][0]))
    #     return False
    return True