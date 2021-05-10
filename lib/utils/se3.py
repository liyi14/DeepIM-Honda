# --------------------------------------------------------
# FCN
# Copyright (c) 2018 NVIDIA
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
import torch
from transforms3d.quaternions import mat2quat, quat2mat, quat2axangle
import random


def qmult_tensor(q1, q2):
    ''' Multiply two quaternions

    Parameters
    ----------
    q1 : Bx4
    q2 : Bx4

    Returns
    -------
    q12 : Bx4

    Notes
    -----
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    '''
    w1, x1, y1, z1 = q1.T
    w2, x2, y2, z2 = q2.T
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return torch.stack([w, x, y, z], dim=1)

# RT is a 3x4 matrix
def quatPose2matPose(quatPose_raw, homo=False):
    quatPose = quatPose_raw.flatten()
    assert(quatPose.shape)==(7,)
    if homo:
        matPose = np.zeros((4,4))
        matPose[3,3] = 1.
    else:
        matPose = np.zeros((3,4))
    matPose[:3, :3] = quat2mat(quatPose[:4])
    matPose[:3, 3] = quatPose[4:]
    return matPose

def matPose2quatPose(matPose):
    if matPose.shape == (4,4):
        matPose = matPose[:3, :]
    assert(matPose.shape)==(3,4)
    quatPose = np.zeros((7,))
    quatPose[:4] = mat2quat(matPose[:,:3])
    quatPose[4:] = matPose[:, 3]
    return quatPose

def quat2mat_tensor(rot_quat):
    s, u, v, w = rot_quat
    s, u, v, w = rot_quat / (s * s + u * u + v * v + w * w)
    rot_mat = torch.zeros((3,3), dtype=torch.float32, device=rot_quat.device)
    rot_mat[0, 0] = s * s + u * u - v * v - w * w
    rot_mat[0, 1] = 2 * (u * v - s * w)
    rot_mat[0, 2] = 2 * (u * w + s * v)
    rot_mat[1, 0] = 2 * (u * v + s * w)
    rot_mat[1, 1] = s * s - u * u + v * v - w * w
    rot_mat[1, 2] = 2 * (v * w - s * u)
    rot_mat[2, 0] = 2 * (u * w - s * v)
    rot_mat[2, 1] = 2 * (v * w + s * u)
    rot_mat[2, 2] = s * s - u * u - v * v + w * w
    return rot_mat

def qtPose2matPose_tensor(pose_qt):
    pose_mat = torch.zeros([3,4], device=pose_qt.device, dtype=pose_qt.dtype)
    pose_mat[:3, :3] = quat2mat_tensor(pose_qt[:4])
    x, y, z = pose_qt[4:]
    pose_mat[0, -1] = x
    pose_mat[1, -1] = y
    pose_mat[2, -1] = z
    return pose_mat

def se3_inverse_tensor(RT):
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3, 1))
    RT_new = torch.zeros_like(RT)
    RT_new[0:3, 0:3] = R.transpose
    RT_new[0:3, 3] = -1*torch.dot(R.transpose, T).flatten()
    return RT_new

def se3_inverse(RT):
    """
    return the inverse of a RT
    :param RT=[R,T], 4x3 np array
    :return: RT_new=[R,T], 4x3 np array
    """
    R = RT[0:3, 0:3]
    T = RT[0:3, 3].reshape((3,1))
    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = R.transpose()
    RT_new[0:3, 3] = -1 * np.dot(R.transpose(), T).reshape((3))
    return RT_new

def se3_mul(RT1, RT2):
    """
    concat 2 RT transform
    :param RT1=[R,T], 4x3 np array
    :param RT2=[R,T], 4x3 np array
    :return: RT_new = RT1 * RT2
    """
    R1 = RT1[0:3, 0:3]
    T1 = RT1[0:3, 3].reshape((3,1))

    R2 = RT2[0:3, 0:3]
    T2 = RT2[0:3, 3].reshape((3,1))

    RT_new = np.zeros((3, 4), dtype=np.float32)
    RT_new[0:3, 0:3] = np.dot(R1, R2)
    T_new = np.dot(R1, T2) + T1
    RT_new[0:3, 3] = T_new.reshape((3))
    return RT_new


def T_inv_transform(T_src, T_tgt):
    '''
    :param T_src: 
    :param T_tgt:
    :return: T_delta: delta in pixel 
    '''
    T_delta = np.zeros((3, ), dtype=np.float32)

    T_delta[0] = T_tgt[0] / T_tgt[2] - T_src[0] / T_src[2]
    T_delta[1] = T_tgt[1] / T_tgt[2] - T_src[1] / T_src[2]
    T_delta[2] = np.log(T_src[2] / T_tgt[2])

    return T_delta


def rotation_x(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = 1
    R[1, 1] = np.cos(t)
    R[1, 2] = -np.sin(t)
    R[2, 1] = np.sin(t)
    R[2, 2] = np.cos(t)
    return R

def rotation_y(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = np.cos(t)
    R[0, 2] = np.sin(t)
    R[1, 1] = 1
    R[2, 0] = -np.sin(t)
    R[2, 2] = np.cos(t)
    return R

def rotation_z(theta):
    t = theta * np.pi / 180.0
    R = np.zeros((3, 3), dtype=np.float32)
    R[0, 0] = np.cos(t)
    R[0, 1] = -np.sin(t)
    R[1, 0] = np.sin(t)
    R[1, 1] = np.cos(t)
    R[2, 2] = 1
    return R

def angular_distance(quat):
    vec, theta = quat2axangle(quat)
    return theta / np.pi * 180
from math import sqrt, sin, cos, pi

def get_random_rotation(device=None, type='quat'):
    # u1, u2, u3 = np.random.random(3)
    # return np.array([sqrt(1-u1)*sin(2*pi*u2), sqrt(1-u1)*cos(2*pi*u2), sqrt(u1)*cos(2*pi*u3), sqrt(u1)*cos(2*pi*u3)])
    v = np.random.normal(size=4)
    v /= np.linalg.norm(v)
    if type == 'mat':
        v = quat2mat(v)

    if device is None:
        return v
    else:
        return torch.tensor(v, dtype=torch.float32, device=device)

def get_valid_location(K, valid_range):
    """

    :param K: 3x3
    :param valid_range: 3x2, [wrange, hrange, zrange]
    :return:
    """
    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]
    w = random.randint(valid_range[0,0], valid_range[0,1])
    h = random.randint(valid_range[1,0], valid_range[1,1])
    z = random.random()*(valid_range[2,1] - valid_range[2,0])+valid_range[2,0]
    x = (w-cx)/fx*z
    y = (h-cy)/fy*z
    return np.array([x,y,z])

# def get_reasonable_occ(K, target, extent):
#     fx = K[0,0]
#     fy = K[1,1]
#     cx = K[0,2]
#     cy = K[1,2]
#     z = target[2]
#     w = (target[0]*fx+cx)/z
#     h = (target[1]*fy+cy)/z
#     range_offset_w = extent*fx/z
#     range_offset_h = extent*fy/z
#     range_offset_z =




