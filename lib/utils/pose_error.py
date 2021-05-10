# Author: Tomas Hodan (hodantom@cmp.felk.cvut.cz)
# Center for Machine Perception, Czech Technical University in Prague

# Implementation of the pose error functions described in:
# Hodan et al., "On Evaluation of 6D Object Pose Estimation", ECCVW 2016

import math
import numpy as np
from scipy import spatial
from transforms3d.quaternions import qmult, qinverse, quat2axangle, quat2mat

def transform_pts_Rt(pts, R, t):
    """
    Applies a rigid transformation to 3D points.

    :param pts: nx3 ndarray with 3D points.
    :param R: 3x3 rotation matrix.
    :param t: 3x1 translation vector.
    :return: nx3 ndarray with transformed 3D points.
    """
    assert(pts.shape[1] == 3)
    pts_t = R.dot(pts.T) + t.reshape((3, 1))
    return pts_t.T

def pts2pixels(pts, R, t, K):
    pts = transform_pts_Rt(pts, R, t)
    pixels = K.dot(pts.T)
    pixels = pixels.T
    pixels[:, 0] /= pixels[:, 2]
    pixels[:, 1] /= pixels[:, 2]
    return pixels

def reproj(K, R_est, t_est, R_gt, t_gt, pts):
    """
    reprojection error.
    :param K intrinsic matrix
    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    pixels_est = K.dot(pts_est.T)
    pixels_est = pixels_est.T
    pixels_gt = K.dot(pts_gt.T)
    pixels_gt = pixels_gt.T

    n = pts.shape[0]
    est = np.zeros((n, 2), dtype=np.float32);
    est[:, 0] = np.divide(pixels_est[:, 0], pixels_est[:, 2])
    est[:, 1] = np.divide(pixels_est[:, 1], pixels_est[:, 2])

    gt = np.zeros((n, 2), dtype=np.float32);
    gt[:, 0] = np.divide(pixels_gt[:, 0], pixels_gt[:, 2])
    gt[:, 1] = np.divide(pixels_gt[:, 1], pixels_gt[:, 2])

    e = np.linalg.norm(est - gt, axis=1).mean()
    return e

def add(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with no indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)
    e = np.linalg.norm(pts_est - pts_gt, axis=1).mean()
    return e

def adi(R_est, t_est, R_gt, t_gt, pts):
    """
    Average Distance of Model Points for objects with indistinguishable views
    - by Hinterstoisser et al. (ACCV 2012).

    :param R_est, t_est: Estimated pose (3x3 rot. matrix and 3x1 trans. vector).
    :param R_gt, t_gt: GT pose (3x3 rot. matrix and 3x1 trans. vector).
    :param model: Object model given by a dictionary where item 'pts'
    is nx3 ndarray with 3D model points.
    :return: Error of pose_est w.r.t. pose_gt.
    """
    pts_est = transform_pts_Rt(pts, R_est, t_est)
    pts_gt = transform_pts_Rt(pts, R_gt, t_gt)

    # Calculate distances to the nearest neighbors from pts_gt to pts_est
    nn_index = spatial.cKDTree(pts_est)
    nn_dists, _ = nn_index.query(pts_gt, k=1)

    e = nn_dists.mean()
    return e

def re(R_est, R_gt):
    """
    Rotational Error.

    :param R_est: Rotational element of the estimated pose (3x1 vector).
    :param R_gt: Rotational element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(R_est.shape == R_gt.shape == (3, 3))
    error_cos = 0.5 * (np.trace(R_est.dot(np.linalg.inv(R_gt))) - 1.0)
    error_cos = min(1.0, max(-1.0, error_cos)) # Avoid invalid values due to numerical errors
    error = math.acos(error_cos)
    error = 180.0 * error / np.pi # [rad] -> [deg]
    return error

def re_quat(q_est, q_gt):
    """
    https://fgiesen.wordpress.com/2013/01/07/small-note-on-quaternion-distance-metrics/
    https://ww2.mathworks.cn/help/fusion/ref/quaternion.dist.html
    :param q_est: (4,) or (B, 4)
    :param q_gt: (4,) or (1, 4)
    :return:
    """
    # assert (q_est.shape == q_gt.shape == (4,))
    # qd = qmult(qinverse(q_est), q_gt)
    # vec, theta = quat2axangle(qd)
    if np.isnan(q_est.sum()+q_gt.sum()):
        print(q_est, q_gt)
    q_est = q_est.reshape([-1,4])
    q_gt = q_gt.reshape([-1,4])
    q_est_norm = np.sqrt((q_est*q_est).sum(axis=1,keepdims=True))
    q_est = q_est / q_est_norm
    q_gt_norm = np.sqrt((q_gt*q_gt).sum(axis=1,keepdims=True))
    q_gt = q_gt / q_gt_norm
    # theta = np.arccos(2 * np.power(np.dot(q_est, q_gt), 2) - 1)
    theta = np.arccos(2*np.power(np.sum(q_est*q_gt, axis=1), 2)-1)
    if np.isnan(theta.sum()):
        print(np.where(np.isnan(theta)))
    return theta / np.pi * 180

def te(t_est, t_gt):
    """
    Translational Error.

    :param t_est: Translation element of the estimated pose (3x1 vector).
    :param t_gt: Translation element of the ground truth pose (3x1 vector).
    :return: Error of t_est w.r.t. t_gt.
    """
    assert(t_est.size == t_gt.size == 3)
    error = np.linalg.norm(t_gt - t_est)
    return error

if __name__ == '__main__':
    import time
    for i in range(10):
        q1, q2 = np.random.rand(2,4)
        t1 = time.time()
        print(re_quat(q1, q2))
        t2 = time.time()
        print(re(quat2mat(q1), quat2mat(q2)))
        t3 = time.time()
        print(t2-t1, t3-t2)

