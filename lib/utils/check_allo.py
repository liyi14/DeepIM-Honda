import numpy as np

from transforms3d.quaternions import mat2quat, quat2mat
from numpy import linalg as LA
from math import pi, cos, sin, acos

import torch
import torch.nn as nn
import cv2
from transforms3d.quaternions import mat2quat, qinverse

def zoom_object(RT, x3d, K):
    ratio = 0.75
    x3d_full = np.ones([4, x3d.shape[0]])
    x3d_full[:3, :] = x3d.transpose([1, 0])
    x2d = np.matmul(K, np.matmul(RT, x3d_full))
    x2d[0, :] = np.divide(x2d[0, :], x2d[2, :])
    x2d[1, :] = np.divide(x2d[1, :], x2d[2, :])

    obj_start_x = np.min(x2d[0, :])
    obj_start_y = np.min(x2d[1, :])
    obj_end_x = np.max(x2d[0, :])
    obj_end_y = np.max(x2d[1, :])
    obj_imgn_c = np.dot(K, RT[:, 3])
    zoom_c_x = obj_imgn_c[0] / obj_imgn_c[2]
    zoom_c_y = obj_imgn_c[1] / obj_imgn_c[2]

    left_dist = zoom_c_x - obj_start_x
    right_dist = obj_end_x - zoom_c_x
    up_dist = zoom_c_y - obj_start_y
    down_dist = obj_end_y - zoom_c_y
    crop_height = np.max([ratio * right_dist, ratio * left_dist, up_dist, down_dist]) * 2.8
    crop_width = crop_height / ratio
    print("width: {:.2f}, height: {:.2f}, c: ({:.2f}, {:.2f})".format(obj_end_x-obj_start_x, obj_end_y-obj_start_y, zoom_c_x, zoom_c_y))
    print("left: {:.2f}, right: {:.2f}, up: {:.2f}, down: {:.2f}".format(left_dist, right_dist, up_dist, down_dist, crop_height))

    x1 = (zoom_c_x - crop_width / 2) * 2 / width - 1;
    x2 = (zoom_c_x + crop_width / 2) * 2 / width - 1;
    y1 = (zoom_c_y - crop_height / 2) * 2 / height - 1;
    y2 = (zoom_c_y + crop_height / 2) * 2 / height - 1;

    pts1 = np.float32([[x1, y1], [x1, y2], [x2, y1]])
    pts2 = np.float32([[-1, -1], [-1, 1], [1, -1]])
    affine_matrix = cv2.getAffineTransform(pts2, pts1)

    return affine_matrix

def apply_zoom(input, affine_matrix):
    inputs = torch.tensor(np.expand_dims(input, 0).transpose([0, 3, 1, 2]).copy(), dtype=torch.float64)
    affine_matrics = torch.tensor(np.expand_dims(affine_matrix, 0).copy(), dtype=torch.float64)
    grids = nn.functional.affine_grid(affine_matrics, inputs.size())
    input_zoom = nn.functional.grid_sample(inputs, grids).detach().cpu().numpy()
    input_zoom = input_zoom[0].transpose([1,2,0])
    return input_zoom

def qmult(q1, q2):
    ''' Multiply two quaternions
    See : http://en.wikipedia.org/wiki/Quaternions#Hamilton_product
    '''
    w1 = q1[:, 0::4]
    x1 = q1[:, 1::4]
    y1 = q1[:, 2::4]
    z1 = q1[:, 3::4]
    w2 = q2[:, 0::4]
    x2 = q2[:, 1::4]
    y2 = q2[:, 2::4]
    z2 = q2[:, 3::4]
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 + y1*w2 + z1*x2 - x1*z2
    z = w1*z2 + z1*w2 + x1*y2 - y1*x2
    return torch.cat([w, x, y, z], dim=-1)

def _cross_matrix(x):
    '''
    cross product matrix
    '''
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])

def a2e(q):
    '''
    allocentric to egocentric
    '''
    p = np.array([0, 0, 1])
    r = _cross_matrix(np.cross(p, q))
    Rae = np.eye(3) + r + r.dot(r)/(1 + np.dot(p, q))
    return Rae

def convert_between_allo_ego(t_src, t_est):
    '''
    generate a2e and e2a matrices
    '''
    quat_e2a = qinverse(a2e(t_src))
    quat_a2e = a2e(t_est)
    return quat_a2e, quat_e2a

def aa2mat(axis, degree):
    return quat2mat(aa2quat(axis, degree))

def aa2quat(axis, degree):
    axis /= LA.norm(axis)
    return [cos(degree / 360. * pi), axis[0] * sin(degree / 360. * pi), axis[1] * sin(degree / 360. * pi),
     axis[2] * sin(degree / 360. * pi)]

def quat2aa(quat):
    quat /= LA.norm(quat)
    if quat[0]<0:
        quat = -quat
    axis = quat[1:]
    axis /= LA.norm(axis)
    angle = acos(quat[0])/pi*360
    return (axis, angle)

def mat2aa(mat):
    quat = mat2quat(mat)
    return quat2aa(quat)

def render(renderer, rot, trans):
    pose = list(trans)+list(mat2quat(rot))
    renderer.set_poses([pose])
    frame = renderer.render([0], cpu_only=True)
    return frame[0][:, :, [2,1,0]]

def render_allo(renderer, rot, trans):
    pose = trans+list(mat2quat(rot))
    renderer.set_allocentric_poses([pose])
    frame = renderer.render([0], cpu_only=True)
    return frame[0][:, :, [2,1,0]]


if __name__ == '__main__':
    from ycb_render.ycb_renderer import YCBRenderer
    width = 640
    height = 480
    model_path = '/home/yili/PoseEst/deepim-pytorch/data/YCB_Video_Dataset'
    models = ["003_cracker_box"]
    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(model_path, item) for item in models]
    points_file = ['{}/models/{}/points.xyz'.format(model_path, item) for item in models]
    points = np.array([np.loadtxt(point_file) for point_file in points_file])

    colors = [[255,255,255]]
    renderer = YCBRenderer(width=width, height=height, render_marker=False)
    renderer.load_objects(obj_paths, texture_paths, colors)
    K = np.array([[572., 0, 320.], [0, 572., 240.], [0, 0, 1.]])
    renderer.set_projection_matrix(width, height, 572., 572., 320., 240., 0.2, 6.0)
    # renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
    renderer.set_camera_default()
    # renderer.set_fov(40)
    renderer.set_light_pos([0, 1, 1])

    # zoom
    import matplotlib.pyplot as plt

    rot_allo = aa2mat([1, 0, 0], 90.)

    trans_1 = [-0.1, -0.1, 0.7]
    rot_1 = np.matmul(a2e(trans_1), rot_allo)
    RT_1 = np.zeros([3, 4])
    RT_1[:, :3] = rot_1
    RT_1[:, 3] = trans_1
    est_img_1a = render(renderer, rot_1, trans_1)
    est_img_1b = render_allo(renderer, rot_allo, trans_1)
    plt.subplot(2, 4, 1)
    plt.imshow(est_img_1a)
    plt.title('allo using matrix, location 1')
    # plt.subplot(2, 4, 3)
    # plt.imshow(est_img_1b)
    # plt.title('allo using ycb_renderer, location 1')

    zoom_factor = zoom_object(RT_1, points[0], K)
    print(zoom_factor)
    zoomed_img_1a = apply_zoom(est_img_1a, zoom_factor)
    plt.subplot(2, 4, 5)
    plt.imshow(zoomed_img_1a)
    # zoomed_img_1b = apply_zoom(est_img_1b, zoom_factor)
    # plt.subplot(2, 4, 7)
    # plt.imshow(zoomed_img_1b)

    trans_2 = [0., 0., 0.7]
    rot_2 = np.matmul(a2e(trans_2), rot_allo)
    RT_2 = np.zeros([3, 4])
    RT_2[:, :3] = rot_2
    RT_2[:, 3] = trans_2
    est_img_2a = render(renderer, rot_2, trans_2)
    est_img_2b = render_allo(renderer, rot_allo, trans_2)
    plt.subplot(2, 4, 2)
    plt.imshow(est_img_2a)
    plt.title('allo using matrix, location 2')
    # plt.subplot(2, 4, 4)
    # plt.imshow(est_img_2b)
    # plt.title('allo using ycb_renderer, location 2')

    zoom_factor = zoom_object(RT_2, points[0], K)
    print(zoom_factor)
    zoomed_img_2a = apply_zoom(est_img_2a, zoom_factor)
    plt.subplot(2, 4, 6)
    plt.imshow(zoomed_img_2a)
    # zoomed_img_2b = apply_zoom(est_img_2b, zoom_factor)
    # plt.subplot(2, 4, 8)
    # plt.imshow(zoomed_img_2b)

    plt.show()