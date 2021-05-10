import numpy as np

from lib.utils.pose_error import re
from transforms3d.quaternions import mat2quat, quat2mat
from numpy import linalg as LA
from math import pi, cos, sin, acos
import torch
def get_closest_pose_batch_gpu(est_pose_batch, gt_pose_batch, sym_info_batch):
    est_pose = est_pose_batch.cpu().numpy()
    est_rot = [quat2mat(x[2:6]) for x in est_pose]
    cls_idx = est_pose[:, 1].astype(int)
    gt_pose = gt_pose_batch.cpu().numpy()
    gt_rot = [quat2mat(x[2:6]) for x in gt_pose]
    sym_info = sym_info_batch.cpu().numpy()
    closest_pose_batch = gt_pose.copy()
    batch_size = est_pose.shape[0]
    for i in range(batch_size):
        closest_rot = get_closest_pose(est_rot[i], gt_rot[i], sym_info[i][cls_idx[i]])
        closest_pose_batch[i][2:6] = mat2quat(closest_rot)
    closest_pose_batch = torch.cuda.FloatTensor(closest_pose_batch, device=gt_pose_batch.device)
    return closest_pose_batch

def get_closest_pose_batch_cpu(est_pose_batch, gt_pose_batch, sym_info_batch):
    est_pose = est_pose_batch
    est_rot = [quat2mat(x[2:6]) for x in est_pose]
    cls_idx = est_pose[:, 1].astype(int)
    gt_pose = gt_pose_batch
    gt_rot = [quat2mat(x[2:6]) for x in gt_pose]
    sym_info = sym_info_batch
    batch_size = est_pose.shape[0]
    closest_pose_batch = gt_pose.copy()
    for i in range(batch_size):
        closest_rot = get_closest_pose(est_rot[i], gt_rot[i], sym_info[i][cls_idx[i]])
        closest_pose_batch[i][2:6] = mat2quat(closest_rot)
    return closest_pose_batch

def RT_transform_batch_gpu(quaternion_delta, translation, poses_src_batch):
    quaternion_delta = quaternion_delta.detach().cpu().numpy()
    translation = translation.detach().cpu().numpy()
    poses_src = poses_src_batch.cpu().numpy()
    poses_tgt = RT_transform_batch_cpu(quaternion_delta, translation, poses_src)
    poses_tgt = torch.cuda.FloatTensor(poses_tgt, device=poses_src_batch.device)
    return poses_tgt

def RT_transform_batch_cpu(rotation_delta, translation, poses_src, rot_type='quaternion'):
    poses_tgt = poses_src.copy()
    if rot_type == 'quaternion':
        quaternion_delta = rotation_delta
        for i in range(poses_src.shape[0]):
            cls = int(poses_src[i, 1]) if quaternion_delta.shape[1]>4 else 0
            if all(poses_src[i, 2:]==0):
                poses_tgt[i, 2:] = 0
            else:
                if np.isnan(quaternion_delta).any():
                    print("Nan detected!")
                    poses_tgt[i, 2:6] = poses_src[i, 2:6]
                else:
                    poses_tgt[i, 2:6] = mat2quat(
                        np.dot(quat2mat(quaternion_delta[i, 4*cls:4*cls+4]), quat2mat(poses_src[i, 2:6])))
                poses_tgt[i, 6:] = translation[i, 3*cls:3*cls+3]
    elif rot_type == 'axisangle':
        axisangle_delta = rotation_delta
        for i in range(poses_src.shape[0]):
            cls = int(poses_src[i, 1]) if axisangle_delta.shape[1]>3 else 0
            if all(poses_src[i, 2:]==0):
                poses_tgt[i, 2:] = 0
            else:
                poses_tgt[i, 2:6] = mat2quat(
                    np.dot(aa2mat(axisangle_delta[i, 4*cls:4*cls+4]), quat2mat(poses_src[i, 2:6])))
                poses_tgt[i, 6:] = translation[i, 3*cls:3*cls+3]
    return poses_tgt

def get_closest_pose(est_rot, gt_rot, sym_info):
    # sym_info: (sym_axis, sym_angle) [0:3, 3]
    def gen_mat(axis, degree):
        axis = axis / LA.norm(axis)
        return quat2mat([cos(degree/360.*pi), axis[0] * sin(degree/360.*pi), axis[1] * sin(degree/360.*pi), axis[2] * sin(degree/360.*pi)])
    if len(sym_info) == 0 or sym_info[0][3] == -1:
        closest_rot = gt_rot
    elif sym_info[0][3] == 0:
        sym_angle = int(sym_info[0][3])
        sym_axis = np.copy(sym_info[0][:3])
        angle = 180.
        gt_rot_1 = np.copy(gt_rot)
        rd_1 = re(gt_rot_1, est_rot)
        gt_rot_2 = np.matmul(gt_rot, gen_mat(sym_axis, angle))
        rd_2 = re(gt_rot_2, est_rot)
        if rd_1<rd_2:
            gt_rot_1 = np.matmul(gt_rot, gen_mat(sym_axis, -90))
            gt_rot_2 = np.matmul(gt_rot, gen_mat(sym_axis, 90))
        else:
            gt_rot_1 = np.matmul(gt_rot, gen_mat(sym_axis, 90))
            gt_rot_2 = np.matmul(gt_rot, gen_mat(sym_axis, 270))
        rd_1 = re(gt_rot_1, est_rot)
        rd_2 = re(gt_rot_2, est_rot)
        count = 1
        thresh = 0.1
        while angle>thresh:
            angle /= 2
            count += 1
            if rd_1<rd_2:
                gt_rot_2 = np.matmul(gt_rot_2, gen_mat(sym_axis, -angle))
                rd_2 = re(gt_rot_2, est_rot)
            else:
                gt_rot_1 = np.matmul(gt_rot_1, gen_mat(sym_axis, angle))
                rd_1 = re(gt_rot_1, est_rot)

        # print("rd_1: {}, rd_2: {}, angle: {}, count: {}".format(rd_1, rd_2, angle, count))
        closest_rot = gt_rot_1 if rd_1<rd_2 else gt_rot_2
    else:
        closest_rot = np.copy(gt_rot)
        closest_angle = re(gt_rot, est_rot)
        for sub_sym_info in sym_info:
            if sub_sym_info[3] < -0.5: # -1
                break
            rot_delta = gen_mat(sub_sym_info[:3], sub_sym_info[3])
            another_gt_rot = np.matmul(gt_rot, rot_delta)
            rd = re(another_gt_rot, est_rot)
            if rd < closest_angle:
                closest_rot = np.copy(another_gt_rot)
                closest_angle = rd

    return closest_rot


if __name__ == '__main__':
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
        pose = trans+list(mat2quat(rot))
        renderer.set_poses([pose])
        frame = renderer.render([0], cpu_only=True)
        return frame[0][:, :, [2,1,0]]
    import time
    from ycb_render.ycb_renderer import YCBRenderer
    width = 640
    height = 480
    model_path = '/home/yili/PoseEst/deepim-pytorch/data/YCB_Video_Dataset'
    models = ["036_wood_block"]
    obj_paths = [
        '{}/models/{}/textured_simple.obj'.format(model_path, item) for item in models]
    texture_paths = [
        '{}/models/{}/texture_map.png'.format(model_path, item) for item in models]
    points_file = ['{}/models/{}/points.xyz'.format(model_path, item) for item in models]
    points = np.array([np.loadtxt(point_file) for point_file in points_file])

    colors = [[255,255,255]]
    renderer = YCBRenderer(width=width, height=height, render_marker=False)
    renderer.load_objects(obj_paths, texture_paths, colors)
    K = np.array([[572, 0, 325], [0, 572, 242], [0, 0, 1]])
    renderer.set_projection_matrix(width, height, 572., 572., 325., 242., 0.2, 6.0)
    # renderer.set_camera(cam_pos, [0, 0, 0], [0, 1, 0])
    renderer.set_camera_default()
    # renderer.set_fov(40)
    renderer.set_light_pos([0, 1, 1])

    trans = [-0.0021458883, 0.0804758, 0.78142926]

    # # cloesest check
    # axis = [0,0,1]
    # est_rot = aa2mat(axis, 0.1)
    # gt_rot = aa2mat(axis, 180)
    # sym_info = np.array(axis+[0])
    # est_pose = np.random.rand(3,4)
    # est_pose[:, :3] = est_rot
    # gt_pose = np.random.rand(3,4)
    # gt_pose[:, :3] = gt_rot
    # rd_ori = re(est_rot, gt_rot)
    # t = time.time()
    # closest_rot = get_closest_pose(est_rot, gt_rot, sym_info)
    # print("use {} seconds".format(time.time()-t))
    # closest_pose = np.copy(gt_pose)
    # closest_pose[:, :3] = closest_rot
    # rd_refined = re(est_rot, closest_pose[:, :3])
    # print("est_rot: {}, gt rot: {}, refined gt: {}".format(mat2aa(est_rot), mat2aa(gt_rot), mat2aa(closest_rot)))
    # print("original rot dist: {}, refined rot dist: {}".format(rd_ori, rd_refined))
    #
    # import matplotlib.pyplot as plt
    # est_img = render(renderer, est_rot, trans)
    # plt.subplot(1, 3, 1)
    # plt.imshow(est_img[:, :, [2,1,0]])
    # gt_img = render(renderer, gt_rot, trans)
    # plt.subplot(1, 3, 2)
    # plt.imshow(gt_img[:, :, [2,1,0]])
    # closest_img = render(renderer, closest_rot, trans)
    # plt.subplot(1, 3, 3)
    # plt.imshow(closest_img[:, :, [2,1,0]])
    # plt.show()

    # import cv2
    # while(1):
    #     est_img = render(renderer, est_rot, trans)
    #     cv2.imshow('test', cv2.cvtColor(est_img, cv2.COLOR_RGB2BGR))
    #     q = cv2.waitKey(16)
    #     if q == ord('w'):
    #         trans[1] += 0.05
    #     elif q == ord('s'):
    #         trans[1] -= 0.05
    #     elif q == ord('a'):
    #         trans[0] -= 0.1
    #     elif q == ord('d'):
    #         trans[0] += 0.1
    #     elif q == ord('q'):
    #         trans[2] += 0.01
    #     elif q == ord('e'):
    #         trans[2] -= 0.01
    #     print trans

    import cv2
    rot = aa2mat([1, 0, 0], 180)
    a = aa2mat([0, 0, 1], 5)
    d = aa2mat([0, 0, 1], -5)
    q = aa2mat([0, 1, 0], 5)
    e = aa2mat([0, 1, 0], -5)
    w = aa2mat([1, 0, 0], 5)
    s = aa2mat([1, 0, 0], -5)
    while(1):
        est_img = render(renderer, rot, trans)
        cv2.imshow('test', cv2.cvtColor(est_img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(16)
        if key == ord('w'):
            rot = np.matmul(w, rot)
        elif key == ord('s'):
            rot = np.matmul(s, rot)
        elif key == ord('a'):
            rot = np.matmul(a, rot)
        elif key == ord('d'):
            rot = np.matmul(d, rot)
        elif key == ord('q'):
            rot = np.matmul(q, rot)
        elif key == ord('e'):
            rot = np.matmul(e, rot)
        print(mat2quat(rot))