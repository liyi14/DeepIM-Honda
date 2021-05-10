# --------------------------------------------------------
# DA-RNN
# Copyright (c) 2016 RSE at UW
# Licensed under The MIT License [see LICENSE for details]
# Written by Yu Xiang
# --------------------------------------------------------

import numpy as np
import torch
from lib.utils.se3 import se3_inverse, se3_mul

# backproject pixels into 3D points in camera's coordinate system
def backproject_camera(depth, intrinsic_matrix):
    # get intrinsic matrix
    K = intrinsic_matrix
    Kinv = np.linalg.inv(K)

    # compute the 3D points
    width = depth.shape[1]
    height = depth.shape[0]

    # construct the 2D points matrix
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    ones = np.ones((height, width), dtype=np.float32)
    x2d = np.stack((x, y, ones), axis=2).reshape(width * height, 3)

    # backprojection
    R = np.dot(Kinv, x2d.transpose())

    # compute the 3D points
    X = np.multiply(np.tile(depth.reshape(1, width * height), (3, 1)), R).reshape(3, height, width)
    return np.array(X)

def backproject_camera_tensor(depth, K):
    Kinv = torch.inverse(K).to(depth.device)
    width = depth.shape[1]
    height = depth.shape[0]
    # torch.meshgrid acts different with np.meshgrid
    y, x = torch.meshgrid(torch.arange(height), torch.arange(width))
    y = y.float().to(depth.device)
    x = x.float().to(depth.device)
    ones = torch.ones(height, width).to(depth.device)
    x2d = torch.stack((x, y, ones), dim=2).reshape(width * height, 3)
    R = torch.matmul(Kinv, x2d.T).reshape(3, height, width)
    X = depth.unsqueeze(0) * R
    X = X.permute([1,2,0])
    return X

def depth2pcloud(depth, K):
    pcloud = np.ascontiguousarray(backproject_camera(depth, K).transpose([1,2,0]))
    return pcloud

def pcloud2normal(pcloud):
    """
dzdx=(z(y,x+1)-z(y,x-1))/2.0;
dzdy=(z(y+1,x)-z(y-1,x))/2.0;
direction=(-dzdx,-dzdy,1.0)
magnitude=sqrt(direction.x**2 + direction.y**2 + direction.z**2)
normal=direction/magnitude
    :param pcloud:
    :return:
    """
    dz_dzdx = pcloud[1:-1, 2:, 2]-pcloud[1:-1, :-2, 2]
    dz_dzdx[pcloud[1:-1, 2:, 2]==0] = 0
    dz_dzdx[pcloud[1:-1, :-2, 2]==0] = 0
    dx_dzdx = pcloud[1:-1, 2:, 0]-pcloud[1:-1, :-2, 0]
    # dx_dzdx = np.max(dx_dzdx, 0.0001)
    dzdx = dz_dzdx/dx_dzdx
    dzdx[dx_dzdx==0] = 0
    dz_dzdy = pcloud[2:, 1:-1, 2]-pcloud[:-2, 1:-1, 2]
    dz_dzdy[pcloud[2:, 1:-1, 2]==0] = 0
    dz_dzdy[pcloud[:-2, 1:-1, 2]==0] = 0
    dy_dzdy = pcloud[2:, 1:-1, 1]-pcloud[:-2, 1:-1, 1]
    # dy_dzdy = np.max(dy_dzdy, 0.0001)
    dzdy = dz_dzdy/dy_dzdy
    dzdy[dy_dzdy==0] = 0

    normal = np.dstack((-dzdx, -dzdy, np.ones_like(dzdx)))
    normal_full = np.zeros_like(pcloud)
    normal_full[:,:,2] = 1.
    normal_full[1:-1, 1:-1, :] = normal
    n = np.linalg.norm(normal_full, axis=2)
    normal_full[:, :, 0] /= n
    normal_full[:, :, 1] /= n
    normal_full[:, :, 2] /= n

    return normal_full


if __name__ == '__main__':
    height = 480
    width = 640
    depth = np.ones([height, width])
    intrinsic_matrix = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02], \
                                [0.000000e+00, 1.067487e+03, 2.413109e+02], \
                                [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    X = backproject_camera(depth, intrinsic_matrix)
    X_tensor = backproject_camera_tensor(torch.FloatTensor(depth), torch.FloatTensor(intrinsic_matrix))
    pass
