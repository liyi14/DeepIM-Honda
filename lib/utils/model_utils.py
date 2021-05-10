import numpy as np
from plyfile import PlyData
import os
from pyassimp import load
import matplotlib.pyplot as plt

def load_model_points(model_path, model_scale=1., max_num_points=99999, model_texture_path=None):
    assert os.path.exists(model_path), 'Path does not exist: {}'.format(model_path)
    model = load(model_path)
    assert (len(model.meshes) == 1)
    vertex = model.meshes[0].vertices
    points, indices = np.unique(vertex, axis=0, return_index=True)
    points = points*model_scale
    normals = model.meshes[0].normals[indices]
    tex_coord = model.meshes[0].texturecoords[0, indices, :2]

    if points.shape[0] > max_num_points:
        sample_points_idx = np.random.choice(np.arange(points.shape[0]), max_num_points)
        points = points[sample_points_idx, :].astype(np.float32)
        normals = normals[sample_points_idx, :].astype(np.float32)

    if model_texture_path is not None:
        tex = plt.imread(model_texture_path)[::-1, :, :3]
        point_color = np.array(
            [tex[int(tex_coord[i, 1] * (tex.shape[0]-1)), int(tex_coord[i, 0] * (tex.shape[1]-1)), :] for i in range(tex_coord.shape[0])])

        return points, normals, point_color
    else:
        return points, normals

if __name__ == '__main__':
    from PIL import Image, ImageFile
    # model_path = '/home/yili/PoseEst/deepim-pytorch/data/BOP/ycbv/models/obj_000002.ply'
    # texture_path = '/home/yili/PoseEst/deepim-pytorch/data/BOP/ycbv/models/obj_000002.png'
    model_path = '/home/yili/Storage/curiosity/ply_dir_from_blender/eval_0.ply'
    texture_path = '/home/yili/Storage/curiosity/ply_dir_from_blender/eval_0_color.png'
    xyz, normals, rgb = load_model_points(model_path, model_texture_path=texture_path)
    img = Image.open(texture_path).transpose(Image.FLIP_TOP_BOTTOM)
    # img = Image.open(path)
    img_data = np.fromstring(img.tobytes(), np.uint8)

    import matplotlib.pyplot as plt
    tex = plt.imread(texture_path)[::-1, :, :]

    K = np.array([[1.066778e+03, 0.000000e+00, 3.129869e+02],
                  [0.000000e+00, 1.067487e+03, 2.413109e+02],
                  [0.000000e+00, 0.000000e+00, 1.000000e+00]])
    # R = np.array([[0.023388461238133155, -0.6309259097300829, 0.7754917946168369],
    #               [0.9331277258632086, -0.2646079492419404, -0.24342225948802707],
    #               [0.35878191775210355, 0.7293255835844041, 0.5825453120678373]])
    R = np.array([0.5596066810828834, -0.8238361552774054, 0.09019791120233563,
                  -0.4048153183723089, -0.36668932120355346, -0.8376536032712047,
                  0.7231633184918785, 0.43224267730153954, -0.5387032002166867]).reshape([3,3])
    t = np.array([-33.83760567223681, -8.659128639509234, 590.451478877042]).reshape([3,1])
    # Rt = np.concatenate([R, t], axis=1)
    uvz = np.matmul(K, np.matmul(R, xyz.T)+t)
    u = uvz[0,:]/uvz[2,:]
    v = uvz[1,:]/uvz[2,:]
    idx = (v.round()*640+u.round()).astype('int')
    img = np.zeros([480,640,3]).reshape([-1,3])
    img[idx,:] = rgb
    img = img.reshape([480,640,3])
    plt.imshow(img)
    plt.show()
