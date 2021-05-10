# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Blob helper functions."""

import numpy as np
import cv2
import torch
import torch.nn as nn
import random

from lib.fcn.config import cfg


def BGR2HLS_torch(im):
    # im values from [0., 1.)
    # H: [0, 360.)
    # L: [0, 1.)
    # S: [0, 1.)
    B = im[:, :, 0]
    G = im[:, :, 1]
    R = im[:, :, 2]
    v_max, i_max = torch.max(im, dim=2)
    v_min, i_min = torch.min(im, dim=2)
    v_sum = v_max + v_min
    v_sub = v_max - v_min
    eps = 1e-4
    safe_v_sub = v_sub + (v_sub == 0.).float() * eps

    L = v_sum / 2.
    safe_L = L + (L == 0.).float() * eps - (L == 1.).float() * eps

    S = v_sub / (2 * safe_L) * (L <= 0.5).float() + v_sub / (2 - 2 * safe_L) * (L > 0.5).float()

    H = ((G - B) * (i_max == 2).float()
         + (B - R) * (i_max == 1).float()
         + (R - G) * (i_max == 0).float()) * 60. / safe_v_sub \
        + (2.0 - i_max.float()) * 120
    H %= 360.

    return H, L, S


def HLS2BGR_torch(H, L, S):
    q = (L * (1 + S)) * (L < 0.5).float() + (L + S - (L * S)) * (L >= 0.5).float()
    p = 2 * L - q
    q = torch.unsqueeze(q, 2)
    p = torch.unsqueeze(p, 2)
    hk = H / 360.
    tr = hk + 1. / 3
    tg = hk
    tb = hk - 1. / 3
    tc = torch.stack([tb, tg, tr], dim=2)
    tc = tc + (tc < 0.).float() - (tc > 1.).float()
    BGR = (p + ((q - p) * 6. * tc)) * (tc < 1. / 6).float() \
          + q * ((tc >= 1. / 6) * (tc < 1. / 2)).float() \
          + (p + ((q - p) * 6 * (2. / 3 - tc))) * ((tc >= 1. / 2) * (tc < 2. / 3)).float() \
          + p * (tc >= 2. / 3).float()
    return BGR


def color_jittering_torch(im_torch, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1)[0] - 0.5) * cfg.DATA.CHROM_NOISE[0] * 360
    if d_l is None:
        d_l = (np.random.rand(1)[0] - 0.5) * cfg.DATA.CHROM_NOISE[1]
    if d_s is None:
        d_s = (np.random.rand(1)[0] - 0.5) * cfg.DATA.CHROM_NOISE[2]
    # Convert the BGR to HLS
    h, l, s = BGR2HLS_torch(im_torch)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 360
    new_l = torch.clamp(l + d_l, 0, 1.)
    new_s = torch.clamp(s + d_s, 0, 1.)
    # Convert the HLS to BGR
    new_im = HLS2BGR_torch(new_h, new_l, new_s)
    # print(torch.max(new_im), torch.min(new_im))
    return new_im


def im_list_to_blob(ims, num_channels):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], num_channels),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        if num_channels == 1:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im[:, :, np.newaxis]
        else:
            blob[i, 0:im.shape[0], 0:im.shape[1], :] = im

    return blob


def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale


def pad_im(im, factor, value=0):
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return np.lib.pad(im, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=value)
    elif len(im.shape) == 2:
        return np.lib.pad(im, ((0, pad_height), (0, pad_width)), 'constant', constant_values=value)


def unpad_im(im, factor):
    height = im.shape[0]
    width = im.shape[1]

    pad_height = int(np.ceil(height / float(factor)) * factor - height)
    pad_width = int(np.ceil(width / float(factor)) * factor - width)

    if len(im.shape) == 3:
        return im[0:height - pad_height, 0:width - pad_width, :]
    elif len(im.shape) == 2:
        return im[0:height - pad_height, 0:width - pad_width]


def chromatic_transform(im, label=None, d_h=None, d_s=None, d_l=None):
    """
    Given an image array, add the hue, saturation and luminosity to the image
    """
    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (np.random.rand(1) - 0.5) * 0.02 * 180
    if d_l is None:
        d_l = (np.random.rand(1) - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (np.random.rand(1) - 0.5) * 0.2 * 256
    # Convert the BGR to HLS
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    h, l, s = cv2.split(hls)
    # Add the values to the image H, L, S
    new_h = (h + d_h) % 180
    new_l = np.clip(l + d_l, 0, 255)
    new_s = np.clip(s + d_s, 0, 255)
    # Convert the HLS to BGR
    new_hls = cv2.merge((new_h, new_l, new_s)).astype('uint8')
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2BGR)

    if label is not None:
        I = np.where(label > 0)
        new_im[I[0], I[1], :] = im[I[0], I[1], :]
    return new_im


def add_noise(image):
    # random number
    r = np.random.rand(1)
    # gaussian noise
    if r < 1 - cfg.DATA.MOTION_BLUR_RATIO:
        row, col, ch = image.shape
        mean = 0
        var = np.random.rand(1) * 0.3 * 256
        sigma = var ** 0.5
        gauss = sigma * np.random.randn(row, col) + mean
        gauss = np.repeat(gauss[:, :, np.newaxis], ch, axis=2)
        noisy = image + gauss
        noisy = np.clip(noisy, 0, 255)
        return noisy
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        diameter = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = np.zeros((diameter, diameter))
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((diameter - 1) / 2), :] = np.ones(diameter)
        else:
            kernel_motion_blur[:, int((diameter - 1) / 2)] = np.ones(diameter)
        kernel_motion_blur = kernel_motion_blur / diameter
        noisy = cv2.filter2D(image, -1, kernel_motion_blur)
        return noisy


def add_noise_torch(image):
    # random number
    r = np.random.rand(1)

    # gaussian noise
    if r < 1 - cfg.DATA.MOTION_BLUR_RATIO:
        noise_level = random.uniform(0, 0.05)
        gauss = torch.randn_like(image) * noise_level
        noisy = image + gauss
        noisy = torch.clamp(noisy, 0, 1.0)
    else:
        # motion blur
        sizes = [3, 5, 7, 9, 11, 15]
        diameter = sizes[int(np.random.randint(len(sizes), size=1))]
        kernel_motion_blur = torch.zeros((diameter, diameter), device=image.device)
        if np.random.rand(1) < 0.5:
            kernel_motion_blur[int((diameter - 1) / 2), :] = torch.ones(diameter)
        else:
            kernel_motion_blur[:, int((diameter - 1) / 2)] = torch.ones(diameter)
        kernel_motion_blur = kernel_motion_blur / diameter
        kernel_motion_blur = kernel_motion_blur.view(1, 1, diameter, diameter)
        kernel_motion_blur = kernel_motion_blur.repeat(image.size(2), 1, 1, 1)

        motion_blur_filter = nn.Conv2d(in_channels=image.size(2),
                                       out_channels=image.size(2),
                                       kernel_size=diameter,
                                       groups=image.size(2),
                                       bias=False,
                                       padding=int(diameter / 2))

        motion_blur_filter.weight.data = kernel_motion_blur
        motion_blur_filter.weight.requires_grad = False
        noisy = motion_blur_filter(image.permute(2, 0, 1).unsqueeze(0))
        noisy = noisy.squeeze(0).permute(1, 2, 0)

    return noisy


def add_holes(image, depth, im_label, bbox, hole_ratio=0.):
    mask_area = im_label.sum() * hole_ratio
    bbox = bbox[0]
    x1, y1, x2, y2 = bbox
    patch_h = patch_w = int(np.sqrt(mask_area))
    patch_h = np.clip(patch_h, 0, y2 - y1 - 1)
    patch_w = np.clip(patch_w, 0, x2 - x1 - 1)
    mask_start_x = np.random.randint(x1, x2 - patch_w)
    mask_start_y = np.random.randint(y1, y2 - patch_h)
    image[mask_start_y:mask_start_y + patch_h, mask_start_x:mask_start_x + patch_w] = 0.
    depth[mask_start_y:mask_start_y + patch_h, mask_start_x:mask_start_x + patch_w] = 0.
    return image, depth


def crop_and_resize(data, bbox, dst_size, keep_aspect_ratio=True, dilate_factor=0.):
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    input_height = dst_size[0]
    input_width = dst_size[1]
    aspect_ratio_hw = input_height / input_width
    if keep_aspect_ratio:
        w_match = h / aspect_ratio_hw
        h_match = w * aspect_ratio_hw
        w_resize = max(w, w_match)
        h_resize = max(h, h_match)
    else:
        w_resize = w
        h_resize = h
    w_resize *= (1 + dilate_factor)
    h_resize *= (1 + dilate_factor)
    srcTri = np.float32([[cx - w_resize / 2, cy - h_resize / 2],
                         [cx + w_resize / 2, cy - h_resize / 2],
                         [cx - w_resize / 2, cy + h_resize / 2]])
    dstTri = np.float32([[0, 0],
                         [input_width - 1, 0],
                         [0, input_height - 1]])

    warp_mat = cv2.getAffineTransform(srcTri, dstTri)
    patch = cv2.warpAffine(data, warp_mat, (input_width, input_height))
    return patch
