# Copyright (c) 2017 Lightricks. All rights reserved.

from cv2 import resize
import numpy as np
from skimage.transform import resize as upsample
from scipy.misc import imresize as downsample


class InterpulationOrders(object):
    nearest = 0
    bilinear = 1
    quadratic = 2
    cubic = 3
    quartic = 4
    quintic = 5


def box_filter(src_im, r=60):
    """
    imDst(x, y)=sum(sum(imSrc(x-r:x+r,y-r:y+r)))
    running time independent on r
    :param src_im: np.array image channel
    :param r: int - kernel radius
    :return: np.array after box filter (without normalization)
    """
    assert (len(src_im.shape)) == 2
    r = int(r)

    dst_im = np.zeros_like(src_im, dtype=np.float32)

    # cimlative sum over Y axis:
    cum_im = np.cumsum(src_im, axis=0)
    # differences over Y axis:
    dst_im[:r + 1, :] = cum_im[r: 2 * r + 1, :]
    dst_im[r + 1:-r, :] = cum_im[2 * r + 1:, :] - cum_im[:- 2 * r - 1, :]
    dst_im[-r:, :] = np.repeat(cum_im[-1:, :], repeats=r, axis=0) - cum_im[-2 * r - 1:-r - 1, :]

    # cimlative sum over X axis:
    cum_im = np.cumsum(dst_im, axis=1)
    # # differences over X axis:
    dst_im[:, :r + 1] = cum_im[:, r: 2 * r + 1]
    dst_im[:, r + 1:-r] = cum_im[:, 2 * r + 1:] - cum_im[:, : - 2 * r - 1]
    dst_im[:, -r:] = np.repeat(cum_im[:, -1:], repeats=r, axis=1) - cum_im[:, -2 * r - 1:-r - 1]

    return dst_im


def validate_input(guide):
    if guide.dtype == np.uint8:
        guide = np.float64(guide / 255.)
    assert np.max(guide) <= 1

    return guide

def staged_resize(img, max_long_side=640, size_divisor=16):

    while max(img.shape[:2]) > max_long_side and np.mod(img.shape[0]/2, size_divisor) == 0 and \
            np.mod(img.shape[1]/2, size_divisor) == 0:

        img = resize(img, (0, 0), fx=0.5, fy=0.5)

    return img

def get_subsampled_inputs(guide, src, radius, subsample_ratio):
    if guide.shape[:2] == src.shape:
        s = subsample_ratio
        src_sub = downsample(src, 1. / s, interp='nearest') / 255.
    else:
        src_sub = src
        s = np.array(guide.shape[:2], dtype=np.float32) / np.array(src.shape[:2])
        if np.abs(s[0] - s[1]) < 0.05:
            s = s[0]
        else:
            # TODO add support for different subsample ratio for axis
            assert 0, 'for using upsample please provide images with the same aspect ratio\n' + \
                      'src shape={0}, guide shape={1}, s={2}'.format(src.shape, guide.shape, s)
    assert np.max(src_sub) <= 1, \
        "src_sub should be float array of probability".format(np.max(src_sub), src_sub.dtype)
    radius_sub = int(np.round(radius / s))
    # guide_sub = downsample(guide, src_sub.shape, interp='nearest') / 255.
    guide_sub = staged_resize(guide)
    assert np.max(guide_sub) <= 1, "src_sub should be float array of probability".format\
        (np.max(src_sub), src_sub.dtype)

    return guide_sub, src_sub, radius_sub


def fast_guided_filter_color(guide, src, radius, eps, subsample_ratio):
    """
    Performs fast guided filter with color guided image.
    Based on matlab implementation provided by Kaiming He (kahe@microsoft.com)

    :param guide: RGB color image np.float scaled 0-1 or np.unit8 0-255
    :param src: Image to be filtered. single channel image (gray-scale / color-channel)
    :param radius: local window raius
    :param eps: regularization parameter
    :param subsample_ratio:int, must divide `radius`. Idealy also divides guide and src.
                                if guide and src have different size - than subsample_ratio is
                                guide.shape / src.shape.
    :return: gray-scale / single-channel image
    """
    guide = validate_input(guide)
    guide_sub, src_sub, radius_sub = get_subsampled_inputs(guide, src, radius, subsample_ratio)
    hei, wid = src_sub.shape
    normalizer = box_filter(np.ones(src_sub.shape, dtype=np.uint16), r=radius_sub)
    mean_guide_r = box_filter(guide_sub[:, :, 0], radius_sub) / normalizer
    mean_guide_g = box_filter(guide_sub[:, :, 1], radius_sub) / normalizer
    mean_guide_b = box_filter(guide_sub[:, :, 2], radius_sub) / normalizer

    mean_src = box_filter(src_sub, radius_sub) / normalizer

    joint_mean_r = box_filter(guide_sub[:, :, 0] * src_sub, radius_sub) / normalizer
    joint_mean_g = box_filter(guide_sub[:, :, 1] * src_sub, radius_sub) / normalizer
    joint_mean_b = box_filter(guide_sub[:, :, 2] * src_sub, radius_sub) / normalizer

    # covariance of (I, p) in each local patch.
    cov_r = joint_mean_r - mean_guide_r * mean_src
    cov_g = joint_mean_g - mean_guide_g * mean_src
    cov_b = joint_mean_b - mean_guide_b * mean_src

    # variance of I in each local patch: the matrix Sigma in Eqn (14).
    # Note the variance in each local patch is a 3x3 symmetric matrix:
    #           rr, rg, rb
    #   Sigma = rg, gg, gb
    #           rb, gb, bb
    var_rr = box_filter(guide_sub[:, :, 0] * guide_sub[:, :, 0], radius_sub) / normalizer
    var_rr -= mean_guide_r * mean_guide_r
    var_rg = box_filter(guide_sub[:, :, 0] * guide_sub[:, :, 1], radius_sub) / normalizer
    var_rg -= mean_guide_r * mean_guide_g
    var_rb = box_filter(guide_sub[:, :, 0] * guide_sub[:, :, 2], radius_sub) / normalizer
    var_rb -= mean_guide_r * mean_guide_b
    var_gg = box_filter(guide_sub[:, :, 1] * guide_sub[:, :, 1], radius_sub) / normalizer
    var_gg -= mean_guide_g * mean_guide_g
    var_gb = box_filter(guide_sub[:, :, 1] * guide_sub[:, :, 2], radius_sub) / normalizer
    var_gb -= mean_guide_g * mean_guide_b
    var_bb = box_filter(guide_sub[:, :, 2] * guide_sub[:, :, 2], radius_sub) / normalizer
    var_bb -= mean_guide_b * mean_guide_b

    a = np.zeros(shape=(hei, wid, 3))

    for y in range(hei):
        for x in range(wid):
            sigma = np.array([[eps + var_rr[y, x], var_rg[y, x], var_rb[y, x]],
                              [var_rg[y, x], eps + var_gg[y, x], var_gb[y, x]],
                              [var_rb[y, x], var_gb[y, x], eps + var_bb[y, x]]])

            cov = [cov_r[y, x], cov_g[y, x], cov_b[y, x]]
            a[y, x, :] = np.dot(cov, np.linalg.inv(sigma))
    shift = mean_src - a[:, :, 0] * mean_guide_r - a[:, :, 1] * mean_guide_g - a[:, :, 2] * \
                                                                               mean_guide_b
    # Eqn.(15) in the paper
    mean_a = np.zeros_like(a)
    mean_a[:, :, 0] = box_filter(a[:, :, 0], radius_sub) / normalizer
    mean_a[:, :, 1] = box_filter(a[:, :, 1], radius_sub) / normalizer
    mean_a[:, :, 2] = box_filter(a[:, :, 2], radius_sub) / normalizer
    mean_b = box_filter(shift, radius_sub) / normalizer
    mean_a = upsample(mean_a, guide.shape[:2], order=InterpulationOrders.bilinear,
                      preserve_range=True, mode='edge')
    mean_b = upsample(mean_b, guide.shape[:2], order=InterpulationOrders.bilinear,
                      preserve_range=True, mode='edge')
    return np.clip(np.sum(mean_a * guide, axis=2) + mean_b, 0, 1)


def fast_guided_filter(guide, src, radius, eps, subsample_ratio):
    """
    Performs fast guided filter with color guided image.
    Based on matlab implementation provided by Kaiming He (kahe@microsoft.com)

    :param guide: single channel image (gray-scale / color-channel
    :param src: Image to be filtered. single channel image (gray-scale / color-channel)
    :param radius: local window raius
    :param eps: regularization parameter
    :param subsample_ratio:int, must divide `radius`. Ideally also divides guide and src.
                                if guide and src have different size - than subsample_ratio is
                                guide.shape / src.shape.
    :return: gray-scale / single-channel image
    """
    guide = validate_input(guide)
    guide_sub, src_sub, radius_sub = get_subsampled_inputs(guide, src, radius, subsample_ratio)

    normalizer = box_filter(np.ones(src_sub.shape, dtype=np.uint16), r=radius_sub)

    mean_guide = box_filter(guide_sub, radius_sub) / normalizer
    mean_src = box_filter(src_sub, radius_sub) / normalizer

    joint_mean = box_filter(guide_sub * src_sub, radius_sub) / normalizer

    cov = joint_mean - mean_guide * mean_src
    mean_guide_sq = box_filter(guide_sub * guide_sub, radius_sub) / normalizer
    var_guide = mean_guide_sq - mean_guide * mean_guide
    scale = cov / (var_guide + eps)
    shift = mean_src - scale * mean_guide

    mean_scale = box_filter(scale, radius_sub) / normalizer
    mean_shift = box_filter(shift, radius_sub) / normalizer
    mean_scale = upsample(mean_scale, guide.shape, order=InterpulationOrders.bilinear,
                          preserve_range=True, mode='edge')
    mean_shift = upsample(mean_shift, guide.shape, order=InterpulationOrders.bilinear,
                          preserve_range=True, mode='edge')
    return np.clip(mean_scale * guide + mean_shift, 0, 1)
