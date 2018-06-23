import cv2
import numpy as np
from argparse import Namespace

from visualize.get_depth_mask import get_depth_mask as get_depth
from guidedFilter import fast_guided_filter_color as guided_filter

# size hyperparameters
PREF_LONG_SIDE = 512
MAX_LONG_SIDE = 640
NETWORK_SIZE_DIVISOR = 16

# filter parameters
RADIUS = 5
EPSILON = 1e-2

# paths
DEPTH_FOLDER = "/Users/yoav/MegaDepth/depths/"

#blur
COC_AT_INF = 5


def calc_size(image_shape, target, divisor):
    # Finds new size with nearest aspect ratio such that after downsampling by factors of 2 the
    # long egde is near target and both edges are a multiple of divisor.

    assert len(image_shape) == 2

    long = np.max(image_shape)
    short = np.min(image_shape)
    aspect_ratio = long / short

    target = np.floor(np.log2(target))
    divisor = np.floor(np.log2(divisor))

    approx1 = np.floor(np.log2(long))

    num_divisions = approx1 - target + 1

    approx2 = (long - 2 ** approx1) // 2 ** (num_divisions + divisor)

    new_long = 2 ** approx1 + approx2 * (2 ** (num_divisions + divisor))

    short_by_aspect = new_long / aspect_ratio

    approx3 = short_by_aspect // 2 ** (num_divisions + divisor)

    new_short_candidates = np.array([approx3 * (2 ** (num_divisions + divisor)), (approx3 + 1) *
                                    (2 ** (num_divisions + divisor))])

    aspect_ratio_candiates = long / new_short_candidates

    best_aspect_ratio = np.argmin(np.abs(aspect_ratio-aspect_ratio_candiates))

    new_short = new_short_candidates[best_aspect_ratio]

    new_long = np.rint(new_long).astype(int)
    new_short = np.rint(new_short).astype(int)

    if np.argmax(image_shape) == 0:
        return new_long, new_short
    else:
        return new_short, new_long


def fix_image_size(image):
    Nrows, Ncolumns = calc_size(image.shape[:2], target=PREF_LONG_SIDE,
                                divisor=NETWORK_SIZE_DIVISOR)
    return cv2.resize(image, (Ncolumns, Nrows))


def get_depth_maps(image):

    image_sub = staged_resize(image)

    depth_map = get_depth(image_sub)

    size_ratio = image.shape[0] / image_sub.shape[0]

    filtered_depth_map = guided_filter(guide=image, src=depth_map / np.max(depth_map),
                                        radius=RADIUS * size_ratio, eps=EPSILON, subsample_ratio=1)

    filtered_depth_map = set_to_range(filtered_depth_map, min=np.min(depth_map),
                                     max=np.max(depth_map))

    return depth_map, filtered_depth_map

def save_depth_maps(image_name, depth_map, filtered_depth_map):

    depth_path = DEPTH_FOLDER + image_name + "_DEPTH.npy"
    filtered_depth_path = DEPTH_FOLDER + image_name + "_DEPTH_filtered.npy"

    np.save(depth_path, depth_map)
    np.save(filtered_depth_path, filtered_depth_map)


def load_depth_maps(image_name):

    depth_path = DEPTH_FOLDER + image_name + "_DEPTH.npy"
    filtered_depth_path = DEPTH_FOLDER + image_name + "_DEPTH_filtered.npy"

    depth_map = np.load(depth_path)
    filtered_depth_map = np.load(filtered_depth_path)

    return depth_map, filtered_depth_map


def set_to_range(array, min=0, max=1):

    return (array - np.min(array)) * (max - min) / (np.max(array) - np.min(array)) + min

def staged_resize(img, max_long_side=MAX_LONG_SIDE, size_divisor=NETWORK_SIZE_DIVISOR):

    while max(img.shape[:2]) > max_long_side and np.mod(img.shape[0]/2, size_divisor) == 0 and \
            np.mod(img.shape[1]/2, size_divisor) == 0:

        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    return img

def update_blur_maps(focal, coc_min, image, depth_map):

    CoC_map = COC_AT_INF * abs(1 - focal / (depth_map + 1e-5))

    focus_start = (COC_AT_INF * focal) / (COC_AT_INF + coc_min)
    focus_end = (COC_AT_INF * focal) / (COC_AT_INF - coc_min)

    segments = 1 + (-1 * (depth_map < focus_start) + 2 * (depth_map > focus_end))
    # in segments: 0 - foreground ; 1 - focus ; 3 - background
    segments_map = segments

    segments = np.transpose(np.repeat(segments[:, np.newaxis], 3, axis=1), [0, 2, 1])

    img_copy = np.copy(image)

    np.copyto(img_copy, 0, where=segments == 1)

    return CoC_map, segments_map, img_copy


def init_blur_variables(min_val, max_val):
    # should be computed from depth map values

    focal = {"val": (max_val - min_val)/2, "from_": min_val, "to_": max_val}
    coc_min = {"val": 1 , "from_": 0.05, "to_": COC_AT_INF - 0.05}

    return focal, coc_min
