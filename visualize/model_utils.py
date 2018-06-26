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

    filtered_depth_map = set_to_range(filtered_depth_map, min_=np.min(depth_map),
                                     max_=np.max(depth_map))

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


def set_to_range(array, min_=0, max_=1):

    return (array - np.min(array)) * (max_ - min_) / (np.max(array) - np.min(array)) + min_


def staged_resize(img, max_long_side=MAX_LONG_SIDE, size_divisor=NETWORK_SIZE_DIVISOR):

    while max(img.shape[:2]) > max_long_side and np.mod(img.shape[0]/2, size_divisor) == 0 and \
            np.mod(img.shape[1]/2, size_divisor) == 0:

        img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    return img


def update_blur_maps(model):
    image = model.original_image
    depth_map = model.filtered_depth_map

    focal = model.focal["val"]
    coc_min = model.coc_min["val"]
    bg_power = model.bg_power["val"]
    fg_power = model.fg_power["val"]

    focus_start = focal / (1 + coc_min)
    focus_end = focal / (1 - coc_min)

    segments = 1 + (-1 * (depth_map < focus_start) + 2 * (depth_map > focus_end))
    # in segments: 0 - foreground ; 1 - focus ; 3 - background
    model.segments_map = segments

    segments = np.transpose(np.repeat(segments[:, np.newaxis], 3, axis=1), [0, 2, 1])

    img_copy = np.copy(image)

    np.copyto(img_copy, 0, where=segments == 1)

    model.image_copy = img_copy

    depth_map = np.transpose(np.repeat(depth_map[:, np.newaxis], 3, axis=1), [0, 2, 1])

    max_depth = np.max(depth_map)

    background_blur_curve = 1 - 1 / ((depth_map - focus_end + 1) ** bg_power)
    background_blur_curve = background_blur_curve / (1 - 1 / ((max_depth - focus_end +1)
                                                              ** bg_power))

    foreground_blur_curve = 1 - 1 / ((1 + focus_start - depth_map) ** fg_power)
    foreground_blur_curve = foreground_blur_curve / (1 - 1 / ((1 + focus_start) ** fg_power))

    #  פרמטרים שאפשר להוסיף למשתמש: שליטה בסיגמה (כמה טשטוש) ובחזקה (מהירות ההתקדמות של הטשטוש)
    # כל אחד מאלה אפשר להפריד לחלק קדמי ואחורי

    blur_weights = np.zeros(shape=image.shape, dtype=np.float32)
    np.copyto(blur_weights, background_blur_curve, where=segments == 3)
    np.copyto(blur_weights, foreground_blur_curve, where=segments == 0)

    model.blur_weights = blur_weights


def init_blur_variables(model):
    min_val = np.min(model.filtered_depth_map)
    max_val = np.max(model.filtered_depth_map)

    model.focal = {"val": (max_val - min_val)/2, "from_": min_val, "to_": max_val}
    model.coc_min = {"val": .5, "from_": 0.01, "to_": 1 - 0.01}
    model.bg_sigma = {"val": 5, "from_": 0.5, "to_": 15}
    model.fg_sigma = {"val": 5, "from_": 0.5, "to_": 15}
    model.bg_power = {"val": 3, "from_": 0.05, "to_":15}
    model.fg_power = {"val": 3, "from_": 0.05, "to_":15}


def calc_blur(model):

    original_image = model.original_image
    img_copy = model.image_copy
    blur_weights = model.blur_weights
    segments = model.segments_map
    bg_sigma = model.bg_sigma["val"]
    fg_sigma = model.fg_sigma["val"]
    bg_ksize = 2 * round(bg_sigma) + 1
    fg_ksize = 2 * round(fg_sigma) + 1


    #blur_weights = np.transpose(np.repeat(blur_weights[:, np.newaxis], 3, axis=1), [0, 2, 1])
    segments = np.transpose(np.repeat(segments[:, np.newaxis], 3, axis=1), [0, 2, 1])

    bg_blur = cv2.GaussianBlur(img_copy, (bg_ksize, bg_ksize), bg_sigma)
    fg_blur = cv2.GaussianBlur(img_copy, (fg_ksize, fg_ksize), fg_sigma)

    binary_segments = np.float32(np.abs(np.sign(segments - 1)))

    bg_normalization = cv2.GaussianBlur(binary_segments, (bg_ksize, bg_ksize), bg_sigma)
    fg_normalization = cv2.GaussianBlur(binary_segments, (fg_ksize, fg_ksize), fg_sigma)

    np.copyto(bg_normalization, 1e-5, where= bg_normalization == 0)
    np.copyto(fg_normalization, 1e-5, where= fg_normalization == 0)

    model.blur = np.copy(img_copy)

    np.copyto(model.blur, bg_blur / bg_normalization, where=segments == 3)
    np.copyto(model.blur, fg_blur / fg_normalization, where=segments == 0)

    model.blurred_image = (blur_weights * model.blur + (1 - blur_weights) * original_image)
