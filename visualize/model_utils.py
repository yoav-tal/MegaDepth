from time import time

import cv2
import numpy as np
from argparse import Namespace

#from visualize.get_depth_mask import get_depth_mask as get_depth
from visualize.get_depth_mask import get_inverse_mask as get_depth

from guidedFilter import fast_guided_filter_color as guided_filter
from skimage.transform import resize as upsample


# size hyperparameters
PREF_LONG_SIDE = 512
NETWORK_MAX_LONG_SIDE = 640
NETWORK_SIZE_DIVISOR = 16
VIEW_MAX_LONG_SIZE = 1200

# filter parameters
RADIUS = 5
EPSILON = 1e-2

# paths
DEPTH_FOLDER = "/Users/yoav/MegaDepth/depths/"

DELTA = 1e-5

def init_depth_maps(model, image_name, radius=RADIUS, epsilon=EPSILON):

    try:
        model.depth_map, model.filtered_depth_map = load_depth_maps(image_name)
        assert model.filtered_depth_map.shape == model.original_image.shape[:2]

    except (AssertionError, FileNotFoundError):
        model.depth_map, model.filtered_depth_map = get_depth_maps(model.original_image,
                                                                   radius=radius, epsilon=epsilon)
        save_depth_maps(image_name, model.depth_map, model.filtered_depth_map)

    except AttributeError:
        model.filtered_depth_map = get_depth_maps(model.original_image, radius=radius,
                                                  epsilon=epsilon, depth_map=model.depth_map)[1]
        save_depth_maps(image_name, model.depth_map, model.filtered_depth_map)

    model.depth_values = model.filtered_depth_map * np.max(model.depth_map)

def init_scaled_depth_maps(model, image_name):

    image_name = image_name + "_SCALED"

    try:
        model.scaled_depth_maps, model.filtered_depth_maps = load_depth_maps(image_name)
    except FileNotFoundError:
        model.scaled_depth_maps, model.filtered_depth_maps = get_scaled_depth_maps(
            model.scaleable_image)
        save_depth_maps(image_name, model.scaled_depth_maps, model.filtered_depth_maps)


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


def fix_image_size(image, target=PREF_LONG_SIDE, divisor=NETWORK_SIZE_DIVISOR):
    Nrows, Ncolumns = calc_size(image.shape[:2], target=target, divisor=divisor)
    image = cv2.resize(image, (Ncolumns, Nrows))

    return staged_resize(image, max_long_side=VIEW_MAX_LONG_SIZE)


def get_depth_maps(image, radius=RADIUS, epsilon=EPSILON, depth_map=None):

    image_sub = staged_resize(image)

    if depth_map is None:
        init = time()
        depth_map = get_depth(image_sub)
        elapsed = time() - init
        print("image size:", image.shape[:2], ". Depth map time:", elapsed)

    size_ratio = image.shape[0] / image_sub.shape[0]

    init = time()
    filtered_depth_map = guided_filter(guide=image, src=depth_map / np.max(depth_map),
                                        radius=radius * size_ratio, eps=epsilon, subsample_ratio=1)
    elapsed = time() - init
    print("image size:", filtered_depth_map.shape, ". Filter time:", elapsed)
    print("calulated with radius:", radius, " ; epsilon:", epsilon)

    print("depth range:", depth_map.max(), depth_map.min())
    print("filtered depth range:", filtered_depth_map.max(), filtered_depth_map.min())

    #depth_map = upsample(depth_map/depth_map.max(), output_shape=image_sub.shape[:2])
    return depth_map, filtered_depth_map

def get_scaled_depth_maps(image, radius=RADIUS, epsilon=EPSILON):

    image_sub = staged_resize(image)

    depth_maps = []
    filtered_depth_maps = []

    for scale in range(3):
        img_scaled = staged_resize(image_sub, times=scale)
        depth_scaled = get_depth(img_scaled)

        depth_maps.append(depth_scaled)
        filtered_depth_maps.append(guided_filter(guide=image_sub, src=depth_scaled / np.max(
            depth_scaled), radius=radius * (2 ** scale), eps=epsilon, subsample_ratio=1))

    return depth_maps, filtered_depth_maps


def save_depth_maps(image_name, depth_map, filtered_depth_map):

    depth_path = DEPTH_FOLDER + image_name + "_DEPTH.npy"
    filtered_depth_path = DEPTH_FOLDER + image_name + "_DEPTH_filtered.npy"

    np.save(depth_path, depth_map)
    np.save(filtered_depth_path, filtered_depth_map)


def load_depth_maps(image_name):

    depth_path = DEPTH_FOLDER + image_name + "_DEPTH.npy"
    filtered_depth_path = DEPTH_FOLDER + image_name + "_DEPTH_filtered.npy"

    depth_map = np.load(depth_path)

    try:
        filtered_depth_map = np.load(filtered_depth_path)
    except FileNotFoundError:
        return depth_map, None

    return depth_map, filtered_depth_map


def set_to_range(array, min_=0.0, max_=1.0):

    return (array - np.min(array)) * (max_ - min_) / (np.max(array) - np.min(array)) + min_


def staged_resize(img, times=None, max_long_side=NETWORK_MAX_LONG_SIDE, \
                                           size_divisor=NETWORK_SIZE_DIVISOR):

    if times:
        for i in range(times):
            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    else:
        while max(img.shape[:2]) > max_long_side and np.mod(img.shape[0]/2, size_divisor) == 0 and \
            np.mod(img.shape[1]/2, size_divisor) == 0:

            img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)

    return img

def calc_segments(model):
    inv_depth_map = model.filtered_depth_map
    focal = model.focal["val"]
    coc_min = model.coc_min["val"]
    coc_at_inf = 1 #model.coc_at_inf["val"]

    model.coc_map = coc_at_inf * abs(1 - inv_depth_map / focal)
    model.inv_focus_start = min((coc_at_inf + coc_min) * focal / coc_at_inf, 1 - 1e-6)
    model.inv_focus_end = max((coc_at_inf - coc_min)  * focal / coc_at_inf, 1e-6)

    segments = 1 + (-1 * (inv_depth_map > model.inv_focus_start) + 2 * (inv_depth_map <
                                                                      model.inv_focus_end))
    # in segments: 0 - foreground ; 1 - focus ; 3 - background

    model.segments_map = copy_to_3_channels(segments)

def copy_to_3_channels(matrix):
    return np.transpose(np.repeat(matrix[:, np.newaxis], 3, axis=1), [0, 2, 1])


def init_blur_variables(model):
    min_val = np.min(model.filtered_depth_map)
    max_val = np.max(model.filtered_depth_map)

    model.focal = {"val": 0.5, "from_": min_val, "to_": max_val}
    model.coc_min = {"val": .5, "from_": 0.01, "to_": 1 - 0.01}
    #model.coc_at_inf = {"val": 1, "from_": 1, "to_": 8}

    model.bg_sigma = {"val": 5, "from_": 0.5, "to_": 15}
    model.fg_sigma = {"val": 5, "from_": 0.5, "to_": 15}
    model.bg_power = {"val": 3, "from_": 0.05, "to_":15}
    model.fg_power = {"val": 3, "from_": 0.05, "to_":15}
    #model.blur_sigma = {"val": 5, "from_": 0.5, "to_": 15}

def update_blur_maps(model):
    image = model.original_image
    inv_depth_map = model.filtered_depth_map

    bg_power = model.bg_power["val"]
    fg_power = model.fg_power["val"]

    calc_segments(model)

    segments = model.segments_map
    inv_focus_start = model.inv_focus_start
    inv_focus_end = model.inv_focus_end

    #segments = np.transpose(np.repeat(segments[:, np.newaxis], 3, axis=1), [0, 2, 1])

    depth_map = copy_to_3_channels(inv_depth_map)

    max_depth = np.max(depth_map)

    foreground_blur_weights = 1 - (depth_map - inv_focus_start) / (1 - inv_focus_start)
    #foreground_blur_curve = 1 - 1 / ((depth_map - inv_focus_start + 1) ** fg_power)
    #foreground_blur_curve = foreground_blur_curve / (1 - 1 / ((max_depth - inv_focus_start +1)
    #                                                         ** fg_power))

    background_blur_weights = depth_map / inv_focus_end
    #background_blur_curve: 1 - 1 / ((1 + inv_focus_end - depth_map) ** bg_power)
    #background_blur_curve = background_blur_curve / (1 - 1 / ((1 + inv_focus_end) ** bg_power))

    blur_weights = np.zeros(shape=image.shape, dtype=np.float32)
    np.copyto(blur_weights, background_blur_weights ** bg_power, where=segments == 3)
    np.copyto(blur_weights, foreground_blur_weights ** fg_power, where=segments == 0)
    np.copyto(blur_weights, 1, where=segments==1)
    model.blur_weights = blur_weights




def calc_blur(model):

    image = model.original_image
    segments = model.segments_map

    bg_blur = calc_bg_blur(image=image, segments=segments,
                                blur_weights = model.blur_weights,
                                blur_sigma = model.bg_sigma["val"], segment_indicator=3)
    fg_blur = calc_fg_blur(image=image, blur_weights = model.blur_weights,
                                blur_sigma = model.fg_sigma["val"], model=model)

    model.blurred_image = model.original_image.copy()
    np.copyto(model.blurred_image, bg_blur, where=model.segments_map == 3)
    np.copyto(model.blurred_image, fg_blur, where=model.segments_map < 3)

def calc_bg_blur(image, segments, blur_weights, blur_sigma, segment_indicator):

    img_copy = np.copy(image)
    np.copyto(img_copy, 0, where=segments != segment_indicator)

    ksize = 2 * int(round(blur_sigma)) + 1

    region_blur = cv2.GaussianBlur(img_copy, (ksize, ksize), blur_sigma)

    binary_segments = np.zeros_like(img_copy)
    np.copyto(binary_segments, 1, where=segments == segment_indicator)

    normalization = cv2.GaussianBlur(binary_segments, (ksize, ksize), blur_sigma)

    np.copyto(normalization, 1e-5, where=normalization < 1e-5)

    normalized_region_blur = region_blur / normalization

    combined_blur = np.zeros_like(region_blur)
    np.copyto(combined_blur, blur_weights * image +
              (1 - blur_weights) * normalized_region_blur,
              where=segments ==segment_indicator)

    return combined_blur

def calc_fg_blur(image, blur_weights, blur_sigma, model):

    img_copy = np.copy(image)

    ksize = 2 * int(round(blur_sigma)) + 1

    blur = cv2.GaussianBlur(img_copy, (ksize, ksize), blur_sigma)

    blur_weights = cv2.GaussianBlur(blur_weights, (2*ksize-1, 2*ksize-1), 2*blur_sigma)

    combined_blur = blur_weights * image + (1 - blur_weights) * blur

    model.blurred_weights = blur_weights


    return combined_blur

def init_haze_variables(model):
    model.ambient = {"val": 0.5, "from_": 0.01, "to_": 1}
    model.beta = {"val": 1, "from_": 0.01, "to_": 3}
    model.end_haze = {"val": 0.5, "from_": 0.01, "to_": 1}

def calc_haze(model):
    ambient = model.ambient["val"]
    beta = model.beta["val"]
    end_haze = model.end_haze["val"]

    original_image = model.original_image # staged_resize(model.original_image)
    depth_map = model.filtered_depth_map + 1e-5 #set_to_range(model.depth_map) #


    haze_weights = np.minimum(np.exp(-(beta * end_haze) / depth_map) / np.exp(-beta) , 1) ** 2
    haze_weights = copy_to_3_channels(haze_weights)
    model.haze_image = haze_weights * original_image + (1-haze_weights) * ambient


def rot90(img):
    return(np.rot90(img, 1, (0,1)))

def rotNEG90(img):
    return np.rot90(img, -1, (0,1))

implemented_transformations = {"fliplr": [np.fliplr, np.fliplr], "flipud": [np.flipud,np.flipud],
                               "rot90": [rot90, rotNEG90], "rotNEG90":[rotNEG90, rot90]}


def get_flip(model, image_name, transformation):

    if not transformation in implemented_transformations.keys():
        raise ValueError("this transformation is not implemented")

    func = implemented_transformations[transformation][0]
    inv_func = implemented_transformations[transformation][1]

    name = image_name + "_" + transformation

    trans_image = func(model.original_image).copy()

    try:
        depth_map, filtered_depth_map = load_depth_maps(name)
        assert filtered_depth_map.shape == trans_image.shape[:2]
    except (AssertionError, FileNotFoundError):
        depth_map, filtered_depth_map = get_depth_maps(trans_image)
        save_depth_maps(name, depth_map, filtered_depth_map)


    depth_straight = inv_func(depth_map)
    filt_depth_straight = inv_func(filtered_depth_map)
    depth_diff = model.filtered_depth_map - filt_depth_straight
    depth_diff = set_to_range(depth_diff)

    setattr(model, "image_" + transformation, trans_image)
    setattr(model, "depth_" + transformation, depth_map)
    setattr(model, "filtered_depth_" + transformation, filtered_depth_map)
    setattr(model, "depth_" + transformation + "_straight", depth_straight)
    setattr(model, "filt_depth_" + transformation + "_straight", filt_depth_straight)
    setattr(model, "depth_diff_" + transformation, depth_diff)

    model.stable_viewables.extend(["image_" + transformation, "depth_" + transformation,
                                   "filtered_depth_" + transformation,
                                   "depth_" + transformation + "_straight",
                                   "filt_depth_" + transformation + "_straight",
                                   "depth_diff_" + transformation])

def ARF_blur(model):

    img = model.original_image
    n_iter = int(model.n_iter["val"] * 100)
    n_rows, n_columns = img.shape[:2]

    calc_segments(model)
    segments = model.segments_map
    coc_map = model.coc_map

    cases_horizontal = np.abs(np.diff(segments, n=1, axis=1))
    cases_vertical = np.abs(np.diff(segments, n=1, axis=0))
    # in cases: 0 - case 1 ; 1%2 - case 2 or 4 ; 2 - case 3

    weights_horizontal = get_weights(coc_map[:, 1:], coc_map[:, :-1], cases_horizontal, DELTA=DELTA)
    weights_vertical = get_weights(coc_map[1:, :], coc_map[:-1, :], cases_vertical, DELTA=DELTA)

    weights_horizontal = np.repeat(weights_horizontal[:, np.newaxis], 3, axis=1)
    weights_vertical = copy_to_3_channels(weights_vertical)

    for i in range(n_iter):

        image_lr_blur = horizontal_blur(np.array(img, copy=True), weights_horizontal, n_columns - 1)
        image_rl_blur = horizontal_blur(np.fliplr(np.array(img, copy=True)),
                                        np.fliplr(weights_horizontal), n_columns - 1)

        image_ud_blur = vertical_blur(np.array(img, copy=True), weights_vertical, n_rows - 1)
        image_du_blur = vertical_blur(np.flipud(np.array(img, copy=True)),
                                      np.flipud(weights_vertical), n_rows - 1)

        img = 0.25*(image_lr_blur + np.fliplr(image_rl_blur) + image_ud_blur +
                    np.flipud(image_du_blur))

    model.ARF_blur = img
    model.segments_map = segments


def get_weights(coc_map1, coc_map2, cases, DELTA=1e-5):
    pre_weights = 0.5 * (coc_map1 + coc_map2)
    np.copyto(dst=pre_weights, src=np.maximum(coc_map1, coc_map2), where=(np.mod(
        cases, 2) == 1))  # max should be taken over a neighborhood
    np.copyto(dst=pre_weights, src=np.minimum(coc_map1, coc_map2),
              where=(cases == 2))

    weights = np.exp(-1/pre_weights, where=pre_weights >= DELTA)
    np.place(arr=weights, mask=pre_weights < DELTA, vals=0)

    return weights

def horizontal_blur(img, weights, N):
    for i in range(N-1):
        img[:, i+1] = np.multiply(1-weights[:, :, i], img[:, i+1]) + np.multiply(weights[:, :, i],
                                                                    img[:, i])

    return img

def vertical_blur(img, weights, N):
    for i in range(N-1):
        img[i+1, :] = np.multiply(1-weights[i, :, :], img[i+1, :]) + np.multiply(weights[i, :, :],
                                                                    img[i, :])

    return img

def apply_threshold(image, depth_map):

    u_depth_map = np.uint8(depth_map**2 * 255)

    FG = cv2.threshold(u_depth_map, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    np.copyto(depth_map, 0, where=FG==1)

    #u_depth_map = np.uint8((depth_map/depth_map.max()) * 255)
    u_depth_map = np.uint8(((depth_map)**2) * 255)

    hist_vals = np.histogram(u_depth_map, bins=256)[0]

    val = np.argmax(hist_vals[1:])

    np.copyto(u_depth_map, val, where=FG==1)


    MG = cv2.threshold(u_depth_map, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    np.copyto(MG, 0, where=FG==1)
    #np.copyto(depth_map, np.uint8(val2), where=thresholded2 == 1)
    #np.copyto(depth_map, 0, where=thresholded2 == 1)

    BG = np.ones(shape=depth_map.shape) - FG - MG
    #np.copyto(background, 0, where=np.logical_or(foreground == 1, background == 1))

    #val3, thresholded3 = cv2.threshold(depth_map, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    foreground = np.zeros_like(image)
    np.copyto(foreground, image, where=copy_to_3_channels(FG) == 1)

    midground = np.zeros_like(image)
    np.copyto(midground, image, where=copy_to_3_channels(MG) == 1)

    background = np.zeros_like(image)
    np.copyto(background, image, where=copy_to_3_channels(BG) == 1)

    return foreground, midground, background

# 1. set forward to val / set forward to 0  - same
# 2. set forward to 0 without normalization in 459 - same as 1. not good e.g. eifel
# 3. forward to 0, square w/o or w/ normalization - better
# 4. fw to val, sq w/ norm - same as 3
# 5. fw to val sq w/o norm - better (compare eifel on midground)