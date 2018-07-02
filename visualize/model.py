import numpy as np
import cv2

from visualize.model_utils import *


class Model:

    def __init__(self, image, image_name):

        self.original_image = fix_image_size(image)

        try:
            self.depth_map, self.filtered_depth_map = load_depth_maps(image_name)
            assert self.filtered_depth_map.shape == self.original_image.shape[:2]

        except (AssertionError, FileNotFoundError):
            self.depth_map, self.filtered_depth_map = get_depth_maps(self.original_image)
            save_depth_maps(image_name, self.depth_map, self.filtered_depth_map)

        self.focal, self.coc_min, self.bg_sigma, self.fg_sigma, self.bg_power, \
        self.fg_power = [None] * 6

        init_blur_variables(self)

        self.segments_map, self.image_copy, self.blur_weights = [None] * 3

        update_blur_maps(self)

        self.blur, self.blurred_image = [None] * 2
        # should work with dictionaries
        calc_blur(self)

        self.stable_viewables = ["original_image", "depth_map", "filtered_depth_map"]

        self.updating_viewables = ["segments_map", "image_copy", "blur", "blur_weights",
                                   "blurred_image"]

        self.control_variables = ["focal", "bg_sigma", "bg_power",
                               "coc_min", "fg_sigma", "fg_power"]

    def update_images(self):
        update_blur_maps(self)
        calc_blur(self)

class HazeModel(Model):

    def __init__(self, image, image_name):
        super().__init__(image, image_name)

        self.ambient, self.beta, self.end_haze= [None] * 3

        init_haze_variables(self)

        self.haze_image = None

        calc_haze(self)

        self.stable_viewables = ["original_image", "filtered_depth_map", "depth_map"]

        self.updating_viewables = ["haze_image"]

        self.control_variables = ["ambient", "beta", "end_haze"]

    def update_images(self):
        calc_haze(self)


class FlipModel(Model):
    def __init__(self, image, image_name):
        super().__init__(image, image_name)

        self.updating_viewables = []
        self.control_variables = []

        get_flip(self, image_name, "flipud")
        get_flip(self, image_name, "fliplr")
        get_flip(self, image_name, "rot90")
        get_flip(self, image_name, "rotNEG90")


class ARFModel(Model):
    def __init__(self, image, image_name):
        super().__init__(image, image_name)
        self.n_iter = {"val": .01, "from_": .01, "to_": .08}
        self.ARF_blur = None
        ARF_blur(self)
        self.control_variables.extend(["coc_at_inf", "n_iter"])
        self.updating_viewables.append("ARF_blur")

    def update_images(self):
        update_blur_maps(self)
        calc_blur(self)
        ARF_blur(self)
