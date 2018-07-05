import numpy as np
import cv2

from visualize.model_utils import *


class Model:

    def __init__(self, image, image_name):

        self.original_image = fix_image_size(image)

        self.depth_map, self.filtered_depth_map = [None, None]
        init_depth_maps(self, image_name)


        self.focal, self.coc_min, self.bg_sigma, self.bg_sigma, \
        self.bg_power, self.fg_power = [None] * 6

        init_blur_variables(self)

        self.segments_map, self.image_copy_BG, self.image_copy_FG, self.blur_weights = [None] * 4

        update_blur_maps(self)

        self.blur, self.blurred_image = [None] * 2
        # should work with dictionaries
        calc_blur(self)

        self.stable_viewables = ["original_image", "depth_map", "filtered_depth_map"]

        self.updating_viewables = ["segments_map", "image_copy_FG", "blur_weights",
                                   "blurred_image"]

        self.control_variables = ["focal", "bg_sigma", "fg_sigma", "coc_min",
                                  "bg_power", "fg_power"]

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
        #update_blur_maps(self)
        #calc_blur(self)
        ARF_blur(self)

class GFstudy:
    def __init__(self, image, image_name):

        self.original_image = fix_image_size(image)
        self.depth_map, self.filtered_depth_map = [None, None]

        init_depth_maps(self, image_name, radius=1, epsilon=0.01)

        radius = 2
        epsilon = 0.01
        self.new_depth_map = get_depth_maps(self.original_image, radius=radius,
                                                 epsilon=epsilon, depth_map=self.depth_map)[1]

        self.stable_viewables = ["original_image", "depth_map", "filtered_depth_map",
                                 "new_depth_map"]

        self.updating_viewables = []#"filtered_depth_map", "filtered_depth_map"]

        self.control_variables = []#"radius", "epsilon"]

    #def update_images(self):
    #    radius = int(self.radius["val"] * 100)
    #    epsilon = self.epsilon["val"]
    #    self.filtered_depth_map = get_depth_maps(self.original_image, radius=radius,
    #                                             epsilon=epsilon, depth_map=self.depth_map)

class ScaleModel(Model):

    def __init__(self, image, image_name):
        super().__init__(image, image_name)
        self.scaleable_image = fix_image_size(image, divisor=64)

        self.scaled_depth_maps, self.filtered_depth_maps = None, None
        init_scaled_depth_maps(self, image_name)

        self.depth_map0 = self.filtered_depth_maps[0]
        self.depth_map1 = self.filtered_depth_maps[1]
        self.depth_map2 = self.filtered_depth_maps[2]

        self.averaged_depth_map = np.average(self.filtered_depth_maps, axis=0,
                                             weights=[1/3, 1/3, 1/3])
        self.filtered_averaged_depth_map = get_depth_maps(self.scaleable_image, radius=RADIUS,
                                                      epsilon=EPSILON,
                                                      depth_map=self.averaged_depth_map)[1]


        self.stable_viewables.extend(["depth_map0", "depth_map1", "depth_map2",
                                      "averaged_depth_map", "filtered_averaged_depth_map"])
        #self.updating_viewables = []
        #self.control_variables = []

        #self.original_image = self.scaleable_image#staged_resize(self.original_image)#
        #self.filtered_depth_map = self.filtered_averaged_depth_map
        #set_to_range(self.depth_map)#self.filtered_averaged_depth_map#
        #super().update_images()

class SmallViewModel(Model):
    def __init__(self, image, image_name):

        image = staged_resize(fix_image_size(image))
        image_name
        super().__init__(image, image_name)
'''
        
        
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
'''