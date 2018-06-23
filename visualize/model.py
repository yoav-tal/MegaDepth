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

        self.focal, self.coc_min = init_blur_variables(np.min(self.filtered_depth_map),
                                                       np.max(self.filtered_depth_map))

        self.coc_map, self.segments_map, self.image_copy = update_blur_maps(self.focal["val"],
                                                                            self.coc_min["val"],
                                                                            self.original_image,
                                                                            self.filtered_depth_map)

        self.stable_viewables = ["original_image", "depth_map", "filtered_depth_map"]
        self.updating_viewables = ["coc_map", "segments_map", "image_copy"]

        self.blur_variables = ["focal", "coc_min"]

    def update_images(self):
        self.coc_map, self.segments_map, self.image_copy = update_blur_maps(self.focal["val"],
                                                                            self.coc_min["val"],
                                                                            self.original_image,
                                                                            self.filtered_depth_map)
