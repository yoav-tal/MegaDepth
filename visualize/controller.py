import sys

import cv2
import tkinter as tk
from tkinter import filedialog
#import visualize.model as model
from visualize import viewer
from visualize import model

class Control:

    def __init__(self, master):

        # Define master widget
        self.master = master

        # open file dialog
        path = '/Users/yoav/MegaDepth/images/eifel.jpeg'
        #path = filedialog.askopenfilename(initialdir = "/Users/yoav/MegaDepth/images")

        image_name = path.split('/')[-1].split('.')[0]
        print("image name:", image_name)
        # set up a model with the image
        #self.MODEL = model.Model(cv2.imread(path).astype(float) / 255.0, image_name)
        #self.MODEL = model.HazeModel(cv2.imread(path).astype(float) / 255.0, image_name)
        #self.MODEL = model.FlipModel(cv2.imread(path).astype(float) / 255.0, image_name)
        #self.MODEL = model.ARFModel(cv2.imread(path).astype(float) / 255.0, image_name)
        #self.MODEL = model.GFstudy(cv2.imread(path).astype(float) / 255.0, image_name)
        #self.MODEL = model.ScaleModel(cv2.imread(path).astype(float) / 255.0, image_name)
        self.MODEL = model.ThresholdingModel(cv2.imread(path).astype(float) / 255.0, image_name)


        # set up a viewer
        self.VIEWER = viewer.viewer(self.master)

        # set up window buttons
        self.VIEWER.add_frame("buttons")
        self.buttons = {}
        self.update_buttons()

        # set up sliders
        self.VIEWER.add_frame("sliders")
        self.sliders = {}
        self.update_sliders()


    def update_buttons(self):

        buttons_frame = self.VIEWER.frames["buttons"]

        for content in self.MODEL.stable_viewables + self.MODEL.updating_viewables:
            self.buttons[content] = tk.Button(master=buttons_frame, text=content,
                            command=self.call_init_image(getattr(self.MODEL, content), content))
            self.VIEWER.grid_button(self.buttons[content])

    def call_init_image(self, image, name):

        def init_image():
            self.VIEWER.init_image(image, name)

        return init_image


    def update_sliders(self):

        sliders_frame = self.VIEWER.frames["sliders"]

        for content in self.MODEL.control_variables:
            from_, to_ = getattr(self.MODEL, content)["from_"], \
                         getattr(self.MODEL, content)["to_"]
            self.sliders[content] = tk.Scale(master=sliders_frame, orient=tk.HORIZONTAL,
                                                 resolution= 0.01, from_=from_, to_=to_,
                                                 label=content,
                                                 command=self.call_image_update(content))
            self.sliders[content].set(getattr(self.MODEL, content)["val"])
            self.VIEWER.grid_slider(self.sliders[content])

    def call_image_update(self, name):
        def image_update(val):
            getattr(self.MODEL, name)["val"] = float(val)
            self.MODEL.update_images()
            for content in self.MODEL.updating_viewables:
                if content in self.VIEWER.image_panels.keys():
                    image = getattr(self.MODEL, content)
                    panel = self.VIEWER.image_panels[content]
                    self.VIEWER.view_image(image, panel)

        return image_update


"""


        # view image
        #self.VIEWER.view(self.MODEL.update_image())

        image, depth_map = self.MODEL.get_image_and_depth()
        self.VIEWER.view_orig_and_depth(image, depth_map)

        # set slider functions (self.slider_functions is a list of functions, function [i] adjusts feature [i])
        #self.slider_functions = self.set_slider_functions()

        # set up buttons (buttons is a list of buttons. button [i] initiates a slider that adjusts feature [i])
        #self.buttons = self.init_buttons()
        #self.VIEWER.view(self.buttons)

        # bind image frame to keep aspect ratio
        #self.binded_frame = self.VIEWER.get_binded_frame()
        #self.bind_frame(self.binded_frame, self.MODEL.get_aspect_ratio())


    


    def get_button_command(self, feature_num):

        # define the button function: init a slider and view
        def init_slider():
            # get master for sliders
            frame = self.VIEWER.buttons_frame

            # get parameters
            slider_params = model.features_parameters[feature_num]['slider_params']

            # set up slider
            slider = tk.Scale(master = frame, **slider_params, command= self.slider_functions[feature_num])
            slider.set(model.features_parameters[feature_num]['val'])

            # view slider
            self.VIEWER.view(slider)

        return init_slider



    def set_slider_functions(self):
        n_features = len(model.features_parameters)
        return [self.slider_function(i) for i in range(n_features)]


    def slider_function(self, feature_num):
        # define slider function: update feature parameter (in model list of features), update image and view
        def update_val_and_image(val):
            # update value
            model.features_parameters[feature_num]['val'] = float(val)

            # update image
            image = self.MODEL.update_image()

            # view image
            self.VIEWER.view(image)

        return update_val_and_image



    def bind_frame(self, frame, aspect_ratio):

        # define a funciton that calculates desired size according to configuration event, upddates size parameters
        # update image and view new image

        def enforce_aspect_ratio(event):
            # when the pad window resizes, fit the content into it, either by fixing the width or the height and then
            # adjusting the height or width based on the aspect ratio.

            # start by using the width as the controlling dimension
            desired_width = event.width
            desired_height = int(event.width / aspect_ratio)

            # if the window is too tall to fit, use the height as
            # the controlling dimension
            if desired_height > event.height:
                desired_height = event.height
                desired_width = int(event.height * aspect_ratio)

            # update size in model list
            model.size_parameters['width'] = desired_width
            model.size_parameters['height']= desired_height

            # update image
            image = self.MODEL.update_image()

            # view image
            self.VIEWER.view(image, desired_width, desired_height)

        frame.bind("<Configure>", enforce_aspect_ratio)

"""


if __name__ == '__main__':

    root = tk.Tk()
    CONTROL = Control(root)
    root.mainloop()
