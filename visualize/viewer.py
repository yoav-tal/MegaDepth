import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk

from visualize.viewer_utils import *


class viewer:

    def __init__(self, master):

        # Define master widget
        self.master = master
        self.master.title("Depth viewer")

        self.frames = {}
        self.image_panels = {}

        self.num_frames = -1
        self.num_buttons = -1
        self.num_sliders = -1

    def add_frame(self, name, **kwargs):
        self.num_frames += 1
        self.frames[name] = tk.Frame(self.master)
        row, column = calc_frame_grid(self.num_frames, name)
        print("frame", self.num_frames, "row", row, "column", column)
        self.frames[name].grid(row=row, column=column)

    def grid_button(self, button):
        self.num_buttons += 1
        row, column = calc_button_grid(self.num_buttons)
        button.grid(row=row, column=column)

    def grid_slider(self, slider):
        self.num_sliders += 1
        row, column = calc_button_grid(self.num_sliders)
        slider.grid(row=row, column=column)

    def init_image(self, img, name):

        if not name in self.frames.keys():
            self.add_frame(name)
            self.image_panels[name] = tk.Label(self.frames[name])
            self.image_panels[name].grid()

            self.view_image(img, self.image_panels[name])

    def view_image(self, image, panel):
         # Change to PIL format
         image = cv2ImageTk(image)

         # update widget
         panel.configure(image=image)
         panel.image = image






"""
        # set up containers
        #self.buttons_frame = tk.Frame(self.master)
        #self.pad_frame = tk.Frame(self.master, bg='blue')
        self.image_frame = tk.Frame(self.master,bg='blue')
        self.depth_frame = tk.Frame(self.master, bg='blue')

        # Initiate widgets
        self.image_panel = tk.Label(self.image_frame)
        self.image_panel.grid()

        self.depth_panel = tk.Label(self.depth_frame)
        self.depth_panel.grid()

        # layout containers
        #self.buttons_frame.grid(row=0, column=0)
        #self.pad_frame.grid(row=1, column=0, sticky='NWSE')
        self.image_frame.grid(row=0, column=0)
        self.depth_frame.grid(row=0, column=1)

        # define window movements
        #self.master.rowconfigure(1, weight=1)
        #self.master.columnconfigure(0, weight=1)

    # view objects. Should handle buttons, sliders and images (with or without resize update)
    def view(self,object, *args):
        
        # image object
        if type(object) == np.ndarray:
            self.view_image(object, *args)

        # buttons
        #if type(object) == list:
        #    self.view_buttons(object)

        # slider
        #if type(object) == tk.Scale:
        #    self.view_slider(object)



    def view_image(self, image, *args):

        # Change to PIL format
        image = cv2ImageTk(image)

        # update widget
        self.image_panel.configure(image=image)
        self.image_panel.image = image


        # multiple args indicate a call for resize (by enforce aspect ratio)
        if (len(args)>0):
            pass
            #self.image_frame.place(in_=self.pad_frame, x=0, y=0, width=args[0], height=args[1])

    def view_depth(self, depth_map, *args):

        # Change to PIL format
        depth_map = cv2ImageTk(depth_map)

        # update widget
        self.depth_panel.configure(image=depth_map)
        self.depth_panel.image = depth_map

        # multiple args indicate a call for resize (by enforce aspect ratio)
        if (len(args)>0):
            pass
            #self.image_frame.place(in_=self.pad_frame, x=0, y=0, width=args[0], height=args[1])



    def view_buttons(self, button_list):
        for i in range(len(button_list)):
            button_list[i].grid()


    def view_slider(self, slider):

        # remove previous slider
        widgets = self.buttons_frame.grid_slaves()
        for widget in widgets:
            if (type(widget) == tk.Scale):
                widget.destroy()

        # place slider in frame
        slider.configure(orient='vertical')
        slider.grid(row=0, column = 1)

    def view_orig_and_depth(self, image, depth_map):
        self.view_image(image)
        self.view_depth(depth_map)


        # layout containers
        # self.buttons_frame.grid(row=0, column=0)
        # self.pad_frame.grid(row=1,


    def get_binded_frame(self):
        # return the frame that should be binded to configuration
        return self.pad_frame

"""






# viewer is initiated with master, which is a tk.TK() object
# viewer should set up 2 frames:

# 1. buttons at the top left end
# 2. image at the bottom right - this frame should maintain constant aspect ratio (given as input from controller)

# viewer has method(s) that handle objects (buttons, sliders, images) sent from the controller. A slider should appear next to the buttons.
# each new slider collapses the previous one.

# viewer has a method to return frame which aspect ratio should be kept (image frame)
