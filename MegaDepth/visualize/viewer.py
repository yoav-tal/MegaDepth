import tkinter as tk
import numpy as np
import cv2
from PIL import Image, ImageTk


class viewer:

    def __init__(self, master):

        # Define master widget
        self.master = master
        self.master.title("Depth viewer")

        # set up containers
        #self.buttons_frame = tk.Frame(self.master)
        #self.pad_frame = tk.Frame(self.master, bg='blue')
        self.image_frame = tk.Frame(self.master,bg='blue')

        # Initiate widgets
        self.image_panel = tk.Label(self.image_frame)
        self.image_panel.grid()

        # layout containers
        #self.buttons_frame.grid(row=0, column=0)
        #self.pad_frame.grid(row=1, column=0, sticky='NWSE')
        self.image_frame.grid()

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


    def get_binded_frame(self):
        # return the frame that should be binded to configuration
        return self.pad_frame



# convert ndarray to PIL format
def cv2ImageTk(img):

    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print("array interface:", img.__array_interface__)
    img = Image.fromarray(img)

    return ImageTk.PhotoImage(img)





# viewer is initiated with master, which is a tk.TK() object
# viewer should set up 2 frames:

# 1. buttons at the top left end
# 2. image at the bottom right - this frame should maintain constant aspect ratio (given as input from controller)

# viewer has method(s) that handle objects (buttons, sliders, images) sent from the controller. A slider should appear next to the buttons.
# each new slider collapses the previous one.

# viewer has a method to return frame which aspect ratio should be kept (image frame)
