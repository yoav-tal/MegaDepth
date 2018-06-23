import cv2
import tkinter as tk
from tkinter import filedialog
import visualize.model as model
import visualize.viewer as viewer



class control:

    def __init__(self, master):

        # Define master widget
        self.master = master

        # open file dialog
        #path = filedialog.askopenfilename()
        path = '/Users/yoav/Downloads/499.png'

        # set up a model with the image
        self.MODEL = model.model(cv2.imread(path))

        # set up a viewer
        self.VIEWER = viewer.viewer(self.master)

        # view image
        self.VIEWER.view(self.MODEL.update_image())

        # set slider functions (self.slider_functions is a list of functions, function [i] adjusts feature [i])
        #self.slider_functions = self.set_slider_functions()

        # set up buttons (buttons is a list of buttons. button [i] initiates a slider that adjusts feature [i])
        #self.buttons = self.init_buttons()
        #self.VIEWER.view(self.buttons)

        # bind image frame to keep aspect ratio
        #self.binded_frame = self.VIEWER.get_binded_frame()
        #self.bind_frame(self.binded_frame, self.MODEL.get_aspect_ratio())


    def init_buttons(self):

        # get parameters
        n_features = len(model.features_parameters)
        names = [model.features_parameters[i]['name'] for i in range(n_features)]

        # get master for buttons
        frame = self.VIEWER.buttons_frame

        # set up buttons list
        buttons = [tk.Button(master = frame, text=names[i], command=self.get_button_command(i)) for i in range(n_features)]

        return buttons


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




if __name__ == '__main__':


    root = tk.Tk()
    CONTROL = control(root)
    root.mainloop()



# controller initiative actions are opening a browse window and getting a path
# (potentially handling exceptions)

# control gets an image (potentially handles errors) and sends it to the model. control asks model for aspect ratio.

# control sets up a viewer (inputs it the aspect ratio)

# control builds buttons along with the commands they activate (which buttons to use is determined by a metadata list)
# (each button toggles a slider) and sends them to the viewer

# control calls the model to return an image and sends it to the viewer


# control has a method to send objects to the viewer asking it to present them (sliders, buttons, images).
# The viewer is responsible to know how to handle each object

# upon a slider event, controller calls the model to update the image, then sends the updated image to the viewer

# when the viewer informs the controller of a change in configuration, the controller asks the model for an update (and sends te updated image to the viewer)

# the function to get aspect ratio from model is get_aspect_ratio