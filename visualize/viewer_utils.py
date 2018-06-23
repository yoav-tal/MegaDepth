import numpy as np
import cv2
from PIL import Image, ImageTk


BUTTONS_PER_COLUMN = 3
FRAMES_PER_ROW = 3

def cv2ImageTk(img):
    if np.max(img) > 1:
        img = img/np.max(img)

    img = np.uint8(img * 255)

    if len(img.shape) == 3:
       img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #print("array interface:", img.__array_interface__)
    img = Image.fromarray(img)

    return ImageTk.PhotoImage(img)

def calc_button_grid(button_num):
    return button_num % BUTTONS_PER_COLUMN, button_num // BUTTONS_PER_COLUMN

def calc_frame_grid(frame_num, name):
    if name == "buttons":
        return 0, 0
    elif name == "sliders":
        return 0, 1
    else: # image frame
        return (frame_num-2) // FRAMES_PER_ROW + 1, (frame_num-2) % FRAMES_PER_ROW
