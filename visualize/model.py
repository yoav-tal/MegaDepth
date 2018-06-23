import numpy as np
import cv2

from .get_depth_mask import get_mask

#########################################################


def show_depth_map(img):

    return get_mask(img)


#########################################################


#hyperparameters

# of masks
depth_parameters = {'name':'depth', 'func': show_depth_map, 'active':True}


# of size:
MAX_IMAGE_WIDTH= 512
MAX_IMAGE_HEIGHT = 512
WIDTH_DIVISOR = 16
HEIGHT_DIVISOR = 16

#########################################################

# parameter lists

size_parameters = {'width':MAX_IMAGE_HEIGHT, 'height':MAX_IMAGE_HEIGHT}
features_parameters = []

masks_parameters = []
masks_parameters.append(depth_parameters)
#########################################################



class model:

    def __init__(self, image):

        # get size to fit mask constraints (depth mask: dims must be 0 mod 16
        size_parameters.update(zip(['width', 'height'], set_dimensions(image)))
        self.aspect_ratio = size_parameters['width'] / size_parameters['height']

        print(size_parameters)

        # keep resized image as original
        self.original_image = cv2.resize(image, (size_parameters['width'],
             size_parameters['height']))
        # does it matter if I resize with cv2 instead of skimage as in the published demo?

        self.inverse_depth_map = self.get_inverse_depth()

    # retuen aspect ratio
    def get_aspect_ratio(self):
        return self.aspect_ratio

    def get_image_and_depth(self):

        uint_depth = image2uint(self.inverse_depth_map)

        return self.original_image, uint_depth


    def get_inverse_depth(self):

        image = self.original_image
        for i in range (len(masks_parameters)):
            if masks_parameters[i]['name'] == 'depth':
                func = masks_parameters[i]['func']
                inv_depth_map = func(image)

        return inv_depth_map



def set_dimensions(image, max_height=MAX_IMAGE_HEIGHT, max_width=MAX_IMAGE_WIDTH,
                   heigt_divisor=HEIGHT_DIVISOR, width_divisor=WIDTH_DIVISOR):
    # get aspect ratio of original
    height, width = np.shape(image)[:2]
    aspect_ratio = width / height

    # Set dimentions within boundaries while maintaining the aspect ratio
    if width > max_width:
        width = max_width
        height = round(width / aspect_ratio)

    if height > max_height:
        height = max_height
        width = round(height * aspect_ratio)

    # Set height and width to be multiples of required divisors
    height = height - height%heigt_divisor
    width = width - width%width_divisor

    return width, height

def image2uint(img):
    #img = img[0][0]

    #img = (img-np.min(img))/np.max(img-np.min(img))*255
    #img = np.array([img,img,img])

    #cv2.imshow('image',img)

    return(np.uint8(img*255))