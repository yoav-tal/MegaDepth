import numpy as np

img = np.reshape(np.arange(5*4*3),(5,4,3))

weights = np.random.random_integers(1,3,size=(4,4))
weights = np.repeat(weights[:, np.newaxis], 3, axis=1)

def image_blur(img, weights, N):
    if N==0:
        return img
    else:
        img[:, N] = img[:, N] - np.multiply(weights[:, :, N-1], img[:,N] - image_blur(img,
                                                                        weights, N-1)[:,N-1])
        return img

new_img = image_blur(img, weights, len(img[0,:])-1)

'''

import cv2
import numpy as np
from cv2.ximgproc import guidedFilter

from visualize.get_depth_mask import get_mask


# hyperparameters (temporary)
focus_start = 0.45
focus_end = 0.55

#world_start = -0.8
#world_end = 0.8

kernel_size = 5
#sigma_start = 1
#sigma_end = 0.3

mask_blur_filter_params = {'ksize':5, 'sigma':1}





path = '/Users/yoav/MegaDepth/images/dinning_room.png'



def main():

    img = cv2.imread(path)
    img = cv2.resize(img, (256, 192))

    depth_mask = get_mask(img) # this is inverse depth, so CoC should be linear with these values

    CoC_mask = get_CoC_mask(depth_mask, focus_start, focus_end)

    blurred_CoC_mask = partial_blur(img=CoC_mask, CoC_mask=CoC_mask, CoC_range=[0,0],
                                    filter_params=mask_blur_filter_params)
# should call a specific function that takes min(val, 0), then pass this mask as img to partial blur
# then do partial blur, but update only pixels with CoC=0 in the original



def partial_blur(img=None, CoC_mask=None, CoC_range=[0,0], depth_mask=None,
                 depth_range=None, filter_params=None):
    return img

# PARTIAL blur needs to take an image and blur some pixels, by mixing with some of the other pixels.
#  All pixels in the given COC range (e.g. pixels with COC<-0.5) are to be blurred using a gaussian
# kernel. However the blur is not necessarily with all the surrounding. The kernel should
# convolve only with pixels in the surrounding for which the value in the depth mask (original
# COC)
# when blurring the mask: the target is to give negative CoC to pixels in the focus near edges
# with pixels in the front. So filter the entire mask, but update values only for pixels with CoC=0.

#it remains to understand how to do the blur, but each region of CoC with what it needs: front
# pixels should convolve with front pixels (but not those with CoC<0 due to the mask), same goes
# for background. only pixels that are from the focus region and changed to have negative CoC
# should convolve with the entire non-positive surrounding.
# This will be done by preprocessing the image before the blur, as follows: The pixels that are
# not to be blurred (e.g. F and B pixels when blurring N) will get a value by the mean of the
# values in the entire blurred region.

# note: maybe I need to use guided filter instead
# cv2 has bilateral filter cv2.bilateralFilter

def get_relative_depth(depth_mask, focus_max, focus_min):

    front_pixels = np.where(depth_mask > focus_max)
    back_pixels = np.where(depth_mask<focus_min)

    relative_depth = np.zeros(depth_mask.shape)

    relative_depth[front_pixels] = depth_mask[front_pixels] - focus_max
    relative_depth[back_pixels] = depth_mask[back_pixels] - focus_min

    return relative_depth




def get_CoC_mask(relative_depth):



    front = get_partial_mask(depth_mask, focus_max, 'front')
    back = get_partial_mask(depth_mask, focus_min, 'back')

    return front + back

def get_partial_mask(depth_mask, focus_depth, type=""):

    partial_mask = np.zeros(depth_mask.shape)

    if type == 'front':
        relevant_pixels = np.where(focus_depth>0)#focus_max)
        partial_mask[relevant_pixels] = -1
        # should do something more sophisticated

    else:
        np.where(focus_depth<0)#focus_max)

    return 1
#c =|d − d_f |/d * f_0^2/(N*(d_f-f_0))

# blur = cv2.GaussianBlur(img,(5,5),0)

if __name__ == '__main__':
    main()
'''

#from guided_filter_pytorch.guided_filter import FastGuidedFilter

