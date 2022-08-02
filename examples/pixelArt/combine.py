import cv2
import numpy as np

class combine:
    #Author:  Alican Akca
    def __init__(self, size = (400,300),images = [],background_image = None):
        self.size = size
        self.images = images
        self.background_image = background_image

    def combiner(self,images,background_image):
        original = images[0]
        masked = images[1]
        background = cv2.resize(background_image,(images[0].shape[1],images[0].shape[0]))
        result = blend_images_using_mask(original, background, masked)
        return result 

def mix_pixel(pix_1, pix_2, perc):

    return (perc/255 * pix_1) + ((255 - perc)/255 * pix_2)

def blend_images_using_mask(img_orig, img_for_overlay, img_mask):

    if len(img_mask.shape) != 3:
        img_mask = cv2.cvtColor(img_mask, cv2.COLOR_GRAY2BGR)

    img_res = mix_pixel(img_orig, img_for_overlay, img_mask)

    return cv2.cvtColor(img_res.astype(np.uint8), cv2.COLOR_BGR2RGB)