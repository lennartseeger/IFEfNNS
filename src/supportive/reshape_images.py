# Author: PSEFSD
# This file is licensed under the Apache 2.0 license available at
# http://www.apache.org/licenses/LICENSE-2.0
import cv2
import numpy as np
import sys

sys.path.insert(1, '../src')

"""
    Function for the resizing of image arrays

    Attributes
    ----------
    image_array: array
        image array that should be resized
    image_size: number
        size the images should be resized to
    
    Returns
    ----------
    reshaped_images : array
        resized image array
"""
def reshape_images(image_array, image_size=64):
    reshaped_images = []
    for x in image_array:
        reshaped_x = cv2.resize(x, dsize=(image_size, image_size),
                                interpolation=cv2.INTER_NEAREST)
        reshaped_images.append(reshaped_x)
    return np.asarray(reshaped_images)
