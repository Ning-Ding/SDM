# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 09:57:11 2016

@author: dingning
"""

import numpy as np
from scipy import sqrt, pi, arctan2
from scipy.ndimage import uniform_filter


def hog(image, xy, orientations=9, pixels_per_cell=3,cells_per_side=1, cells_per_block=1):
    '''
    Given a grey image in numpy array and a vector of sequence of coordinates,
    return the ndarray of hog feature vectors extract from given locations
    ---------------------------------------------------------------------------
    INPUT:
        image: grey image, numpy array, 8-bit
        xy: coordinates, numpy array, float
        orientations: the number that the orientations are divided, int
        pixels_per_cell: int
        cells_per_side: int
        cells_per_block: int
    OUTPUT:
        features: ndarray of all the features extracted from locations in xy
    ---------------------------------------------------------------------------
    '''
    image = np.atleast_2d(image)
    
    if image.ndim > 3: raise ValueError("Currently only supports grey-level images")

    '''normalisation'''
    image = sqrt(image)
    
    '''---------compute the gradients of the input grey image---------------'''
    gx = np.zeros(image.shape)
    gy = np.zeros(image.shape)
    gx[:, :-1] = np.diff(image, n=1, axis=1)
    gy[:-1, :] = np.diff(image, n=1, axis=0)
    '''---------------------------------------------------------------------'''
    
    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 180
    

    r = pixels_per_cell * cells_per_side
    pc = pixels_per_cell
    x, y = xy

    # compute orientations integral images
    orientation_histogram = np.zeros((cells_per_side*2, cells_per_side*2, orientations))
    for i in range(orientations):
        # create new integral image for this orientation
        # isolate orientations in this range

        temp_ori = np.where(orientation <= 180 / orientations * (i + 1) * 2,
                            orientation, 0)
        temp_ori = np.where(orientation > 180 / orientations * i * 2,
                            temp_ori, 0)
        # select magnitudes for those orientations
        cond2 = temp_ori > 0
        temp_mag = np.where(cond2, magnitude, 0)
        

        orientation_histogram[:,:,i] = uniform_filter(temp_mag, size=pc)[x-r+pc/2:x+r:pc, y-r+pc/2:y+r:pc].T
    '''---------------------------------------------------------------------'''
    
    
    n_blocks = cells_per_side * 2 - cells_per_block + 1
    cb = cells_per_block
    normalised_blocks = np.zeros((n_blocks, n_blocks, cb, cb, orientations))
    eps = 1e-5
    for x in range(n_blocks):
        for y in range(n_blocks):
            block = orientation_histogram[x:x + cb, y:y + cb, :]            
            normalised_blocks[x, y, :] = block / sqrt(block.sum() ** 2 + eps)
    
    return normalised_blocks.ravel()
    
