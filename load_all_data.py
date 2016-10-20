# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:23:58 2016
@author: dingning
"""
import os
import numpy as np
from scipy import io

#loading ground_truth mat file
def load_ground_truth_data():
    '''
    output data type: dict
    key: a string of filename
    value: a numpy array of bounding_box
    '''
    file_path = 'data/bounding_boxes/bounding_boxes_lfpw_trainset.mat'
    assert os.path.exists(file_path)
    x = io.loadmat(file_path)['bounding_boxes'][0]
    x = [x[0][0] for x in x]
    return {x[i][0][0]:x[i][1][0] for i in range(len(x))}
    
#loading landmarks pts file
def load_landmarks(image_name):
    '''
    output data type: a numpy array containing tuples of points    
    '''   
    file_path = 'data/trainset/pts/' + image_name + '.pts'  
    assert os.path.exists(file_path)
    with open(file_path) as f: rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows[rows.index('{') + 1:rows.index('}')]]
    return np.array([tuple([float(point) for point in coords]) for coords in coords_set])
    
#loading images png file    
def get_data_path_list(data='train'):
    '''
    input: 'train' or 'test'  by default is 'train'
    putput: get a list containing all the training images' paths
    '''
    folder_path = 'data/' + data + 'set/png'
    assert os.path.exists(folder_path)
    assert os.path.isdir(folder_path)
    return os.listdir(folder_path)
    
