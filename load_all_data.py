# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:23:58 2016
@author: dingning
"""
import os
from scipy import io
from PIL import Image,ImageDraw
from pylab import *

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
    
    
def crop_image(image_name,bbox,data_type = 'train'):
    '''
    input: image_name ex:'image_0122'
           bbox like this: [x1,y1,x2,y2] or [(x1,y1),(x2,y2)]
           data_type maybe 'train' or 'test' by default is 'train'
    output: a tuple containing a cropped image numpy array and it's landmarks
    '''
    image_path = 'data/' + data_type + 'set/png/' + image_name + '.png'
    assert os.path.exists(image_path)
    im = Image.open(image_path)
    bbox = map(lambda x: int(round(x)),bbox)
    im_crop = im.crop(bbox)
    marks = load_landmarks(image_name)
    marks -= bbox[:2]
    im_array = array(im_crop)
    return im_array,marks
    
    '''
    imshow(im_crop)
    plot(marks.T[0],marks.T[1],'b*')
    show()
    '''



if __name__=='__main__':
    '''
    by run this python file, we could get a list named Data
    Data containing 811 tuples and each tuple holds two numpy array containing images and marks
    '''
    bbox = load_ground_truth_data()
    image_list = get_data_path_list()
    Data = []
    for path in image_list:Data.append(crop_image(path[:10], list(bbox[path])))
