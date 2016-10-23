# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 19:02:31 2016

@author: dingning
"""
from __future__ import division
import os
from PIL import Image,ImageOps,ImageDraw
import numpy as np
from scipy import sqrt, pi, arctan2, io
from scipy.ndimage import uniform_filter
from sklearn.linear_model import Lasso
from math import floor

class model_parameters(object):
    def __init__(self,
                 N=5,
                 alpha=0.1,
                 new_size=(400,400),
                 expand=100,
                 expand_rate=0.2,
                 orientations=9,
                 pixels_per_cell=3,
                 cells_per_block=3,
                 cells_per_side=2,
                 train_or_test='train'):
        self.N = N
        self.alpha = alpha
        self.expand =expand
        self.expand_rate = expand_rate
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.cells_per_side = cells_per_side
        self.train_or_test = train_or_test
        
        
def train(parameters):
    '''
    the standard SDM training function
    ---------------------------------------------------------------------------
    INPUT:
        N: number of train iteration
        alpha: the coefficient of L1 regularization of Lasso Linear Regression
        new_size: the final image size
        expand: number of pixels to expand after crop
        expand_rate: the rate of new bbox to original bbox
        orientations: the number of classes of gradient angle histogram in hog
        pixels_per_cell: a hog feature parameter
        cells_per_side: a hog feature parameter
        cells_per_block: a hog feature parameter
    OUTPUT:
        regressors: a list containing a sequence of (R,b)
        initials: a numpy array containing a initial landmarks
    ---------------------------------------------------------------------------
    '''
    image_path_list = get_image_path_list(parameters)
    bbox_dict = load_boxes(parameters)
    mark_list = []
    hog_list = []
    grey_list = []
    print 'computing the hog features for ture landmarks...........'
    for path in image_path_list:
        grey,mark = crop_and_resize_image(path[:10],bbox_dict[path],parameters)
        hog_list.append(hog(grey,mark,parameters))
        grey_list.append(grey)
        mark_list.append(mark.ravel())
        
    HOG_TRUE = np.array(hog_list)
    MARK_TRUE = np.array(mark_list)
    initials = np.mean(MARK_TRUE,axis = 0).astype(int)
    MARK_x = np.array([initials.tolist()] * len(image_path_list))
    initials = initials.reshape(68,2)
    coef = []
    inte = []
    
    for i in range(parameters.N):
        
        print 'Iteration: ',i + 1
        
        MARK_delta = MARK_TRUE - MARK_x
        HOG_x = np.zeros_like(HOG_TRUE)
        for j in range(len(image_path_list)):
            if j+1 % 100 == 0: print 'already computed',j+1,'features'
            HOG_x[j,:] = hog(grey_list[j],MARK_x[j,:].reshape(68,2))
        
        reg = Lasso(alpha=parameters.alpha)
        print 'computing the lasso linear regression.......'
        reg.fit(HOG_x,MARK_delta)  
        coef.append(reg.coef_.T)
        inte.append(reg.intercept_.T)       
                
        MARK_x = MARK_x + np.matmul(HOG_x, coef[i]) + inte[i]
        
    return np.array(coef),np.array(inte),initials
    
    

def test_for_one_image(coef,inte,path,bbox,initials,parameters):
                           
                           
    grey,mark_true = crop_and_resize_image(path[:10],bbox,parameters)
    mark_x = initials.astype(float)
    MSE = []
    
    for i in range(coef.shape[0]):
        hog_x = hog(grey,mark_x,parameters)
        mark_x = (mark_x.ravel() + np.matmul(hog_x,coef[i]).astype(float) + inte[i].astype(float)).reshape(68,2)
        MSE.append((abs(mark_x.astype(int) - mark_true)**2).sum() / len(mark_true))
        
    im = Image.fromarray(grey)
    draw = ImageDraw.Draw(im)
    draw.point(mark_x,fill = 'red')
    im.show()
    
        
    return mark_x.astype(int),mark_true,MSE



def get_image_path_list(parameters):
    '''
    get a list containing all the paths of images in the trainset
    ---------------------------------------------------------------------------
    INPUT:
        data_type: 'train' or 'test'  by default is 'train'
    OUTPUT:
        a list with all the images' paths
    ---------------------------------------------------------------------------
    '''
    folder_path = 'data/' + parameters.train_or_test + 'set/png'
    assert os.path.exists(folder_path)
    assert os.path.isdir(folder_path)
    print 'already get all the image path.'
    return os.listdir(folder_path)




def load_boxes(parameters):
    '''
    load the ground truth ground truth boxes coordinates from .mat file
    ---------------------------------------------------------------------------
    INPUT:
        data_type: 'train' or 'test'  by default is 'train'
    OUTPUT:
        a dict with all the ground truth bounding boxes coordinates
        key: a string of filename    ex: 'image_0122.png'
        value: a numpy array of boungding boxes
    ---------------------------------------------------------------------------
    '''
    file_path = 'data/bounding_boxes/bounding_boxes_lfpw_' + parameters.train_or_test + 'set.mat'
    assert os.path.exists(file_path)
    x = io.loadmat(file_path)['bounding_boxes'][0]
    x = [x[0][0] for x in x]
    print 'loading ground truth bboxes....................'
    return {x[i][0][0]:x[i][1][0] for i in range(len(x))}
    
    
def load_landmarks(image_name,parameters):
    '''
    load the landmarks coordinates from .pts file
    ---------------------------------------------------------------------------
    INPUT:
        image_name: a string without extension   ex: 'image_0122'
    OUTPUT:
        a numpy array containing all the points
    ---------------------------------------------------------------------------
    '''   
    file_path = 'data/' + parameters.train_or_test + 'set/pts/' + image_name + '.pts'  
    assert os.path.exists(file_path)
    with open(file_path) as f: rows = [rows.strip() for rows in f]
    coords_set = [point.split() for point in rows[rows.index('{') + 1:rows.index('}')]]
    return np.array([list([float(point) for point in coords]) for coords in coords_set])
    



def compute_new_bbox(image_size,bbox,parameters):
    '''
    compute the expanded bbox
    a robust function to expand the crop image bbox even the original bbox is
    around the border of the image
    ---------------------------------------------------------------------------
    INPUT:
        image_size: a tuple   ex: (height,width)
        bbox: the ground_truth bounding boxes  ex:[x0,y0,x1,y1]
        expand_rate: the rate to expand the bbox  by default is 0.2
    OUTPUT:
        new bbox: ex:[x0,y0,x1,y1]
    '''
    x_size,y_size = image_size
    bx0,by0,bx1,by1 = bbox
    bw = by1 - by0
    bh = bx1 - bx0
    if bw > bh:
        delta = parameters.expand_rate * bw
        if by1 + delta > y_size:
            nby1 = y_size
        else:
            nby1 = int(floor(by1 + delta))
        if by0 - delta < 0:
            nby0 = 0
        else:
            nby0 = int(floor(by0 - delta))
        new_w = nby1 - nby0
        delta_h = (new_w - bh) / 2
        if bx0 - delta_h < 0:
            nbx0 = 0
        else:
            nbx0 = int(floor(bx0 - delta_h))
        if bx1 + delta_h > x_size:
            nbx1 = x_size
        else:
            nbx1 = int(floor(bx1 + delta_h))
    else:
        delta = parameters.expand_rate * bh
        if bx1 + delta > x_size:
            nbx1 = x_size
        else:
            nbx1 = int(floor(bx1 + delta))
        if bx0 - delta < 0:
            nbx0 = 0
        else:
            nbx0 = int(floor(bx0 - delta))
        new_h = nbx1 - nbx0
        delta_w = (new_h - bw) / 2
        if by0 - delta_w < 0:
            nby0 = 0
        else:
            nby0 = int(floor(by0 - delta_w))
        if by1 + delta_w > y_size:
            nby1 = y_size
        else:
            nby1 = int(floor(by1 + delta_w))
    return nbx0,nby0,nbx1,nby1
    
def crop_and_resize_image(image_name,bbox,parameters):
    '''
    crop and resize the image given the ground truth bounding boxes
    also, compute the new coordinates according to transformation
    ---------------------------------------------------------------------------
    INPUT:
        image_name: a string without extension  ex: 'image_0007'
        bbox: the ground_truth bounding boxes  ex:[x0,y0,x1,y1]
        new_size: the size used by the resize function
        data_type: 'train' or 'test'
    OUTPUT:
        grey: a numpy array of grey image after crop and resize
        landmarks: new landmarks accordance with new image
    ---------------------------------------------------------------------------
    '''
    image_path = 'data/' + parameters.train_or_test + 'set/png/' + image_name + '.png'
    assert os.path.exists(image_path)
    im = Image.open(image_path)
    bbox = compute_new_bbox(im.size,bbox,parameters.expand_rate)
    im_crop = im.crop(bbox)
    Expand = parameters.expand
    im_expand = ImageOps.expand(im_crop,(Expand,Expand,Expand,Expand),fill = 'black')
    im_resize = im_expand.resize(parameters.new_size)
    grey = im_resize.convert('L')
    
    #compute the new landmarks according to transformation procedure
    landmarks = load_landmarks(image_name,parameters.train_or_test)
    landmarks = landmarks - (bbox[:2]) + Expand
    landmarks = landmarks * im_resize.size / im_expand.size
    
    return np.array(grey),landmarks.astype(int)


def hog(image, xys, parameters):
    '''
    Given a grey image in numpy array and a vector of sequence of coordinates,
    return the ndarray of hog feature vectors extract from given locations
    ---------------------------------------------------------------------------
    INPUT:
        image: grey image, numpy array, 8-bit
        xys: coordinates, numpy array, float
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
    
    
    '''----------compute the magnitude and orientation of gradients---------'''
    magnitude = sqrt(gx ** 2 + gy ** 2)
    orientation = arctan2(gy, (gx + 1e-15)) * (180 / pi) + 180
    '''---------------------------------------------------------------------'''

    #just for convinients, make the variables shorter
    r = parameters.pixels_per_cell * parameters.cells_per_side
    pc = parameters.pixels_per_cell
    
    
    '''--------------compute the orientation histogram----------------------'''
    orientation_histogram = np.zeros((len(xys), 
                                      parameters.cells_per_side*2, 
                                      parameters.cells_per_side*2, 
                                      parameters.orientations))    
    for j in range(len(xys)):        
        x, y = xys[j]
        for i in range(parameters.orientations):
            # classify the orientation of the gradients
            temp_ori = np.where(orientation <= 180 / parameters.orientations * (i + 1) * 2,
                                orientation, 0)
            temp_ori = np.where(orientation > 180 / parameters.orientations * i * 2,
                                temp_ori, 0)
            # select magnitudes for those orientations
            cond2 = temp_ori > 0
            temp_mag = np.where(cond2, magnitude, 0)
        
            orientation_histogram[j,:,:,i] = uniform_filter(temp_mag, size=pc)[x-r+pc/2:x+r:pc, y-r+pc/2:y+r:pc].T
    '''---------------------------------------------------------------------'''
    
    
    '''----------------compute the block normalization----------------------'''
    n_blocks = parameters.cells_per_side * 2 - parameters.cells_per_block + 1
    cb = parameters.cells_per_block
    normalised_blocks = np.zeros((len(xys), n_blocks, n_blocks, cb, cb, parameters.orientations))
    eps = 1e-5
    for i in range(len(xys)):
        for x in range(n_blocks):
            for y in range(n_blocks):
                block = orientation_histogram[i,x:x + cb, y:y + cb, :]            
                normalised_blocks[i, x, y, :] = block / sqrt(block.sum() ** 2 + eps)
    '''---------------------------------------------------------------------'''
    
    return normalised_blocks.ravel()



#just for the test purpose
if __name__ == '__main__':
    parameters = model_parameters()
    image_path_list = get_image_path_list(parameters)
    bbox_dict = load_boxes(parameters)
