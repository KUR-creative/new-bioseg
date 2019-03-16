from keras import backend as K
#https://stackoverflow.com/questions/45947351/how-to-use-tensorflow-metrics-in-keras
# IOU metric: https://www.pyimagesearch.com/wp-content/uploads/2016/09/iou_equation.png
import functools
import tensorflow as tf
def as_keras_metric(method):
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper

@as_keras_metric
def mean_iou(y_true, y_pred, num_classes=1):
    return tf.metrics.mean_iou(y_true, y_pred, num_classes)

def jaccard_coefficient(y_true, y_pred, smooth=100, weight1=1.):
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(y_true + y_pred, axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return jac

def jaccard_distance(n_channels, smooth=1):
    ''' 
    Cacluate channel by channel intersection & union.
    And then calculate smoothed jaccard_coefficient.
    Finally, calculate jaccard_distance.
    '''
    axis = tuple(range(n_channels))
    def jacc_dist(y_true, y_pred):
        intersection = y_pred * y_true
        sum_ = y_pred + y_true
        numerator = K.sum(intersection, axis) + smooth
        denominator = K.sum(sum_ - intersection, axis) + smooth
        jacc =  K.mean(numerator / denominator)
        return 1-jacc
    return jacc_dist
'''
A weighted version of categorical_crossentropy for keras (2.0.6). 
This lets you apply a weight to unbalanced classes.
@url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
@author: wassname
'''
from keras import backend as K
def weighted_categorical_crossentropy(weights):
    '''
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    '''
    weights = K.variable(weights)        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    return loss

def jacc(y_pred,y_true, smooth=0.000001):
    axis = tuple(range(len(y_pred.shape)))
    intersection = y_pred * y_true
    #print('i\n',intersection)
    sum_ = y_pred + y_true
    #print('s\n',sum_)
    numerator = np.sum(intersection, axis) + smooth
    #print('numerator',numerator)
    denominator = np.sum(sum_ - intersection, axis) + smooth
    #print('denominator',denominator)
    jac =  np.mean(numerator / denominator)
    return 1-jac

import numpy as np
import cv2
if __name__ == '__main__':
    p = np.array([
        [[1,0],[1,0],[1,0]],
        [[1,0],[1,0],[1,0]],
        [[0,1],[0,1],[0,1]],
    ]) 
    p = np.array([
        [[1,0,0],[1,0,0],[1,0,0]],
        [[1,0,0],[1,0,0],[1,0,0]],
        [[0,1,0],[0,1,0],[0,1,0]],
    ]) 
    t = np.array([
        [[1,0,0],[1,0,0],[1,0,0]],
        [[0,1,0],[0,1,0],[0,1,0]],
        [[1,0,0],[1,0,0],[1,0,0]],
    ]) 
    #cv2.imshow('p',(p*t).astype(np.uint8)*255); cv2.waitKey(0)
    #print(p*t)
    a = np.array([
        [[1,0,0],[1,0,0],[1,0,0]],
        [[0,1,0],[0,1,0],[0,1,0]],
        [[1,0,0],[1,0,0],[1,0,0]],
        [[1,0,0],[1,0,0],[1,0,0]],
    ]) 
    b = np.array([
        [[1,0],[1,0],[1,0]],
        [[1,0],[1,0],[1,0]],
        [[0,1],[0,1],[0,1]],
        [[0,1],[0,1],[0,1]],
    ]) 
    c = np.array([
        [[1,0],[1,0],[1,0]],
        [[0,1],[0,1],[0,1]],
        [[1,0],[1,0],[1,0]],
        [[0,1],[0,1],[0,1]],
    ]) 

    b = np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
    ]) 
    c = np.array([
        [1,1,1,1,1,1],
        [1,1,1,1,1,1],
        [0,0,0,0,0,0],
        [0,0,0,0,0,0],
    ]) 

    b = np.array([
        [[1,0],[1,0],[1,0]],
        [[1,0],[1,0],[1,0]],
        [[1,0],[1,0],[1,0]],
        [[1,0],[1,0],[1,0]],
    ]) 
    c = np.array([
        [[0,1],[0,1],[0,1]],
        [[0,1],[0,1],[0,1]],
        [[1,0],[1,0],[1,0]],
        [[1,0],[1,0],[1,0]],
    ]) 

    b = c.copy()
    print('jacc:', jacc(b,c))
    #print('j sum',np.sum(j,axis=(0,1)))

