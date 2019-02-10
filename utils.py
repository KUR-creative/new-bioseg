import os, re
import cv2
import numpy as np
import time
from datetime import datetime
#-------- utils --------
class ElapsedTimer(object):
    def __init__(self,msg='Elapsed'):
        self.start_time = time.time()
        self.msg = msg
    def elapsed(self,sec):
        if sec < 60:
            return str(sec) + " sec"
        elif sec < (60 * 60):
            return str(sec / 60) + " min"
        else:
            return str(sec / (60 * 60)) + " hr"
    def elapsed_time(self):
        print(self.msg + ": %s " % self.elapsed(time.time() - self.start_time),
              flush=True)
        return (self.msg + ": %s " % self.elapsed(time.time() - self.start_time))

def now_time_str():
    return datetime.now().strftime('%Y-%m-%d_%H_%M_%S')

def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

def bgr_float32(uint8img):
    c = 1 if len(uint8img.shape) == 2 else 3
    h,w = uint8img.shape[:2]
    uint8img = (uint8img / 255).astype(np.float32)
    return uint8img.reshape((h,w,c))

def bgr_uint8(float32img):
    return (float32img * 255).astype(np.uint8)

def file_paths(root_dir_path):
    ''' generate file_paths of directory_path ''' 
    it = os.walk(root_dir_path)
    for root,dirs,files in it:
        for path in map(lambda name:os.path.join(root,name),files):
            yield path

from collections import namedtuple
def filename_ext(path):
    name_ext = namedtuple('name_ext','name ext')
    return name_ext( *os.path.splitext(os.path.basename(path)) )

def unique_colors(img):
    return np.unique(img.reshape(-1,img.shape[2]), axis=0)

from sklearn.model_selection import train_test_split
def splited_paths(img_paths,mask_paths, train_r=0.7,valid_r=0.2,test_r=0.1):
    ''' Randomly split paths into train/valid/test (default = 7:2:1) '''
    pair_paths = list(zip(human_sorted(img_paths), human_sorted(mask_paths)))
    train_pairs, other_pairs = train_test_split(pair_paths)
    valid_pairs, test_pairs = train_test_split(other_pairs)
    train_imgs, train_masks = zip(*train_pairs)
    valid_imgs, valid_masks = zip(*valid_pairs)
    test_imgs, test_masks = zip(*test_pairs)
    return (list(train_imgs), list(train_masks), 
            list(valid_imgs), list(valid_masks), 
            list(test_imgs), list(test_masks))

def load_imgs(img_paths, mode_flag=cv2.IMREAD_COLOR):
    def imread(path, mode_flag):
        #print('wtf?:', path)
        return cv2.imread(path,mode_flag)
    #print(img_paths)
    #return map(lambda path: bgr_float32(cv2.imread(path, mode_flag)), img_paths) 
    return map(lambda path: bgr_float32(imread(path, mode_flag)), img_paths) 

from keras.utils import to_categorical
def categorize(img):
    colors = np.unique(img.reshape(-1,img.shape[2]), axis=0)

    h,w,_ = img.shape
    n_classes = colors.shape[0]
    ret_img = np.zeros((h,w,n_classes))

    img_b, img_g, img_r = np.rollaxis(img, axis=-1)
    origin_map = {}
    for i,(b,g,r) in enumerate(colors):
        category = to_categorical(i, n_classes)
        masks = (img_b == b) & (img_g == g) & (img_r == r) # if [0,0,0]
        ret_img[masks] = category
        origin_map[tuple(map(np.asscalar,category))] \
            = [np.asscalar(b), np.asscalar(g), np.asscalar(r)]
    return ret_img, origin_map

def decategorize(categorized, origin_map):
    #TODO: Need to vectorize!
    h,w,n_classes = categorized.shape
    n_channels = len(next(iter(origin_map.values())))
    ret_img = np.zeros((h,w,n_channels))
    for c in range(n_classes):
        category = to_categorical(c, n_classes)
        origin = origin_map[tuple(category)]
        for y in range(h):
            for x in range(w):
                if np.alltrue(categorized[y,x] == category):
                    ret_img[y,x] = origin
    return ret_img
        

import unittest
class test_categorize_func(unittest.TestCase):
    def setUp(self):
        self.img = np.array(
            [[[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],], 
             [[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.],], 
             [[1.,1.,1.], [1.,1.,1.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.],], 
             [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.],],])

    def test_categorize3color(self):
        img = np.copy(self.img)
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(img == decategorized))

    def test_categorize4color(self):
        img = np.copy(self.img)
        img[0,0] = [2,2,2]
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(img == decategorized))

    def test_real_img(self):
        img = bgr_float32(cv2.imread('../t.bmp'))
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        cv2.imshow('c',categorized)
        cv2.imshow('d',decategorized); cv2.waitKey(0)
        self.assertTrue(np.alltrue(img == decategorized))

if __name__ == '__main__':
    '''
    print(splited_paths(['11','1','2','3','4','11','1','2','3','4',
                         '11','10','20','30','40','11','10','20','30','40',],
                        ['110','100','200','300','400','110','100','200','300','400',
                         '110','100','200','300','400','110','10','20','30','40',
                         ]))
    print(*splited_paths(human_sorted(file_paths('../boundary_data190125/image/')),
                        human_sorted(file_paths('../boundary_data190125/label/'))),sep='\n')
    '''
    unittest.main()

    img = bgr_float32(cv2.imread('../t.bmp'))
    img = img[101:105,:10,:]
    unique_colors = np.unique(img.reshape(-1,img.shape[2]), axis=0)
    num_classes = unique_colors.shape[0]
    print(img)
    print(unique_colors)
    print(num_classes)
    #print('->', img[img[:,:] == [0,0,0]])
    #print('->', img[:,:] == [0,1,0])
    #print(img[img[:,:]==[0,1,0],:])
    #img[img[:,:] == [0,0,0]] = 2;
    r, g, b = np.rollaxis(img, axis=-1)
    mask = (r == 0) & (g == 0) & (b == 0) # if [0,0,0]
    img[mask] = [2,2,2]                 # then [2,2,2]
    #print('-->',img[mask])
    print(img)
    #print(r,g,b,sep='\n')
    #print(img)
