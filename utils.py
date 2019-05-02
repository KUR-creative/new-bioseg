import os, re
import cv2
import numpy as np
import time
from datetime import datetime
import funcy as F
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
    float32img = (uint8img / 255).astype(np.float32)
    #print('channel:',c)
    return float32img.reshape((h,w,c))

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

def filename(path):
    return filename_ext(path).name

def extension(path):
    return filename_ext(path).ext

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

def map_colors(img, dst_src_colormap): # {dst1:src1, dst2:src2, ...}
    h,w,_ = img.shape
    n_classes = len(dst_src_colormap)
    ret_img = np.zeros((h,w,n_classes))

    if n_classes == 2:
        img_b, img_g, img_r = np.rollaxis(img, axis=-1)
        for i,(dst_color,(src_b,src_g,src_r)) in enumerate(dst_src_colormap.items()):
            masks = ((img_b == src_b) 
                   & (img_g == src_g) 
                   & (img_r == src_r)) # if [0,0,0]
            ret_img[masks] = dst_color
    elif n_classes == 3:
        img_b, img_g, img_r = np.rollaxis(img, axis=-1)
        for i,(dst_color,(src_b,src_g,src_r)) in enumerate(dst_src_colormap.items()):
            masks = ((img_b == src_b) 
                   & (img_g == src_g) 
                   & (img_r == src_r)) # if [0,0,0]
            ret_img[masks] = dst_color
    elif n_classes == 4:
        img_b, img_g, img_r, img_3 = np.rollaxis(img, axis=-1)
        for i,(dst_color,(src_b,src_g,src_r,src_3)) in enumerate(dst_src_colormap.items()):
            masks = ((img_b == src_b) 
                   & (img_g == src_g) 
                   & (img_r == src_r)
                   & (img_3 == src_3)) # if [0,0,0]
            ret_img[masks] = dst_color
    # ... TODO: refactor it!!!
    return ret_img

from keras.utils import to_categorical
def categorize_with(img, origin_map):
    colors = np.unique(img.reshape(-1,img.shape[2]), axis=0)
    #print(colors, origin_map)
    assert set(map(tuple, colors.tolist() )) <= set(map(tuple, origin_map.values() ))

    ret_img = map_colors(img, origin_map)
    return ret_img
    '''
    img_b, img_g, img_r = np.rollaxis(img, axis=-1)
    for i,(dst_color,(b,g,r)) in enumerate(origin_map.items()):
        masks = (img_b == b) & (img_g == g) & (img_r == r) # if [0,0,0]
        ret_img[masks] = dst_color
    '''

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

def map_max_row(img, val=1):
    assert len(img.shape) == 3
    img2d = img.reshape(-1,img.shape[2])
    ret = np.zeros_like(img2d)
    ret[np.arange(len(img2d)), img2d.argmax(1)] = val
    return ret.reshape(img.shape)

def decategorize(categorized, origin_map):
    '''
    #TODO: Need to vectorize!
    h,w,n_classes = categorized.shape
    n_channels = len(next(iter(origin_map.values())))
    ret_img = np.zeros((h,w,n_channels))
    for c in range(n_classes):
        category = to_categorical(c, n_classes)
        origin = origin_map[tuple(category)]
        print('origin', origin)
        for y in range(h):
            for x in range(w):
                if np.alltrue(categorized[y,x] == category):
                    ret_img[y,x] = origin
    return ret_img
    '''
    #TODO: Need to vectorize!
    h,w,n_classes = categorized.shape
    n_channels = len(next(iter(origin_map.values())))
    ret_img = np.zeros((h,w,n_channels))

    if n_classes == 3:
        img_b, img_g, img_r = np.rollaxis(categorized, axis=-1)
        for c in range(n_classes):
            category = to_categorical(c, n_classes)
            origin = origin_map[tuple(category)]

            key_b, key_g, key_r = category
            masks = ((img_b == key_b) 
                   & (img_g == key_g) 
                   & (img_r == key_r)) # if [0,0,0]
            ret_img[masks] = origin

    elif n_classes == 2:
        img_0, img_1 = np.rollaxis(categorized, axis=-1)
        for c in range(n_classes):
            category = to_categorical(c, n_classes)
            origin = origin_map[tuple(category)]

            key_0, key_1 = category
            masks = ((img_0 == key_0) 
                   & (img_1 == key_1)) # if [0,0,0]
            ret_img[masks] = origin

    elif n_classes == 4:
        img_0, img_1, img_2, img_3 = np.rollaxis(categorized, axis=-1)        
        for c in range(n_classes):
            category = to_categorical(c, n_classes)
            origin = origin_map[tuple(category)]

            key_0, key_1, key_2, key_3 = category
            masks = ((img_0 == key_0) 
                   & (img_1 == key_1) 
                   & (img_2 == key_2) 
                   & (img_3 == key_3)) # if [0,0,0]
            ret_img[masks] = origin

    #print('cat\n', unique_colors(categorized))
    #print('ret\n', unique_colors(ret_img))
    return ret_img
        

import unittest
class test_categorize_func(unittest.TestCase):
    #TODO: Add 2 classes case, 4 classes case, 10 classes case..
    def setUp(self):
        self.img = np.array(
            [[[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],], 
             [[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.],], 
             [[1.,1.,1.], [1.,1.,1.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.],], 
             [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.], [0.,1.,0.],],])

        self.img2 = np.array(
            [[[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],], 
             [[0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],], 
             [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],], 
             [[1.,1.,1.], [1.,1.,1.], [1.,1.,1.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.], [0.,0.,0.],],])

        self.rk_img = np.array(
            [[[0.,0.,0.], [0.,0.,0.]],  
             [[0.,0.,0.], [0.,0.,0.]],  
             [[0.,0.,1.], [0.,0.,1.]],  
             [[0.,0.,1.], [0.,0.,1.]],])

        self.predicted = np.array(
            [[[0.,0.,0.], [0.,0.,0.]],  
             [[0.,0.,0.], [0.,.3,0.]],  
             [[.2,.3,.9], [1.,.3,.2]],  
             [[.2,.4,.7], [.9,.4,.8]],])  
        self.rounded = np.array(
            [[[1.,0.,0.], [1.,0.,0.]],  
             [[1.,0.,0.], [0.,1.,0.]],  
             [[0.,0.,1.], [1.,0.,0.]],  
             [[0.,0.,1.], [1.,0.,0.]],])

        self.predicted_wk = np.array(
            [[[0.,0.], [0.,0.]],  
             [[0.,0.], [0.,.3]],  
             [[.2,.9], [1.,.3]],  
             [[.2,.7], [.9,.4]],])  
        self.rounded_wk = np.array(
            [[[1.,0.], [1.,0.]],  
             [[1.,0.], [0.,1.]],  
             [[0.,1.], [1.,0.]],  
             [[0.,1.], [1.,0.]],])

        self.wk_img = np.array(
            [[[0.,0.,0.], [0.,0.,0.]],  
             [[0.,0.,0.], [0.,0.,0.]],  
             [[1.,1.,1.], [1.,1.,1.]],  
             [[1.,1.,1.], [1.,1.,1.]],])

        self.err_img = np.array(
            [[[0.,0.,0.], [0.,0.,0.]],  
             [[0.,0.,0.], [0.,0.,0.]],  
             [[1.,1.,1.], [1.,1.,1.]],  
             [[1.,1.,1.], [1.,1.,1.]],])

    def test_predicted(self):
        self.assertTrue(np.alltrue(
            map_max_row(self.predicted) == self.rounded
        ))
        self.assertTrue(np.alltrue(
            map_max_row(self.predicted_wk) == self.rounded_wk
        ))


    def test_categorize3color(self):
        img = np.copy(self.img)
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(img == decategorized))

    def test_if_img_has_color_not_in_origin_map_then_exception(self):
        origin_map = {
            (0.0, 0.0, 1.0): [1., 0., 0.],
            (0.0, 1.0, 0.0): [0., 0., 1.],
            (1.0, 0.0, 0.0): [0., 0., 0.]
        }
        with self.assertRaises(AssertionError):
            categorized = categorize_with(self.err_img, origin_map)
            print('\n------>\n',categorized)

    def test_categorize_with_for_training_wk_img(self):
        origin_map = {
            (0.0, 1.0): [1., 1., 1.],
            (1.0, 0.0): [0., 0., 0.]
        }
        img = np.copy(self.wk_img)
        categorized = categorize_with(img, origin_map)
        expected,_ = categorize(img)

        print('------')
        print(categorized)
        print(expected)
        self.assertTrue(np.alltrue(categorized == expected))

    def test_rk_img_categorize_with(self):
        origin_map = {
            (0.0, 0.0, 1.0): [1., 0., 0.],
            (0.0, 1.0, 0.0): [0., 0., 1.],
            (1.0, 0.0, 0.0): [0., 0., 0.]
        }
        img = np.copy(self.rk_img)
        categorized = categorize_with(img, origin_map)

        #cv2.imshow('origin', self.rk_img)
        #cv2.imshow('categorized', categorized)
        #cv2.waitKey(0)

        self.assertEqual(categorized.shape[-1], 3)
        expected = np.array(
            [[[1.,0.,0.], [1.,0.,0.]],  
             [[1.,0.,0.], [1.,0.,0.]],  
             [[0.,1.,0.], [0.,1.,0.]],  
             [[0.,1.,0.], [0.,1.,0.]],])
        print(categorized)
        self.assertTrue(np.alltrue(categorized == expected))

        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(img == decategorized))

    def test_categorize4color(self):
        img = np.copy(self.img)
        img[0,0] = [2,2,2]
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(img == decategorized))

    def test_categorize2color(self):
        img = np.copy(self.img2)
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(img == decategorized))

    @unittest.skip('later')
    def test_categorize_real2color(self):
        origin = cv2.imread('./fixture/0_ans.png')
        img = origin.copy()
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        self.assertTrue(np.alltrue(origin == decategorized))
        cv2.imshow('origin',origin)
        cv2.imshow('decategorized',decategorized)
        cv2.waitKey(0)

    @unittest.skip('no t.png')
    def test_real_img(self):
        img = bgr_float32(cv2.imread('../t.bmp'))
        categorized,origin_map = categorize(img)
        decategorized = decategorize(categorized, origin_map)
        cv2.imshow('c',categorized)
        cv2.imshow('d',decategorized); cv2.waitKey(0)
        self.assertTrue(np.alltrue(img == decategorized))


import timeit
if __name__ == '__main__':
    unittest.main()

    im = bgr_float32(cv2.imread('./fixture/7_ans.png'))
    cv2.imshow('im', im); cv2.waitKey(0)
    categorized,omap = categorize(im)
    cv2.imshow('categorized', categorized); cv2.waitKey(0)
    decategorized = decategorize(categorized,omap)
    assert np.alltrue(im == decategorized)
    cv2.imshow('decategorized', decategorized); cv2.waitKey(0)

    im = bgr_float32(cv2.imread('./fixture/19.png'))
    categorized,omap = categorize(im)
    decategorized = decategorize(categorized,omap)
    assert np.alltrue(im == decategorized)
    #print('u',unique_colors(decategorized))

    '''
    start = timeit.default_timer()
    #Your statements here
    stop = timeit.default_timer()
    print('Time: ', stop - start)  
    print(splited_paths(['11','1','2','3','4','11','1','2','3','4',
                         '11','10','20','30','40','11','10','20','30','40',],
                        ['110','100','200','300','400','110','100','200','300','400',
                         '110','100','200','300','400','110','10','20','30','40',
                         ]))
    print(*splited_paths(human_sorted(file_paths('../boundary_data190125/image/')),
                        human_sorted(file_paths('../boundary_data190125/label/'))),sep='\n')

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
    '''
