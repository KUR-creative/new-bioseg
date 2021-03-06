import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
import traceback
import numpy as np
import cv2
from utils import decategorize, categorize_with, bgr_uint8
from utils import human_sorted, file_paths, load_imgs, filename_ext
from utils import map_max_row
from metric import advanced_metric, my_old_metric
from pathlib import Path

def iou(y_true,y_pred,thr=0.5):
    ''' 
    Cacluate channel by channel intersection & union.
    And then calculate smoothed jaccard_coefficient.
    Finally, calculate jaccard_distance.
    '''
    axis = tuple(range(y_true.shape[-1]))
    y_true = (y_true >= thr).astype(np.uint8)
    y_pred = (y_pred >= thr).astype(np.uint8)

    intersection = y_pred * y_true
    sum_ = y_pred + y_true
    #print(y_true, y_pred)
    #print(intersection.shape)
    #print(sum_.shape)
    #print(axis)
    numerator = np.sum(intersection, axis)
    denominator = np.sum(sum_ - intersection, axis)
    return np.mean(numerator / denominator)
'''
def iou(y_true,y_pred,thr=0.5):
    y_true = (y_true.flatten() >= thr).astype(np.uint8)
    y_pred = (y_pred.flatten() >= thr).astype(np.uint8)
    cnfmat = confusion_matrix(y_true, y_pred, labels=[0, 1])
    intersection = np.diag(cnfmat)
    prediction = cnfmat.sum(axis=0) # 
    ground_truth = cnfmat.sum(axis=1)
    union = ground_truth + prediction - intersection
    return ((intersection + 0.001) / (union.astype(np.float32) + 0.001)).tolist()
    # TODO: why didn't np.mean and np.scalar??
'''

def get_segmap(segnet, img_batch, batch_size=1):
    try:
        segmap = segnet.predict(img_batch, batch_size)
    except Exception as e:
        print('wtf?:', e)
    return segmap

    #if segmap.shape[-1] == 4:
        #segmap = bgrk2bgr(segmap)

def modulo_padded(img, modulo=16):
    h,w = img.shape[:2]
    h_padding = (modulo - (h % modulo)) % modulo
    w_padding = (modulo - (w % modulo)) % modulo
    if len(img.shape) == 3:
        return np.pad(img, [(0,h_padding),(0,w_padding),(0,0)], mode='reflect')
    elif len(img.shape) == 2:
        return np.pad(img, [(0,h_padding),(0,w_padding)], mode='reflect')

def segment_or_oom(segnet, inp, modulo=16):
    ''' If image is too big, return None '''
    h,w = inp.shape[:2]

    img = modulo_padded(inp, modulo) 
    img_bat = np.expand_dims(img,0) 
    segmap = get_segmap(segnet, img_bat)#segnet.predict(img_bat, batch_size=1)#, verbose=1)
    segmap = np.squeeze(segmap[:,:h,:w,:], 0) #segmap[:,:h,:w,:].reshape((h,w,2))
    return segmap
    '''
    try:
        segmap = get_segmap(segnet, img_bat)#segnet.predict(img_bat, batch_size=1)#, verbose=1)
        segmap = np.squeeze(segmap[:,:h,:w,:], 0) #segmap[:,:h,:w,:].reshape((h,w,2))
        return segmap
    except Exception as e: # ResourceExhaustedError:
        print(traceback.print_tb(e.__traceback__)); exit()
        print(img_shape,'OOM error: image is too big. (in segnet)')
        return None
    '''

"""
# old implementation
def segment_or_oom(segnet, inp, modulo=16):
    ''' If image is too big, return None '''
    org_h,org_w = inp.shape[:2]

    img = modulo_padded(inp, modulo) 
    img_shape = img.shape #NOTE grayscale?
    img_bat = img.reshape((1,) + img_shape) # size 1 batch
    #print('---->',img_bat.shape)
    try:
        segmap = get_segmap(segnet, img_bat)#segnet.predict(img_bat, batch_size=1)#, verbose=1)
        segmap = segmap[:,:org_h,:org_w,:]#.reshape((org_h,org_w)) # remove padding
        return segmap
    except Exception as e: # ResourceExhaustedError:
        print(traceback.print_tb(e.__traceback__)); exit()
        print(img_shape,'OOM error: image is too big. (in segnet)')
        return None
"""

size_limit = 4000000 # dev-machine
def segment(segnet, inp, modulo=16):
    ''' oom-free segmentation '''
    global size_limit
    
    h,w = inp.shape[:2] # 1 image, not batch.
    result = None
    if h*w < size_limit:
        result = segment_or_oom(segnet, inp, modulo)
        if result is None: # size_limit: Ok but OOM occur!
            size_limit = h*w
            print('segmentation size_limit =', size_limit, 'updated!')
    else:
        print('segmentation size_limit exceed! img_size =', 
              h*w, '>', size_limit, '= size_limit')

    if result is None: # exceed size_limit or OOM
        if h > w:
            upper = segment(segnet, inp[:h//2,:], modulo) 
            downer= segment(segnet, inp[h//2:,:], modulo)
            return np.concatenate((upper,downer), axis=0)
        else:
            left = segment(segnet, inp[:,:w//2], modulo)
            right= segment(segnet, inp[:,w//2:], modulo)
            return np.concatenate((left,right), axis=1)
    else:
        return result # image segmented successfully!


from utils import load_imgs, splited_paths, file_paths, human_sorted, bgr_float32, bgr_uint8
from utils import filename_ext

def make_new_path(dst_dir_path,old_path,
        new_path_rule=lambda name,ext: name + ext):
    name,ext = filename_ext(old_path)
    img_name_ext = new_path_rule(name,ext)
    return os.path.join(dst_dir_path,img_name_ext)

def save_imgs(origin_path_seq, img_seq, dst_dir_path, 
              origin_map=None, rule=lambda name,ext: name + ext):
    for imgpath, img in zip(origin_path_seq, img_seq):
        img_dstpath = make_new_path(dst_dir_path,imgpath,rule)
        if origin_map is not None:
            img = decategorize(img, origin_map)
        #print(img_dstpath)
        cv2.imwrite(img_dstpath, bgr_uint8(img))

def make_result_paths(img_paths, result_dir):
    return list(map(
        lambda img_path: make_new_path(
            result_dir, img_path, 
            lambda name,ext: name+'_result'+ext),
        img_paths))

def evaluate(model, img, ans, modulo=32, origin_map=None):
    segmap = segment(model, img, modulo)
    if len(segmap.shape) == 4:
        _,h,w,c = segmap.shape 
    elif len(segmap.shape) == 3:
        h,w,c = segmap.shape 
    result = segmap.reshape((h,w,c))
    if origin_map is not None:
        result = decategorize(result, origin_map)
    #print('img:', img.shape, 'modulo:', modulo, 
    #      'res:', result.shape, 'ans:', ans.shape)
    #print('res:',result.dtype,'ans:',ans.dtype)
    iou_score = iou(ans, result)
    return result, iou_score

def not_black2color(img, color=[1,1,1]):
    h,w,_ = img.shape
    for y in range(h):
        for x in range(w):
            if any(img[y][x]): # if color != [0,0,0]:
                img[y][x] = color

import timeit
def evaluate_ultimate(model, image, answer, modulo=32, origin_map=None):
    ''' evaluate decategorized predicted mask with ultimate answer '''
    #NOTE: calculate IoU using *black/white" result image
    #      (bw img is mapped from decategorized img)
    segmap = segment(model, image, modulo)
    if len(segmap.shape) == 4:
        _,h,w,c = segmap.shape 
    elif len(segmap.shape) == 3:
        h,w,c = segmap.shape 

    result = segmap.reshape((h,w,c))
    #cv2.imshow('free segmap',result); #cv2.waitKey(0)
    if origin_map is not None:
        #result = decategorize(np.around(result), origin_map)
        result = decategorize(map_max_row(result), origin_map)
    #cv2.imshow('decategorized',result); cv2.waitKey(0)
    ret_result = result.copy()

    # result: red,blue, black -> white, black
    img_b, img_g, img_r = np.rollaxis(result, axis=-1)
    for b,g,r in [(1,0,0), (0,1,0), (0,0,1)]:
        masks = ((img_b == b) 
                &(img_g == g) 
                &(img_r == r)) # if [0,0,0]
        result[masks] = [1.,1.,1.]

    # answer: red,blue, black -> white, black
    img_b, img_g, img_r = np.rollaxis(answer, axis=-1)
    for b,g,r in [(1,0,0), (0,1,0), (0,0,1)]:
        masks = ((img_b == b) 
                &(img_g == g) 
                &(img_r == r)) # if [0,0,0]
        answer[masks] = [1.,1.,1.]

    iou_score = iou(answer, result)
    return ret_result, iou_score

def eval_advanced_metric(model, img, ans, origin_map, modulo=32):
    segmap = segment(model, img, modulo)
    #print(segmap.shape)
    h,w,c = segmap.shape
    result = segmap.reshape((h,w,c))

    decategorized = decategorize(np.around(result),origin_map)

    ans = ans[:,:,0].reshape((h,w,1))
    gray = decategorized[:,:,0].reshape((h,w,1))

    f1_v2, dice_obj = advanced_metric(ans, gray)
    return decategorized, f1_v2, dice_obj

from keras.models import load_model
import yaml
import os
def eval_and_save(model_path, dataset_dict_path, experiment_yml_path,
                  train_imgs=None, train_masks=None,
                  valid_imgs=None, valid_masks=None,
                  test_imgs=None, test_masks=None):
    with open(experiment_yml_path,'r') as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])
    print('modulo =',modulo)
    with open(dataset_dict_path,'r') as f:
        print(dataset_dict_path)
        dataset_dict = yaml.load(f)
    
    train_img_paths = dataset_dict['train_imgs']
    train_mask_paths= dataset_dict['train_masks']
    valid_img_paths = dataset_dict['valid_imgs']
    valid_mask_paths= dataset_dict['valid_masks']
    test_img_paths  = dataset_dict['test_imgs']
    test_mask_paths = dataset_dict['test_masks']
    origin_map = dataset_dict['origin_map']

    model_name = filename_ext(model_path).name
    result_dir = model_name
    print(result_dir)
    train_result_dir = os.path.join(result_dir,'train')
    valid_result_dir = os.path.join(result_dir,'valid')
    test_result_dir  = os.path.join(result_dir,'test')

    #---- make result paths ----
    train_result_paths = make_result_paths(train_img_paths,train_result_dir)
    valid_result_paths = make_result_paths(valid_img_paths,valid_result_dir)
    test_result_paths  = make_result_paths(test_img_paths,test_result_dir)

    #---- save results ----
    os.makedirs(os.path.join(result_dir,'train'))
    os.makedirs(os.path.join(result_dir,'valid'))
    os.makedirs(os.path.join(result_dir,'test'))

    '''
    # why it didn't work? why so slow???
    if train_imgs is None: # not passed from arguments.
        train_imgs = list(load_imgs(train_img_paths))
        train_masks= list(load_imgs(train_mask_paths))
        valid_imgs = list(load_imgs(valid_img_paths))
        valid_masks= list(load_imgs(valid_mask_paths))
        test_imgs  = list(load_imgs(test_img_paths))
        test_masks = list(load_imgs(test_mask_paths))
        passed_origin_map = None
    else:
        passed_origin_map = origin_map
    '''
    train_imgs = load_imgs(train_img_paths)
    train_masks= load_imgs(train_mask_paths)
    valid_imgs = load_imgs(valid_img_paths)
    valid_masks= load_imgs(valid_mask_paths)
    test_imgs  = load_imgs(test_img_paths)
    test_masks = load_imgs(test_mask_paths)
    # save images and answers before calculate iou scores
    ans_rule = lambda name,ext: name + '_ans' + ext # ext[0] is '.' dot
    save_imgs(train_img_paths, train_imgs, train_result_dir)
    save_imgs(valid_img_paths, valid_imgs, valid_result_dir)
    save_imgs(test_img_paths,  test_imgs,  test_result_dir)
    save_imgs(train_mask_paths, train_masks, train_result_dir, rule=ans_rule)
    save_imgs(valid_mask_paths, valid_masks, valid_result_dir, rule=ans_rule)
    save_imgs(test_mask_paths,  test_masks,  test_result_dir, rule=ans_rule)
    # categorize answer masks for calculate iou score
    train_imgs = load_imgs(train_img_paths)
    train_masks= load_imgs(train_mask_paths)
    valid_imgs = load_imgs(valid_img_paths)
    valid_masks= load_imgs(valid_mask_paths)
    test_imgs  = load_imgs(test_img_paths)
    test_masks = load_imgs(test_mask_paths)
    train_masks= map(lambda img: categorize_with(img,origin_map), train_masks)
    valid_masks= map(lambda img: categorize_with(img,origin_map), valid_masks)
    test_masks = map(lambda img: categorize_with(img,origin_map), test_masks)

    results = {'train_ious':[], 'valid_ious':[], 'test_ious':[],
            'mean_train_iou':0, 'mean_valid_iou':0, 'mean_test_iou':0}
    model = load_model(model_path, compile=False)
    # NOTE: because of keras bug, 'compile=False' is mendatory.
    for path, img, ans in tqdm(zip(train_result_paths, train_imgs, train_masks),
                               total=len(train_img_paths)): 
        result, score = evaluate(model, img, ans, modulo)
        #print(score,path)
        val = np.asscalar(np.mean(score))
        val = 0 if np.isnan(val) else val
        results['train_ious'].append( val )

        np.save(Path(path).with_suffix('.npy'), result) # categorized, float32
        decategorized = decategorize(map_max_row(result),origin_map)
        uint8img = bgr_uint8(decategorized)
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(valid_result_paths, valid_imgs, valid_masks),
                               total=len(valid_img_paths)): 
        result, score = evaluate(model, img, ans, modulo)
        #print(score,path)
        val = np.asscalar(np.mean(score))
        val = 0 if np.isnan(val) else val
        results['valid_ious'].append( val )

        np.save(Path(path).with_suffix('.npy'), result) # categorized, float32
        decategorized = decategorize(map_max_row(result),origin_map)
        uint8img = bgr_uint8(decategorized)
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(test_result_paths, test_imgs,test_masks),
                               total=len(test_img_paths)): 
        result, score = evaluate(model, img, ans, modulo)
        #print(score,path)
        val = np.asscalar(np.mean(score))
        val = 0 if np.isnan(val) else val
        results['test_ious'].append( val )

        np.save(Path(path).with_suffix('.npy'), result) # categorized, float32
        decategorized = decategorize(map_max_row(result),origin_map)
        uint8img = bgr_uint8(decategorized)
        cv2.imwrite(path, uint8img)

    results['mean_train_iou'] = np.asscalar(np.mean( results['train_ious'] ))
    results['mean_valid_iou'] = np.asscalar(np.mean( results['valid_ious'] ))
    results['mean_test_iou' ] = np.asscalar(np.mean( results['test_ious' ] ))
    result_yml_name = os.path.join(result_dir,'[result]'+model_name) + '.yml'
    result_dict = dict(results, **dataset_dict)

    with open(result_yml_name,'w') as f:
        f.write(yaml.dump(result_dict))

    print('result images and ' + result_yml_name + ' are saved successfully!')

#TODO: add final evaluation to dataset yml.
def eval_and_save_ultimate(model_path, dataset_dict_path, experiment_yml_path,
                        train_imgs=None, train_masks=None,
                        valid_imgs=None, valid_masks=None,
                        test_imgs=None, test_masks=None):
    with open(experiment_yml_path,'r') as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])
    print('modulo =',modulo)
    with open(dataset_dict_path,'r') as f:
        print(dataset_dict_path)
        dataset_dict = yaml.load(f)
    
    train_img_paths = dataset_dict['train_imgs']
    train_mask_paths= dataset_dict['train_masks']
    valid_img_paths = dataset_dict['valid_imgs']
    valid_mask_paths= dataset_dict['valid_masks']
    test_img_paths  = dataset_dict['test_imgs']
    test_mask_paths = dataset_dict['test_masks']
    origin_map = dataset_dict['origin_map']

    model_name = filename_ext(model_path).name
    result_dir = model_name
    print(result_dir)
    train_result_dir = os.path.join(result_dir,'train')
    valid_result_dir = os.path.join(result_dir,'valid')
    test_result_dir  = os.path.join(result_dir,'test')

    #---- make result paths ----
    train_result_paths = make_result_paths(train_img_paths,train_result_dir)
    valid_result_paths = make_result_paths(valid_img_paths,valid_result_dir)
    test_result_paths  = make_result_paths(test_img_paths,test_result_dir)

    #---- save results ----
    os.makedirs(os.path.join(result_dir,'train'))#, exist_ok=True)
    os.makedirs(os.path.join(result_dir,'valid'))#, exist_ok=True)
    os.makedirs(os.path.join(result_dir,'test'))#, exist_ok=True)

    train_imgs = load_imgs(train_img_paths)
    train_masks= load_imgs(train_mask_paths)
    valid_imgs = load_imgs(valid_img_paths)
    valid_masks= load_imgs(valid_mask_paths)
    test_imgs  = load_imgs(test_img_paths)
    test_masks = load_imgs(test_mask_paths)
    # save images and answers before calculate iou scores
    ans_rule = lambda name,ext: name + '_ans' + ext # ext[0] is '.' dot
    save_imgs(train_img_paths, train_imgs, train_result_dir)
    save_imgs(valid_img_paths, valid_imgs, valid_result_dir)
    save_imgs(test_img_paths,  test_imgs,  test_result_dir)
    save_imgs(train_mask_paths, train_masks, train_result_dir, rule=ans_rule)
    save_imgs(valid_mask_paths, valid_masks, valid_result_dir, rule=ans_rule)
    save_imgs(test_mask_paths,  test_masks,  test_result_dir, rule=ans_rule)
    
    train_imgs = load_imgs(train_img_paths)
    train_masks= load_imgs(train_mask_paths)
    valid_imgs = load_imgs(valid_img_paths)
    valid_masks= load_imgs(valid_mask_paths)
    test_imgs  = load_imgs(test_img_paths)
    test_masks = load_imgs(test_mask_paths)
    #train_masks= map(lambda img: categorize_with(img,origin_map), train_masks)
    #valid_masks= map(lambda img: categorize_with(img,origin_map), valid_masks)
    #test_masks = map(lambda img: categorize_with(img,origin_map), test_masks)

    results = {'train_ious':[], 'valid_ious':[], 'test_ious':[],
            'mean_train_iou':0, 'mean_valid_iou':0, 'mean_test_iou':0}
    model = load_model(model_path, compile=False)
    # NOTE: because of keras bug, 'compile=False' is mendatory.
    for path, img, ans in tqdm(zip(train_result_paths, train_imgs, train_masks),
                               total=len(train_img_paths)): 
        result, score = evaluate_ultimate(model, img, ans, modulo, origin_map)
        #print(score,path)
        val = np.asscalar(np.mean(score))
        val = 0 if np.isnan(val) else val
        results['train_ious'].append( val )

        uint8img = bgr_uint8(result)
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(valid_result_paths, valid_imgs, valid_masks),
                               total=len(valid_img_paths)): 
        result, score = evaluate_ultimate(model, img, ans, modulo, origin_map)
        #print(score,path)
        val = np.asscalar(np.mean(score))
        val = 0 if np.isnan(val) else val
        results['valid_ious'].append( val )

        uint8img = bgr_uint8(result)
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(test_result_paths, test_imgs,test_masks),
                               total=len(test_img_paths)): 
        result, score = evaluate_ultimate(model, img, ans, modulo, origin_map)
        #print(score,path)
        val = np.asscalar(np.mean(score))
        val = 0 if np.isnan(val) else val
        results['test_ious'].append( val )

        uint8img = bgr_uint8(result)
        cv2.imwrite(path, uint8img)

    results['mean_train_iou'] = np.asscalar(np.mean( results['train_ious'] ))
    results['mean_valid_iou'] = np.asscalar(np.mean( results['valid_ious'] ))
    results['mean_test_iou' ] = np.asscalar(np.mean( results['test_ious' ] ))
    result_yml_name = os.path.join(result_dir,'[result]'+model_name) + '.yml'
    result_dict = dict(results, **dataset_dict)

    with open(result_yml_name,'w') as f:
        f.write(yaml.dump(result_dict))

    print('result images and ' + result_yml_name + ' are saved successfully!')

from utils import unique_colors
def eval_and_save_advanced_metric(
        model_path, dataset_dict_path, experiment_yml_path,
        train_imgs=None, train_masks=None,
        valid_imgs=None, valid_masks=None,
        test_imgs=None, test_masks=None):
    with open(experiment_yml_path,'r') as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])
    print('modulo =',modulo)
    with open(dataset_dict_path,'r') as f:
        print(dataset_dict_path)
        dataset_dict = yaml.load(f)
    
    train_img_paths = dataset_dict['train_imgs']
    train_mask_paths= dataset_dict['train_masks']
    valid_img_paths = dataset_dict['valid_imgs']
    valid_mask_paths= dataset_dict['valid_masks']
    test_img_paths  = dataset_dict['test_imgs']
    test_mask_paths = dataset_dict['test_masks']
    origin_map = dataset_dict['origin_map']

    model_name = filename_ext(model_path).name
    result_dir = model_name
    print(result_dir)
    train_result_dir = os.path.join(result_dir,'train')
    valid_result_dir = os.path.join(result_dir,'valid')
    test_result_dir  = os.path.join(result_dir,'test')

    #---- make result paths ----
    train_result_paths = make_result_paths(train_img_paths,train_result_dir)
    valid_result_paths = make_result_paths(valid_img_paths,valid_result_dir)
    test_result_paths  = make_result_paths(test_img_paths,test_result_dir)

    #---- save results ----
    os.makedirs(os.path.join(result_dir,'train'))
    os.makedirs(os.path.join(result_dir,'valid'))
    os.makedirs(os.path.join(result_dir,'test'))

    passed_origin_map = None
    train_imgs = list(load_imgs(train_img_paths))
    train_masks= list(load_imgs(train_mask_paths))
    valid_imgs = list(load_imgs(valid_img_paths))
    valid_masks= list(load_imgs(valid_mask_paths))
    test_imgs  = list(load_imgs(test_img_paths))
    test_masks = list(load_imgs(test_mask_paths))

    save_imgs(train_img_paths, train_imgs, train_result_dir, passed_origin_map)
    save_imgs(train_mask_paths,train_masks,train_result_dir, passed_origin_map)
    save_imgs(valid_img_paths, valid_imgs, valid_result_dir, passed_origin_map)
    save_imgs(valid_mask_paths,valid_masks,valid_result_dir, passed_origin_map)
    save_imgs(test_img_paths, test_imgs, test_result_dir, passed_origin_map)
    save_imgs(test_mask_paths,test_masks,test_result_dir, passed_origin_map)

    results = {
        'train_f1':[], 'valid_f1':[], 'test_f1':[],
        'train_dice_obj':[], 'valid_dice_obj':[], 'test_dice_obj':[],
        'mean_train_iou':0, 'mean_valid_iou':0, 'mean_test_iou':0
    }
    model = load_model(model_path, compile=False)
    # NOTE: because of keras bug, 'compile=False' is mendatory.
    for path, img, ans in tqdm(zip(train_result_paths, train_imgs, train_masks),
                               total=len(train_imgs)): 
        #cv2.imshow('ans?', ans);cv2.waitKey(0)
        result, f1, dice_obj = eval_advanced_metric(model, img, ans, origin_map, modulo)
        results['train_f1'].append(f1)
        results['train_dice_obj'].append(dice_obj)
        uint8img = bgr_uint8(result)
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(valid_result_paths, valid_imgs, valid_masks),
                               total=len(valid_imgs)): 
        result, f1, dice_obj = eval_advanced_metric(model, img, ans, origin_map, modulo)
        results['valid_f1'].append(f1) 
        results['valid_dice_obj'].append(dice_obj)
        uint8img = bgr_uint8(result)
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(test_result_paths, test_imgs,test_masks),
                               total=len(test_imgs)): 
        result, f1, dice_obj = eval_advanced_metric(model, img, ans, origin_map, modulo)
        results['test_f1'].append(f1)
        results['test_dice_obj'].append(dice_obj)
        uint8img = bgr_uint8(result)
        cv2.imwrite(path, uint8img)

    result_yml_name = os.path.join(result_dir,'[result]'+model_name) + '.yml'
    result_dict = dict(results, **dataset_dict)

    with open(result_yml_name,'w') as f:
        f.write(yaml.dump(result_dict))

    print('result images and ' + result_yml_name + ' are saved successfully!')

def eval_postprocessed(pred_dir, ans_dir):
    def result_tuples(names,predictions,answers):
        return (
            (name,)+advanced_metric(ans,pred) 
            for name,pred,ans
            in zip(names,predictions,answers)
        )
        '''
        #DEBUG
        for pred,ans in zip(predictions,answers):
            cv2.imshow('pred', pred)
            cv2.imshow('ans', ans); cv2.waitKey(0)
        exit()
        '''

    def ret(pred_paths, ans_paths):
        names = map(lambda p:filename_ext(p).name, pred_paths)
        predictions = load_imgs(pred_paths, cv2.IMREAD_GRAYSCALE)
        answers = load_imgs(ans_paths, cv2.IMREAD_GRAYSCALE)

        return result_tuples(names,predictions,answers)
        '''
        for name, f1, dice_obj in gen:
            print(name, f1, dice_obj, sep='\t')
        '''

    train_pred_dir = os.path.join(pred_dir,'train')
    train_ans_dir  = os.path.join(ans_dir, 'train')
    valid_pred_dir = os.path.join(pred_dir,'valid')
    valid_ans_dir  = os.path.join(ans_dir, 'valid')
    test_pred_dir  = os.path.join(pred_dir,'test')
    test_ans_dir   = os.path.join(ans_dir, 'test')

    is_result = lambda s:'_result' in str(s)
    train_pred_paths = human_sorted(filter(is_result,file_paths(train_pred_dir)))
    train_ans_paths  = human_sorted(file_paths(train_ans_dir))
    valid_pred_paths = human_sorted(filter(is_result,file_paths(valid_pred_dir)))
    valid_ans_paths  = human_sorted(file_paths(valid_ans_dir))
    test_pred_paths  = human_sorted(filter(is_result,file_paths(test_pred_dir)))
    test_ans_paths   = human_sorted(file_paths(test_ans_dir))
    #print(*list(zip(train_pred_paths,train_ans_paths)),sep='\n'); exit()

    # tups = tuples<name,f1,dice_obj>
    train_tups = ret(train_pred_paths, train_ans_paths)
    valid_tups = ret(valid_pred_paths, valid_ans_paths)
    test_tups  = ret(test_pred_paths,  test_ans_paths)
    return train_tups,valid_tups,test_tups

    '''
    names = map(lambda p:filename_ext(p).name, pred_paths)
    predictions = load_imgs(pred_paths, cv2.IMREAD_GRAYSCALE)
    answers = load_imgs(ans_paths, cv2.IMREAD_GRAYSCALE)

    gen = result_tuples(names,predictions,answers)
    for name, f1, dice_obj in gen:
        print(name, f1, dice_obj, sep='\t')
    '''
    #return pred_paths,ans_paths

def eval_old_and_new(pred_dir, ans_dir):
    def result_tuples(names,predictions,answers):
        return (
            (name,) + my_old_metric(ans,pred) + advanced_metric(ans,pred) 
            for name,pred,ans
            in zip(names,predictions,answers)
        )
        '''
        #DEBUG
        for pred,ans in zip(predictions,answers):
            cv2.imshow('pred', pred)
            cv2.imshow('ans', ans); cv2.waitKey(0)
        exit()
        '''

    def ret(pred_paths, ans_paths):
        names = map(lambda p:filename_ext(p).name, pred_paths)
        predictions = load_imgs(pred_paths, cv2.IMREAD_GRAYSCALE)
        answers = load_imgs(ans_paths, cv2.IMREAD_GRAYSCALE)

        return result_tuples(names,predictions,answers)
        '''
        for name, f1, dice_obj in gen:
            print(name, f1, dice_obj, sep='\t')
        '''

    train_pred_dir = os.path.join(pred_dir,'train')
    train_ans_dir  = os.path.join(ans_dir, 'train')
    valid_pred_dir = os.path.join(pred_dir,'valid')
    valid_ans_dir  = os.path.join(ans_dir, 'valid')
    test_pred_dir  = os.path.join(pred_dir,'test')
    test_ans_dir   = os.path.join(ans_dir, 'test')

    is_result = lambda s:'_result' in str(s)
    train_pred_paths = human_sorted(filter(is_result,file_paths(train_pred_dir)))
    train_ans_paths  = human_sorted(file_paths(train_ans_dir))
    valid_pred_paths = human_sorted(filter(is_result,file_paths(valid_pred_dir)))
    valid_ans_paths  = human_sorted(file_paths(valid_ans_dir))
    test_pred_paths  = human_sorted(filter(is_result,file_paths(test_pred_dir)))
    test_ans_paths   = human_sorted(file_paths(test_ans_dir))
    #print(*list(zip(train_pred_paths,train_ans_paths)),sep='\n'); exit()

    # tups = tuples<name,f1,dice_obj>
    train_tups = ret(train_pred_paths, train_ans_paths)
    valid_tups = ret(valid_pred_paths, valid_ans_paths)
    test_tups  = ret(test_pred_paths,  test_ans_paths)
    return train_tups,valid_tups,test_tups

import fp
def eval4paper(res_dirpath, res_postfix, ans_dirpath, ans_postfix='ans'):
    def imgpaths(dirpath, postfix):
        return fp.pipe(
            file_paths,
            fp.cfilter(lambda name: postfix in name),
            human_sorted,
        )(dirpath)
    res_paths = imgpaths(res_dirpath,res_postfix)
    ans_paths = imgpaths(ans_dirpath,ans_postfix)

    resultseq = load_imgs(res_paths)
    answerseq = load_imgs(ans_paths)

    #for res,ans in zip(res_paths,ans_paths):
        #print(res,ans)

    #for res,ans in zip(resultseq,answerseq):
        #pass
        #cv2.imshow('res', res); cv2.imshow('ans', ans); cv2.waitKey(0)

    ious = fp.lmap( iou, resultseq,answerseq )
    print(ious)
    print(np.mean(ious))

import fp
from pathlib import Path
import sys
if __name__ == '__main__':
    '''
    eval_and_save_ultimate(
        './olds/border_nucleus/border_nucleus_190705_173231/border_nucleus_190705_173231.h5',
        './olds/border_nucleus/border_nucleus_190705_173231/config_border_nucleus_190705_173231.yml', 
        './experiments/borderNucleus190704/border_nucleus.yml'
    )
    exit()
    '''
    dataset_dict_path = './borderNucleus190704/borderNucleus190704.yml'
    with open(dataset_dict_path,'r') as f:
        print(dataset_dict_path)
        dataset_dict = yaml.load(f)
    origin_map = dataset_dict['origin_map']

    experiment_yml_path = './experiments/borderNucleus190704/border_nucleus.yml'
    with open(experiment_yml_path,'r') as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])

    model_path = './olds/border_nucleus/border_nucleus_190705_173231/border_nucleus_190705_173231.h5'
    model = load_model(model_path, compile=False)

    img_paths = file_paths('./eval_data/')
    imgs = load_imgs(img_paths)
    for image,name in zip(imgs, file_paths('./eval_data/')):
        print(name)
        result = segment(model, image, modulo)
        decategorized = decategorize(map_max_row(result), origin_map)
        #cv2.imshow('ma', decategorized)
        #cv2.imshow('im', image)
        #cv2.waitKey(0)
        cv2.imwrite('out_data/' + name, bgr_uint8(decategorized))
    exit()

    eval_and_save_ultimate(
        './test_nucleus_190705_152946/test_nucleus_190705_152946.h5', 
        './borderNucleus190704/borderNucleus190704.yml', 
        './experiments/borderNucleus190704/test/test_nucleus.yml'
    )
    exit()

    conf_path = './[wk200f16d4fv31_3000eph]2019-04-30_10_17_29/[config][wk200f16d4fv31_3000eph]2019-04-30_10_17_29.yml'
    model_path = './[wk200f16d4fv31_3000eph]2019-04-30_10_17_29/[wk200f16d4fv31_3000eph]2019-04-30_10_17_29.h5'
    with open(conf_path) as f:
        config = yaml.load(f)

    eval_and_save_ultimate(
        model_path,
        config['DATASET_YML'],
        conf_path
    )
        

    exit()#-----------------------------------

    conf_path = './eval4paper/wk3000eph.yml'
    with open(conf_path) as f:
        config = yaml.load(f)
    model_path = './eval4paper/wk3000eph.h5'
    modulo = 2**(config['NUM_MAXPOOL'])

    with open(config['DATASET_YML'],'r') as f:
        origin_map = yaml.load(f)['origin_map']
    model = load_model(model_path, compile=False)

    img_paths = human_sorted(file_paths('./eval4paper/dset/'))
    imgseq = load_imgs(img_paths)

    resultseq = map(
        fp.pipe(
            lambda img: segment(model, img, modulo)[0],
            np.around,
            lambda mask: decategorize(mask, origin_map), 
            bgr_uint8
        ),
        imgseq
    )

    dstdir = './eval4paper/mask'
    def mk_outpath(srcpath):
        return str(
            Path(dstdir) / Path(srcpath).parts[-1]
        )
    dstpaths = fp.lmap(mk_outpath, img_paths)

    for img,dstpath in tqdm(zip(resultseq, dstpaths), 
                            total=len(dstpaths)):
        cv2.imwrite(dstpath,img)
        #cv2.imshow(dstpath,img); cv2.waitKey(0)

    exit()
    ######################
    conf_path = '/home/kur/dev/szmc/segnet/old/EXPR1/[rbk200f16d4fv31]2019-04-25_05_47_31/[config][rbk200f16d4fv31]2019-04-25_05_47_31.yml'
    #^~~~~ rbk

    #conf_path = '/home/kur/dev/szmc/segnet/[wk200f16d4fv31]2019-04-28_19_04_51/[config][wk200f16d4fv31]2019-04-28_19_04_51.yml'
    #^~~~~ best
    with open(conf_path) as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])
    model_path = '/home/kur/dev/szmc/segnet/old/EXPR1/[rbk200f16d4fv31]2019-04-25_05_47_31/[rbk200f16d4fv31]2019-04-25_05_47_31.h5'
    #^~~~~ rbk

    #model_path = '/home/kur/dev/szmc/segnet/[wk200f16d4fv31]2019-04-28_19_04_51/[wk200f16d4fv31]2019-04-28_19_04_51.h5'
    #^~~~~ best
    eval_and_save_ultimate(
        model_path, config['DATASET_YML'], conf_path
    )
    #eval4paper('./not_mine/test_data/', 'mask', './not_mine/test')
    exit()
    '''
    with open('/home/kur/dev/szmc/segnet/[rbk200f32d4fv31]2019-04-24_20_37_44/[config][rbk200f32d4fv31]2019-04-24_20_37_44.yml','r') as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])
    model_path = '/home/kur/dev/szmc/segnet/[rbk200f32d4fv31]2019-04-24_20_37_44/[rbk200f32d4fv31]2019-04-24_20_37_44.h5'
    model = load_model(model_path, compile=False)
    # r,g,b -> w k
    ans = bgr_float32(cv2.imread('./fixture/2_ans.png'))
    img = bgr_float32(cv2.imread('./fixture/2.png'))
    omap = {
        (0., 0., 1.): [1.,0.,0.],
        (0., 1., 0.): [0.,0.,1.],
        (1., 0., 0.): [0.,0.,0.],
    }
    _,iou_score = evaluate_ultimate(
        model, img, ans, modulo, omap)
    print(iou_score,'vs',0.7133488070819389)
    assert iou_score == 0.7133488070819389

    res = bgr_float32(cv2.imread('./fixture/2_result.png'))
    _,iou_score = evaluate_ultimate(
        model, img, res, modulo, omap)
    print(iou_score,'vs',1.0)
    assert iou_score == 1.0
    '''

    #--------
    print('---------------------------------')
    with open('/home/kur/dev/szmc/segnet/[manga_test_wk50]2019-04-25_04_02_38/[config][manga_test_wk50]2019-04-25_04_02_38.yml','r') as f:
        config = yaml.load(f)
    modulo = 2**(config['NUM_MAXPOOL'])
    model_path = '/home/kur/dev/szmc/segnet/[manga_test_wk50]2019-04-25_04_02_38/[manga_test_wk50]2019-04-25_04_02_38.h5'
    model = load_model(model_path, compile=False)
    # r,g,b -> w k
    omap = {
        (0., 1.): [1.,1.,1.],
        (1., 0.): [0.,0.,0.],
    }
    img = bgr_float32(cv2.imread('./fixture/0.png'))
    ans = bgr_float32(cv2.imread('./fixture/0_ans.png'))
    res = bgr_float32(cv2.imread('./fixture/0_result.png'))
    result,iou_score = evaluate_ultimate(
        model, img, ans, modulo, omap)
    print('iou 2nd:',iou_score)
    cv2.imshow('result', result); cv2.waitKey(0)

    result,iou_score = evaluate_ultimate(
        model, img, res, modulo, omap)
    print(iou_score,'vs',1.0)
    assert iou_score == 1.0
    print('iou 2-2nd:',iou_score)

    print('iou rnd 2: 2nd:',iou_score)
    cv2.imshow('result 2', result); cv2.waitKey(0)
    exit()

    if len(sys.argv) == 1 + 2:
        train_tups,valid_tups,test_tups \
            = eval_postprocessed(sys.argv[1], sys.argv[2])
            #= eval_postprocessed('./eval_postprocessed/thick2/', './eval_postprocessed/GT/')
        print('-------train-------')
        for name,f1,dice_obj in train_tups:
            print(name, f1, dice_obj, sep=',')

        print('-------valid-------')
        for name,f1,dice_obj in valid_tups:
            print(name, f1, dice_obj, sep=',')

        print('-------test-------')
        for name,f1,dice_obj in test_tups:
            print(name, f1, dice_obj, sep=',')
    if len(sys.argv) == 1 + 3 and sys.argv[3] == 'old':
        train_tups,valid_tups,test_tups \
            = eval_old_and_new(sys.argv[1], sys.argv[2])
            #= eval_postprocessed('./eval_postprocessed/thick2/', './eval_postprocessed/GT/')
        print('-------train-------')
        #for name, old_f1,old_dice, f1,dice_obj in train_tups:
            #print(name, old_f1,old_dice, f1,dice_obj, sep=',')

        print('-------valid-------')
        for name, old_f1,old_dice, f1,dice_obj in valid_tups:
            print(name, old_f1,old_dice, f1,dice_obj, sep=',')

        print('-------test-------')
        for name, old_f1,old_dice, f1,dice_obj in test_tups:
            print(name, old_f1,old_dice, f1,dice_obj, sep=',')
    elif len(sys.argv) == 1 + 3: # eval with advanced_metric
        model_path = sys.argv[1]
        dataset_dict_path = sys.argv[2]
        experiment_yml_path = sys.argv[3]
        eval_and_save_advanced_metric(
            model_path, dataset_dict_path, experiment_yml_path)
    else:
        print('Usage: python evaluator.py predict_dirpath GT_dirpath')  
        print('Usage: python evaluator.py model_path dataset_dict_path experiment_yml_path')  
        print('Usage: python evaluator.py model_path dataset_dict_path old')  


    '''
    pp,ap = eval_postprocessed('./eval_postprocessed/thick2/', './eval_postprocessed/GT/')
    for p,a in zip(pp,ap):
        print(p,a, os.path.basename(p)[:-12] == os.path.basename(a)[:-9])
    pass
    '''
    #eval_and_save('dataset_2019-01-28_03_11_43.h5')
