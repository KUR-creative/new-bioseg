from tqdm import tqdm
from sklearn.metrics import confusion_matrix 
import traceback
import numpy as np
import cv2
from utils import decategorize
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

def save_imgs(origin_path_seq, img_seq, dst_dir_path, passed_origin_map=None):
    for imgpath, img in zip(origin_path_seq, img_seq):
        img_dstpath = make_new_path(dst_dir_path,imgpath)
        if passed_origin_map is not None:
            img = decategorize(img, passed_origin_map)
        #print(img_dstpath)
        cv2.imwrite(img_dstpath, bgr_uint8(img))

def make_result_paths(img_paths, result_dir):
    return list(map(
        lambda img_path: make_new_path(
            result_dir, img_path, 
            lambda name,ext: name+'_result'+ext),
        img_paths))

def evaluate(model, img, ans, modulo=32):
    segmap = segment(model, img, modulo)
    n,h,w,c = segmap.shape
    result = segmap.reshape((h,w,c))
    iou_score = iou(result, ans)
    return result, iou_score

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

    results = {'train_ious':[], 'valid_ious':[], 'test_ious':[],
            'mean_train_iou':0, 'mean_valid_iou':0, 'mean_test_iou':0}
    model = load_model(model_path, compile=False)
    # NOTE: because of keras bug, 'compile=False' is mendatory.
    for path, img, ans in tqdm(zip(train_result_paths, train_imgs, train_masks),
                               total=len(train_imgs)): 
        result, score = evaluate(model, img, ans, modulo)
        decategorized = decategorize(np.around(result),origin_map)
        uint8img = bgr_uint8(decategorized)
        #print(score,path)
        results['train_ious'].append( np.asscalar(np.mean(score)) )
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(valid_result_paths, valid_imgs, valid_masks),
                               total=len(valid_imgs)): 
        result, score = evaluate(model, img, ans, modulo)
        decategorized = decategorize(np.around(result),origin_map)
        uint8img = bgr_uint8(decategorized)
        #print(score,path)
        results['valid_ious'].append( np.asscalar(np.mean(score)) )
        cv2.imwrite(path, uint8img)
    print('----')
    for path, img, ans in tqdm(zip(test_result_paths, test_imgs,test_masks),
                               total=len(test_imgs)): 
        result, score = evaluate(model, img, ans, modulo)
        decategorized = decategorize(np.around(result),origin_map)
        uint8img = bgr_uint8(decategorized)
        #print(score,path)
        results['test_ious'].append( np.asscalar(np.mean(score)) )
        cv2.imwrite(path, uint8img)

    result_yml_name = os.path.join(result_dir,'[result]'+model_name) + '.yml'
    result_dict = dict(results, **dataset_dict)

    with open(result_yml_name,'w') as f:
        f.write(yaml.dump(result_dict))

    print('result images and ' + result_yml_name + ' are saved successfully!')

if __name__ == '__main__':
    pass
    #eval_and_save('dataset_2019-01-28_03_11_43.h5')
