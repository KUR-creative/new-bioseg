import os
from os.path import join as pjoin
import shutil 

import yaml
import cv2
import numpy as np
from imgaug import augmenters as iaa
from itertools import cycle, islice
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from utils import human_sorted, splited_paths, file_paths, filename_ext
from utils import ElapsedTimer, now_time_str
from utils import load_imgs, bgr_float32, categorize, decategorize, unique_colors

from segmentation_models import Unet
from segmentation_models.utils import set_trainable

from metrics import jaccard_coefficient, weighted_categorical_crossentropy, mean_iou
from metrics import jaccard_distance

import evaluator
import my_model

#-------- data augmentation --------
#def random_crop
def crop_augmenter(batch_size=4, crop_size=256, num_channels=3,
              crop_before_augs=(), crop_after_augs=()):
    n_img = batch_size
    size = crop_size
    #n_ch = num_channels
    def func_images(images, random_state, parents, hooks):
        """ random cropping """
        _,_,n_ch = images[0].shape
        ret_imgs = np.empty((n_img,size,size,n_ch))
        for idx,img in enumerate(images):
            h,w,c = img.shape
            y = random_state.randint(0, h - size)
            x = random_state.randint(0, w - size)
            ret_imgs[idx] = img[y:y+size,x:x+size].reshape((size,size,n_ch))
        return ret_imgs
        
    def func_heatmaps(heatmaps, random_state, parents, hooks):
        return heatmaps
    def func_keypoints(keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    aug_list = (\
       list(crop_before_augs) + 
       [iaa.Lambda(
         func_images=func_images,
         func_heatmaps=func_heatmaps,
         func_keypoints=func_keypoints)] +
       list(crop_after_augs)
    )
    print(aug_list)
    return iaa.Sequential(aug_list)
 
#-------- data preprocessing --------
def batch_gen(imgs, masks, batch_size, 
              both_aug=None, img_aug=None, mask_aug=None,
              num_classes=1):
    assert len(imgs) == len(masks)
    img_flow = cycle(imgs)
    mask_flow = cycle(masks)
    while True:
        img_batch = list(islice(img_flow,batch_size))
        mask_batch = list(islice(mask_flow,batch_size))

        if both_aug:
            aug_det = both_aug.to_deterministic()
            img_batch = aug_det.augment_images(img_batch)
            mask_batch = aug_det.augment_images(mask_batch)
        else: # no random crop - use square crop dataset
            img_batch = np.array(img_batch, np.float32)
            mask_batch = np.array(mask_batch, np.float32)

        if img_aug:
            img_batch = img_aug.augment_images(img_batch.astype(np.float32))
        if mask_aug:
            mask_batch = mask_aug.augment_images(mask_batch)

        yield img_batch, mask_batch

def bgr_weights(masks):
    n_all, n_bgr = 0, [0,0,0]
    for mask in masks:
        class_map = np.argmax(mask, axis=-1)
        classes,counts = np.unique(class_map,return_counts=True)
        for color,count in zip(classes,counts):
            n_bgr[color] += count
        n_all += sum(n_bgr)
    w_b = n_all / n_bgr[0]
    w_g = n_all / n_bgr[1]
    w_r = n_all / n_bgr[2]
    return w_b,w_g,w_r


def main(experiment_yml_path):
    start_time = now_time_str()

    with open(experiment_yml_path,'r') as f:
        print('now: ', experiment_yml_path)
        config = yaml.load(f)
    experiment_name = filename_ext(experiment_yml_path).name
    print(experiment_name, '\n', config)

    train_timer = ElapsedTimer(experiment_yml_path + ' training')
    #-------------------------------------------------------------------------------------------------
    EXPR_TYPE = config['EXPR_TYPE']
    NUM_CLASSES = config['NUM_CLASSES']
    IMG_SIZE = config['IMG_SIZE']
    BATCH_SIZE = config['BATCH_SIZE']
    NUM_EPOCHS = config['NUM_EPOCHS']
    STEPS_PER_EPOCH = config['STEPS_PER_EPOCH']
    DATASET_YML = config['DATASET_YML']
    MODEL = config['MODEL']
    NUM_MAXPOOL = config['NUM_MAXPOOL']
    TRANSFER_LEARNING = config['TRANSFER_LEARNING']
    NUM_FILTERS = config['NUM_FILTERS']
    OPTIMIZER = config['OPTIMIZER']
    PATIENCE = config['PATIENCE']
    MIN_LR = config['MIN_LR']
    if config.get('FILTER_VEC') is None:
        FILTER_VEC = (3,3,1)
    else:
        FILTER_VEC = tuple(config['FILTER_VEC'])

    aug,img_aug,mask_aug = None,None,None
    aug = crop_augmenter(
        BATCH_SIZE, IMG_SIZE, 1, 
        crop_before_augs=[
          iaa.Fliplr(0.5),
          iaa.Flipud(0.5),
          iaa.Affine(rotate=(-90,90),mode='reflect'),
        ],
        crop_after_augs=[
          iaa.ElasticTransformation(alpha=(100,200),sigma=14,mode='reflect'),
        ]
    )
    img_aug = iaa.Sequential([
        iaa.LinearContrast((0.8,1.2)),
        iaa.Sharpen((0.0,0.3)),
        iaa.AdditiveGaussianNoise(scale=(0.0,0.05)),
    ])

    if DATASET_YML is None: 
        # save DATASET_YML
        #NOTE: You must use ^ this paths to evaluate model.
        # dataset is randomly splited per trainings..
        (train_img_paths, train_mask_paths, 
         valid_img_paths, valid_mask_paths, 
         test_img_paths, test_mask_paths) \
            = splited_paths(human_sorted(file_paths('./boundary_data190125/image/')),
                            human_sorted(file_paths('./boundary_data190125/label/')))
        mask = bgr_float32(cv2.imread(train_mask_paths[0]))
        categorized_mask,origin_map = categorize(mask)
        print(origin_map)
        dataset_name = 'challenge_data_' + start_time + '.yml'
        dataset_dict = {
            'train_imgs':train_img_paths, 'train_masks':train_mask_paths,
            'valid_imgs':valid_img_paths, 'valid_masks':valid_mask_paths,
            'test_imgs':test_img_paths, 'test_masks':test_mask_paths,
            'origin_map':origin_map
        }
        with open(dataset_name,'w') as f:
            f.write(yaml.dump(dataset_dict))
        exit('not implemented')
    else:
        # load DATASET_YML
        with open(DATASET_YML,'r') as f:
            print('now: ', DATASET_YML)
            dataset = yaml.load(f)
        train_img_paths  = dataset['train_imgs']
        train_mask_paths = dataset['train_masks']
        valid_img_paths  = dataset['valid_imgs']
        valid_mask_paths = dataset['valid_masks']
        test_img_paths   = dataset['test_imgs']
        test_mask_paths  = dataset['test_masks']

    train_imgs = list(load_imgs(train_img_paths))
    train_masks= list(map(lambda img: categorize(img)[0],load_imgs(train_mask_paths)))
    valid_imgs = list(load_imgs(valid_img_paths))
    valid_masks= list(map(lambda img: categorize(img)[0],load_imgs(valid_mask_paths)))
    test_imgs  = list(load_imgs(test_img_paths))
    test_masks = list(map(lambda img: categorize(img)[0],load_imgs(test_mask_paths)))
    #print('-------->', len(valid_imgs))

    train_weights = bgr_weights(train_masks)
    valid_weights = bgr_weights(valid_masks)
    test_weights = bgr_weights(test_masks)
    print('train weights:', train_weights)
    print('valid weights:', valid_weights)
    print(' test weights:', test_weights)
    weights = np.array(train_weights) + np.array(valid_weights) + np.array(test_weights)
    print('total weights:', weights)
    weights = weights[:NUM_CLASSES]
    weights /= np.sum(weights)
    print('normalized weights:', weights)
    print('NUM_CLASSES', NUM_CLASSES)

    train_gen = batch_gen(train_imgs, train_masks, BATCH_SIZE, aug, img_aug, num_classes=NUM_CLASSES)
    valid_gen = batch_gen(valid_imgs, valid_masks, BATCH_SIZE, aug, img_aug, num_classes=NUM_CLASSES)
    test_gen  = batch_gen(test_imgs, test_masks, BATCH_SIZE, aug, img_aug, num_classes=NUM_CLASSES)

    # DEBUG
    '''
    mask= bgr_float32(cv2.imread(train_mask_paths[0]))
    categorized_mask,origin_map = categorize(mask)
    for ims,mas in valid_gen:
        for im,ma in zip(ims,mas):
            print(origin_map)
            print('ma',np.unique(ma.reshape(-1,ma.shape[2]), axis=0))
            ma = np.around(ma)
            print('rounded ma',np.unique(ma.reshape(-1,ma.shape[2]), axis=0))
            de = decategorize(ma,origin_map)
            print('de',np.unique(de.reshape(-1,de.shape[2]), axis=0))
            cv2.imshow('i',im)
            #cv2.imshow('m',ma)
            cv2.imshow('dm',de); cv2.waitKey(0)
    '''

    # prepare model
    if MODEL == 'resnet34':
        model = Unet(backbone_name='resnet34', encoder_weights='imagenet',
                     classes=3, activation='softmax',freeze_encoder=True)
    elif MODEL == 'naive_unet':
        model = my_model.unet(
            num_classes=NUM_CLASSES,
            num_maxpool=NUM_MAXPOOL,
            num_filters=NUM_FILTERS,
            filter_vec=FILTER_VEC)

    if config.get('LOSS') is None: #default :TODO:remove it!
        loss = weighted_categorical_crossentropy(weights[:NUM_CLASSES])
    elif config.get('LOSS') == 'jaccard_distance':
        loss = jaccard_distance(NUM_CLASSES)

    model.compile(
        optimizer=OPTIMIZER,
        #optimizer='Adam', 
        loss=loss,
        metrics=[jaccard_coefficient]
        #metrics=[mean_iou]
    )


    model_name = '['+experiment_name +']'+start_time 
    result_dir = model_name
    os.makedirs(result_dir)
    shutil.copyfile(experiment_yml_path, 
                    pjoin(result_dir, '[config]'+model_name+'.yml'))

    from keras.utils import plot_model
    plot_model(model, to_file=pjoin(result_dir,model_name+'.png'), 
               show_shapes=True)

    model_path = pjoin(result_dir,model_name) + '.h5'
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss',
                                       verbose=1, save_best_only=True)
    tboard = TensorBoard(log_dir='model_logs/'+model_name+'_logs',
                         batch_size=BATCH_SIZE, write_graph=False)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', 
                                  patience=PATIENCE,min_lr=MIN_LR)# NEW
    if TRANSFER_LEARNING:
        # train model(encoder only)
        model.fit_generator(train_gen, steps_per_epoch=-1, epochs=NUM_EPOCHS, #10 (dataset_2019-01-28_03_11_43)
                            validation_data=valid_gen, validation_steps=4,# 4 * 8 bat = 32(30 valid imgs)
                            callbacks=[model_checkpoint,tboard])
        set_trainable(model)
        # train model(whole network)
        model.fit_generator(train_gen, steps_per_epoch=-1, epochs=NUM_EPOCHS, #90 (dataset_2019-01-28_03_11_43)
                            validation_data=valid_gen, validation_steps=4,# 4 * 8 bat = 32(30 valid imgs)
                            callbacks=[model_checkpoint,tboard,reduce_lr])
    else:
        print('validation_steps =', ((len(valid_imgs)+BATCH_SIZE) // BATCH_SIZE))
        model.fit_generator(
            train_gen, steps_per_epoch=STEPS_PER_EPOCH, epochs=NUM_EPOCHS, #90 (dataset_2019-01-28_03_11_43)
            validation_data=valid_gen, validation_steps=((len(valid_imgs)+BATCH_SIZE) // BATCH_SIZE),
            callbacks=[model_checkpoint,tboard,reduce_lr])


    print('Model ' + model_name + ' is trained successfully!')

    #-------------------------------------------------------------------------------------------------
    train_time_str = train_timer.elapsed_time()

    eval_timer = ElapsedTimer(experiment_yml_path + ' evaluation')
    if EXPR_TYPE == 'boundary_bioseg':
        evaluator.eval_and_save(
            model_path, DATASET_YML, experiment_yml_path
        )
    else:
        evaluator.eval_and_save_advanced_metric(
            model_path, DATASET_YML, experiment_yml_path
        )
    eval_time_str = eval_timer.elapsed_time()

        #train_imgs, train_masks, valid_imgs, valid_masks, test_imgs, test_masks)

import sys
if __name__ == '__main__':
    experiment_yml_path = sys.argv[1]
    main(experiment_yml_path) 
