import os
from os.path import join as pjoin
import shutil 
from pathlib import Path

import yaml
import cv2
import numpy as np
from imgaug import augmenters as iaa
from itertools import cycle, islice
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

from utils import human_sorted, splited_paths, file_paths, filename_ext
from utils import ElapsedTimer, now_time_str
from utils import load_imgs, bgr_float32, categorize, decategorize, unique_colors, categorize_with

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
            #print('wtf:',h,w,c)
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

def weights(mask):
    ''' all1s / channel1s. So, lesser 1s, bigger weight. '''
    channels = cv2.split(mask)
    all1s = np.sum(mask)
    ws = []
    for channel in channels:
        ch_sum = np.sum(channel)
        weight = (all1s / ch_sum) if ch_sum != 0 else 0
        ws.append( weight )
    return np.array(ws)

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
    #NUM_CLASSES = config['NUM_CLASSES'] # Don't care this! automatically set!
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

    if config.get('EVAL_TYPE') is not None:
        EVAL_TYPE = config['EVAL_TYPE']
    else:
        EVAL_TYPE = None

    DROPOUT = config.get('DROPOUT')

    assert EXPR_TYPE in ('manga', 'boundary_bioseg', 'bm_bioseg'), EXPR_TYPE

    aug,img_aug,mask_aug = None,None,None
    if EXPR_TYPE == 'manga':
        aug = crop_augmenter(
            BATCH_SIZE, IMG_SIZE, #num_channels=2,
            crop_before_augs=[
              iaa.Affine(
                rotate=(-3,3), shear=(-3,3), 
                scale={'x':(0.8,1.5), 'y':(0.8,1.5)},
                mode='reflect'),
            ]
        )

    elif EXPR_TYPE == 'boundary_bioseg' or EXPR_TYPE == 'bm_bioseg':
        aug = crop_augmenter(
            BATCH_SIZE, IMG_SIZE, #num_channels=2,
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
        origin_map = dataset['origin_map']

        stem = lambda p: Path(p).stem
        train_img_paths  = dataset['train_imgs']
        train_mask_paths = dataset['train_masks']
        for istem, mstem in zip(map(stem,train_img_paths), map(stem,train_mask_paths)):
            assert istem == mstem[:len(istem)], '{} != {}'.format(istem, mstem)

        valid_img_paths  = dataset['valid_imgs']
        valid_mask_paths = dataset['valid_masks']
        for istem, mstem in zip(map(stem,valid_img_paths), map(stem,valid_mask_paths)):
            assert istem == mstem[:len(istem)], '{} != {}'.format(istem, mstem)

        test_img_paths   = dataset['test_imgs']
        test_mask_paths  = dataset['test_masks']
        for istem, mstem in zip(map(stem,test_img_paths), map(stem,test_mask_paths)):
            assert istem == mstem[:len(istem)], '{} != {}'.format(istem, mstem)

    NUM_CLASSES = len(origin_map) 
    print(origin_map)
    print(len(origin_map))
    print('NUM_CLASSES', NUM_CLASSES)
    #label_paths = human_sorted(train_mask_paths + valid_mask_paths + test_mask_paths)
    #print('diff',set(label_paths) ^ set(human_sorted( file_paths('./borderNucleus190704/label_dirs/bn190704/'))))
    #print('not in dset',set(label_paths) - set(human_sorted( file_paths('./borderNucleus190704/label_dirs/bn190704/'))))

    train_imgs = list(load_imgs(train_img_paths))
    print('train image loaded')
    train_masks= list(map(lambda img: categorize_with(img,origin_map),
                          load_imgs(train_mask_paths)))
    print('train mask loaded')

    valid_imgs = list(load_imgs(valid_img_paths))
    print('valid image loaded')
    valid_masks= list(map(lambda img: categorize_with(img,origin_map),
                          load_imgs(valid_mask_paths)))
    print('valid mask loaded')

    test_imgs  = list(load_imgs(test_img_paths))
    print('test image loaded')
    test_masks = list(map(lambda img: categorize_with(img,origin_map),
                          load_imgs(test_mask_paths)))
    print('test mask loaded')

    #print('-------->', len(valid_imgs))
    #print('is categorize_with are failed?', np.unique(test_masks[0].reshape(-1, test_masks[0].shape[2]), axis=0))

    '''
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
    '''
    print('NUM_CLASSES', NUM_CLASSES)

    train_gen = batch_gen(train_imgs, train_masks, BATCH_SIZE, aug, img_aug, num_classes=NUM_CLASSES)
    valid_gen = batch_gen(valid_imgs, valid_masks, BATCH_SIZE, aug, img_aug, num_classes=NUM_CLASSES)
    test_gen  = batch_gen(test_imgs, test_masks, BATCH_SIZE, aug, img_aug, num_classes=NUM_CLASSES)

    # TODO:DEBUG
    '''
    mask= bgr_float32(cv2.imread(train_mask_paths[0]))
    categorized_mask = categorize_with(mask, origin_map)
    for ims,mas in valid_gen:
        for im,ma in zip(ims,mas):
            print(origin_map)
            print('ma',np.unique(ma.reshape(-1,ma.shape[2]), axis=0))
            ma = np.around(ma)
            print('rounded ma',np.unique(ma.reshape(-1,ma.shape[2]), axis=0))
            de = decategorize(ma,origin_map)
            print('de',np.unique(de.reshape(-1,de.shape[2]), axis=0))
            cv2.imshow('i',im)
            cv2.imshow('m',ma)
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
            filter_vec=FILTER_VEC,
            dropout=DROPOUT)
    elif MODEL == 'no_bn_unet':
        model = my_model.unet(
            num_classes=NUM_CLASSES,
            num_maxpool=NUM_MAXPOOL,
            num_filters=NUM_FILTERS,
            filter_vec=FILTER_VEC,
            basic_layer=my_model.layer_relu,
            dropout=DROPOUT)
    elif MODEL == 'bn_unet': # same as naive_unet
        model = my_model.unet(
            num_classes=NUM_CLASSES,
            num_maxpool=NUM_MAXPOOL,
            num_filters=NUM_FILTERS,
            filter_vec=FILTER_VEC,
            dropout=DROPOUT)
    else:
        print('not supported model!')
        exit()

    if config.get('LOSS') is None: #default :TODO:remove it!
        loss = weighted_categorical_crossentropy(weights[:NUM_CLASSES])
    elif config.get('LOSS') == 'wbce':
        loss = weighted_categorical_crossentropy(weights[:NUM_CLASSES])
    elif config.get('LOSS') == 'jaccard_distance':
        loss = jaccard_distance(NUM_CLASSES)
    elif config.get('LOSS') == 'weighted_jaccard_distance':
        # masks are already categorized.
        all_weights = sum(map(
            weights, train_masks + valid_masks + test_masks
        ))
        print(all_weights)
        loss = jaccard_distance(NUM_CLASSES, all_weights)

    model.compile(
        optimizer=OPTIMIZER,
        #optimizer='Adam', 
        loss=loss,
        #metrics=['accuracy']
        metrics=[jaccard_coefficient]
        #metrics=[mean_iou]
    )


    model_name = experiment_name +'_'+start_time 
    result_dir = model_name
    os.makedirs(result_dir)
    shutil.copyfile(experiment_yml_path, 
                    pjoin(result_dir, 'config_'+model_name+'.yml'))

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

    if EVAL_TYPE is None:
        if EXPR_TYPE == 'boundary_bioseg':
            #evaluator.eval_and_save_ultimate( model_path, DATASET_YML, experiment_yml_path)
            evaluator.eval_and_save( 
                model_path, DATASET_YML, experiment_yml_path)
                #model_path, DATASET_YML, experiment_yml_path,
                #train_imgs,train_masks, valid_imgs,valid_masks, test_imgs,test_masks)
        elif EXPR_TYPE == 'manga':
            #evaluator.eval_and_save_ultimate(
            evaluator.eval_and_save(
                model_path, DATASET_YML, experiment_yml_path
            )
        else:
            evaluator.eval_and_save_advanced_metric(
                model_path, DATASET_YML, experiment_yml_path
            )
    else: # 'ultimate'
        evaluator.eval_and_save_ultimate(
            model_path, DATASET_YML, experiment_yml_path
        )


    eval_time_str = eval_timer.elapsed_time()

        #train_imgs, train_masks, valid_imgs, valid_masks, test_imgs, test_masks)

import sys
if __name__ == '__main__':
    experiment_yml_path = sys.argv[1]
    main(experiment_yml_path) 
