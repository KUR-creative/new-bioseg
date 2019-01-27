import numpy as np
from imgaug import augmenters as iaa
from itertools import cycle, islice
from utils import load_imgs, splited_paths, file_paths, human_sorted, bgr_float32
from utils import now_time_str
from utils import categorize, decategorize, unique_colors
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau

#-------- data augmentation --------
#def random_crop
def augmenter(batch_size=4, crop_size=256, num_channels=3,
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
            img_batch = img_aug.augment_images(img_batch)
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

BATCH_SIZE = 8
IMG_SIZE = 256
num_classes = 3

aug,img_aug,mask_aug = None,None,None
aug = augmenter(BATCH_SIZE, IMG_SIZE, 1, 
        crop_before_augs=[
          iaa.Fliplr(0.5),
          iaa.Flipud(0.5),
          iaa.Affine(rotate=(-180,180),mode='reflect'),
        ],
        crop_after_augs=[
          iaa.ElasticTransformation(alpha=(100,200),sigma=14,mode='reflect'),
        ]
      )
(train_img_paths, train_mask_paths, 
 valid_img_paths, valid_mask_paths, 
 test_img_paths, test_mask_paths) \
    = splited_paths(human_sorted(file_paths('../boundary_data190125/image/')),
                    human_sorted(file_paths('../boundary_data190125/label/')))
#NOTE: You must use ^ this paths to evaluate model.
# dataset is randomly splited per trainings..
train_imgs = list(load_imgs(train_img_paths))
train_masks= list(map(lambda img: categorize(img)[0],load_imgs(train_mask_paths)))
valid_imgs = list(load_imgs(valid_img_paths))
valid_masks= list(map(lambda img: categorize(img)[0],load_imgs(valid_mask_paths)))
test_imgs  = list(load_imgs(test_img_paths))
test_masks = list(map(lambda img: categorize(img)[0],load_imgs(test_mask_paths)))

train_weights = bgr_weights(train_masks)
valid_weights = bgr_weights(valid_masks)
test_weights = bgr_weights(test_masks)
print('train weights:', train_weights)
print('valid weights:', valid_weights)
print(' test weights:', test_weights)
weights = np.array(train_weights) + np.array(valid_weights) + np.array(test_weights)
print('total weights:', weights)
weights /= np.sum(weights)
print('normalized weights:', weights)

train_gen = batch_gen(train_imgs, train_masks, BATCH_SIZE, aug, img_aug, num_classes=num_classes)
valid_gen = batch_gen(valid_imgs, valid_masks, BATCH_SIZE, aug, img_aug, num_classes=num_classes)
test_gen  = batch_gen(test_imgs, test_masks, BATCH_SIZE, aug, img_aug, num_classes=num_classes)

'''
# DEBUG
import cv2
mask= bgr_float32(cv2.imread(train_mask_paths[0]))
categorized_mask,origin_map = categorize(mask)
print(origin_map)
for ims,mas in test_gen:
    for im,ma in zip(ims,mas):
        print('ma',np.unique(ma.reshape(-1,ma.shape[2]), axis=0))
        ma = np.around(ma)
        print('rounded ma',np.unique(ma.reshape(-1,ma.shape[2]), axis=0))
        de = decategorize(ma,origin_map)
        print('de',np.unique(de.reshape(-1,de.shape[2]), axis=0))
        cv2.imshow('i',im)
        cv2.imshow('m',ma)
        cv2.imshow('dm',de); cv2.waitKey(0)
        #if num_classes == 4:
        #    print(ma.shape)
        #    cv2.imshow('m',bgrk2bgr(ma)); cv2.waitKey(0)
        #else:
        #    cv2.imshow('m',ma); cv2.waitKey(0)
'''

from segmentation_models import Unet
from segmentation_models.utils import set_trainable

# prepare model
model = Unet(backbone_name='resnet34', encoder_weights='imagenet',
             classes=3, activation='softmax',freeze_encoder=True)
from metrics import jaccard_coefficient, weighted_categorical_crossentropy
model.compile(
    optimizer='Adam', 
    loss=weighted_categorical_crossentropy(weights),#'binary_crossentropy', 
    metrics=[jaccard_coefficient]#['binary_accuracy']
)


#from keras.utils import plot_model
start_time = now_time_str()
model_name = 'tmp_model_' + start_time + '.h5'

import yaml
import cv2
mask = bgr_float32(cv2.imread(train_mask_paths[0]))
categorized_mask,origin_map = categorize(mask)
print(origin_map)
dataset_name = 'dataset_' + start_time + '.yml'
dataset_dict = {
    'train_imgs':train_img_paths, 'train_masks':train_mask_paths,
    'valid_imgs':valid_img_paths, 'valid_masks':valid_mask_paths,
    'test_imgs':test_img_paths, 'test_masks':test_mask_paths,
    'origin_map':origin_map
}
with open(dataset_name,'w') as f:
    f.write(yaml.dump(dataset_dict))
#NOTE: You must use this ^ paths to evaluate model.

#plot_model(model, to_file='pt_model.png', show_shapes=True)

model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss',
                                   verbose=1, save_best_only=True)
tboard = TensorBoard(log_dir='model_logs/'+start_time+'_logs',
                     batch_size=BATCH_SIZE, write_graph=False)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=12)# NEW
# train model(encoder only)
model.fit_generator(train_gen, steps_per_epoch=100, epochs=30, #10 (dataset_2019-01-28_03_11_43)
                    validation_data=valid_gen, validation_steps=4,# 4 * 8 bat = 32(30 valid imgs)
                    callbacks=[model_checkpoint,tboard])
set_trainable(model)
# train model(whole network)
model.fit_generator(train_gen, steps_per_epoch=100, epochs=270, #90 (dataset_2019-01-28_03_11_43)
                    validation_data=valid_gen, validation_steps=4,# 4 * 8 bat = 32(30 valid imgs)
                    callbacks=[model_checkpoint,tboard,reduce_lr])

print('Model ' + model_name + ' is trained successfully!')

import evaluator
evaluator.eval_and_save(model_name)
