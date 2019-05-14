import os, sys
import cv2
import yaml
from utils import human_sorted,file_paths,filename_ext, categorize

train_img_dir = './CRAG-border/image/train/'
valid_img_dir = './CRAG-border/image/valid/'

train_mask_dir = './CRAG-border/label/t6/train/'
valid_mask_dir = './CRAG-border/label/t6/valid/'

data_yml_name = 'crag6thick.yml'

dataset_type = 'boundary_bioseg'

train_img_paths = human_sorted(file_paths(train_img_dir))
valid_img_paths = human_sorted(file_paths(valid_img_dir))
train_mask_paths= human_sorted(file_paths(train_mask_dir))
valid_mask_paths= human_sorted(file_paths(valid_mask_dir))

print('train img paths:',  train_img_paths)
print('valid img paths:',  *valid_img_paths, sep='\n')
print('train mask paths:', train_mask_paths)
print('valid mask paths:', valid_mask_paths, sep='\n')
print('#train img:', len(train_img_paths))
print('#valid img:', len(valid_img_paths))
print('#train mask:', len(train_mask_paths))
print('#valid mask:', len(valid_mask_paths))
assert len(train_img_paths) == len(train_mask_paths)
assert len(valid_img_paths) == len(valid_mask_paths)


if dataset_type == 'boundary_bioseg':
    origin_map = {(0.0, 0.0, 1.0): [1.0, 1.0, 1.0], 
                  (0.0, 1.0, 0.0): [0.0, 1.0, 0.0], 
                  (1.0, 0.0, 0.0): [0.0, 0.0, 0.0]}
    dataset_dict = {
        'origin_map':origin_map,

        'train_imgs':train_img_paths,
        'valid_imgs':valid_img_paths,
        'test_imgs': [valid_img_paths[0]],

        'train_masks':train_mask_paths,
        'valid_masks':valid_mask_paths,
        'test_masks': [valid_mask_paths[0]],
    }

    with open(data_yml_name,'w') as dic:
        dic.write(yaml.dump( dataset_dict ))
