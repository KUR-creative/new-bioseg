from utils import human_sorted,file_paths,filename_ext, categorize
import cv2, yaml

def bm_split_dataset(origin_map, img_paths,label_paths):
    #NOTE: DO NOT use 'train' or 'test' in name of directory.
    return {
        'origin_map':origin_map,

        'train_imgs':[p for p in img_paths if 'train' in p],
        'valid_imgs':[p for p in img_paths if ('testA' in p or 'testB' in p)],
        'test_imgs': [p for p in img_paths if 'testB' in p],

        'train_masks':[p for p in label_paths if 'train' in p],
        'valid_masks':[p for p in label_paths if ('testA' in p or 'testB' in p)],
        'test_masks': [p for p in label_paths if 'testB' in p],
    }
dset_name = 'b'
image_dirpath = './2step_datas/images/b/'
label_dirpath = './2step_datas/labels/b/'
image_paths = human_sorted(file_paths(image_dirpath))
label_paths = human_sorted(file_paths(label_dirpath))
label = cv2.imread(label_paths[0])
_,origin_map = categorize(label)
with open(dset_name+'.yml','w') as dic:
    dic.write(yaml.dump(
        bm_split_dataset(origin_map,image_paths,label_paths)
    ))

dset_name = 'm'
image_dirpath = './2step_datas/images/m/'
label_dirpath = './2step_datas/labels/m/'
image_paths = human_sorted(file_paths(image_dirpath))
label_paths = human_sorted(file_paths(label_dirpath))
label = cv2.imread(label_paths[0])
_,origin_map = categorize(label)
with open(dset_name+'.yml','w') as dic:
    dic.write(yaml.dump(
        bm_split_dataset(origin_map,image_paths,label_paths)
    ))

dset_name = 'bm'
image_dirpath = './2step_datas/images/bm/'
label_dirpath = './2step_datas/labels/bm/'
image_paths = human_sorted(file_paths(image_dirpath))
label_paths = human_sorted(file_paths(label_dirpath))
label = cv2.imread(label_paths[0])
_,origin_map = categorize(label)
with open(dset_name+'.yml','w') as dic:
    dic.write(yaml.dump(
        bm_split_dataset(origin_map,image_paths,label_paths)
    ))
