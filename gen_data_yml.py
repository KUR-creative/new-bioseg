import os, sys
import cv2
import yaml
from utils import human_sorted,file_paths,filename_ext, categorize

img_dir = './exact_boundary190223/image/'
labels_directory = './exact_boundary190223/label_dirs/'
dataset_type = 'boundary_bioseg'

img_paths = human_sorted(file_paths(img_dir))
label_dirnames = human_sorted(os.listdir(labels_directory))
label_dirpaths = list(map(lambda name: os.path.join(labels_directory,name), label_dirnames))
label_paths_seq = map(lambda ldir: [filename_ext(ldir).name,file_paths(ldir)], label_dirpaths)
label_paths_dic = {dirname:human_sorted(label_paths) for dirname,label_paths in label_paths_seq} 

def bioseg_dataset(origin_map, img_paths,label_paths):
    #NOTE: DO NOT use 'train' or 'test' in name of directory.
    return {
            'origin_map':origin_map,

        'train_imgs':[p for p in img_paths if 'train' in p],
        'valid_imgs':[p for p in img_paths if 'testA' in p],
         'test_imgs':[p for p in img_paths if 'testB' in p],

        'train_masks':[p for p in label_paths if 'train' in p],
        'valid_masks':[p for p in label_paths if 'testA' in p],
         'test_masks':[p for p in label_paths if 'testB' in p]
    }

if dataset_type == 'boundary_bioseg':
    origin_map = {(0.0, 0.0, 1.0): [1.0, 1.0, 1.0], 
                  (0.0, 1.0, 0.0): [0.0, 1.0, 0.0], 
                  (1.0, 0.0, 0.0): [0.0, 0.0, 0.0]}
    dataset_dict = bioseg_dataset
else:
    #TODO: incomplete.
    origin_maps = []
    #for label_dirpath in label_dirpaths:
    for label_dirname in label_dirnames:
        for label_path in label_paths_dic[label_dirname]:
            label = cv2.imread(label_path)
            print(categorize(label)[1])


for label_dirname in label_dirnames:
    print( bioseg_dataset(origin_map, img_paths, label_paths_dic[label_dirname]) )
with open('test.yml','w') as dic:
    dic.write(yaml.dump(
        dataset_dict(origin_map, img_paths, label_paths_dic[label_dirname])
    ))
