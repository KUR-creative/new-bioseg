'''
Generate dataset_yml.

Usage:
    python gen_data_yml.py image_dirpath label_dirpath dataset_type

image_dirpath
    ans1.png
    ans2.png
    ...
    ans_end.png

labels_dirpath
    label1_directory
        ans1.png
        ans2.png
        ...
        ans_end.png

    label2_directory
        ans1.png
        ans2.png
        ...
        ans_end.png

    label3_directory
        ans1.png
        ans2.png
        ...
        ans_end.png
ex)
    python gen_data_yml.py ./exact_boundary190223/image/ ./exact_boundary190223/label_dirs/ boundary_bioseg
'''
import os, sys
import cv2
import yaml
from utils import human_sorted,file_paths,filename_ext, categorize

img_dir = './exact_boundary190223/image/'
labels_directory = './exact_boundary190223/label_dirs/'
dataset_type = 'boundary_bioseg'

img_dir = sys.argv[1]
labels_directory = sys.argv[2]
dataset_type = sys.argv[3]

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

def thick_label_name_rule(label_name):
    return label_name[:-5]

for label_dirname in label_dirnames:
    dic = bioseg_dataset(origin_map, img_paths, label_paths_dic[label_dirname])

    def sanity_check(img_paths, label_paths):
        img_names = map(lambda p:filename_ext(p).name, img_paths)
        label_names = map(lambda p:filename_ext(p).name, label_paths)
        for img_name, label_file_name in zip(img_names,label_names):
            label_name = thick_label_name_rule(label_file_name)
            assert img_name == label_name

    # Sanity Check
    print('------------- labels:', label_dirname)
    sanity_check(dic['train_imgs'],dic['train_masks'])
    sanity_check(dic['valid_imgs'],dic['valid_masks'])
    sanity_check(dic['test_imgs'], dic['test_masks'])

    data_yml_name = label_dirname + 'data.yml'
    with open(data_yml_name,'w') as dic:
        dic.write(yaml.dump(
            dataset_dict(origin_map, img_paths, label_paths_dic[label_dirname])
        ))
