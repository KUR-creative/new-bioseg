'''
python result_split.py ./dataset.yml ./dst_directory

Copy masks(postprocessed results) with respect to dataset_yml.
[label] => [train|valid|test]
'''

import os, shutil, sys
import yaml

def move(key, dst_dir):
    for img_path in dic[key]:
        file_name = os.path.basename(img_path)
        dst_img_path = os.path.join(dst_dir, file_name)

        shutil.copyfile(img_path, dst_img_path)


dataset_yml_path = sys.argv[1]#'./boundary190203/thick0data.yml'
dst_directory = sys.argv[2]#'./eval_postprocessed/GT/'

with open(dataset_yml_path, 'r') as f:
    dic = yaml.load(f)

train_dst_dir = os.path.join(dst_directory,'train')
valid_dst_dir = os.path.join(dst_directory,'valid')
test_dst_dir  = os.path.join(dst_directory,'test')

os.makedirs(train_dst_dir,exist_ok=True)
os.makedirs(valid_dst_dir,exist_ok=True)
os.makedirs(test_dst_dir, exist_ok=True)

move('train_masks', train_dst_dir)
move('valid_masks', valid_dst_dir)
move('test_masks',  test_dst_dir)
