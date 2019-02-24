import os, sys
import cv2
from utils import human_sorted,file_paths,filename_ext, categorize

img_dir = './exact_boundary190223/image/'
labels_directory = './exact_boundary190223/label_dirs/'
dataset_name = 'boundary_bioseg'

img_paths = human_sorted(file_paths(img_dir))
label_dirnames = human_sorted(os.listdir(labels_directory))
label_dirpaths = list(map(lambda name: os.path.join(labels_directory,name), label_dirnames))
label_paths_seq = map(lambda ldir: [filename_ext(ldir).name,file_paths(ldir)], label_dirpaths)
label_paths_dic = {dirname:human_sorted(label_paths) for dirname,label_paths in label_paths_seq} 

if dataset_name == 'boundary_bioseg':
    origin_map = {(0.0, 0.0, 1.0): [1.0, 1.0, 1.0], 
                  (0.0, 1.0, 0.0): [0.0, 1.0, 0.0], 
                  (1.0, 0.0, 0.0): [0.0, 0.0, 0.0]}
else:
    #TODO: incomplete.
    origin_maps = []
    #for label_dirpath in label_dirpaths:
    for label_dirname in label_dirnames:
        for label_path in label_paths_dic[label_dirname]:
            label = cv2.imread(label_path)
            print(categorize(label)[1])

