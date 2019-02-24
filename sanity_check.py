import os, sys
import cv2
from utils import human_sorted,file_paths,filename_ext

#img_dir = './exact_boundary190223/image/'
#labels_directory = './exact_boundary190223/label_dirs/'

img_dir = sys.argv[1]
labels_directory = sys.argv[2]

img_paths = human_sorted(file_paths(img_dir))
print(img_paths)
label_dirnames = human_sorted(os.listdir(labels_directory))
print(label_dirnames)
label_dirpaths = map(lambda name: os.path.join(labels_directory,name), label_dirnames)
label_paths_seq = map(lambda ldir: [filename_ext(ldir).name,file_paths(ldir)], label_dirpaths)
label_paths_dic = {dirname:human_sorted(label_paths) for dirname,label_paths in label_paths_seq} 
#print(label_paths_dic)


print('-----------------------------')
print('images: %d of input images' % len(img_paths))
print('-----------------------------')
for dirname in label_dirnames:
    num_labels = len(label_paths_dic[dirname])
    print('labels: [%s] has %d of labels' % (dirname, num_labels),
          end = '\n' if len(img_paths) == num_labels 
                     else ' <------------- ERROR! \n')

def thick_label_name_rule(label_name):
    return label_name[:-5]

print('-----------------------------')
for dirname in label_dirnames:
    print('---- labels', dirname, '----')
    img_names = map(lambda p:filename_ext(p).name, img_paths)
    label_names = map(lambda p:filename_ext(p).name, label_paths_dic[dirname])
    for img_name, label_file_name in zip(img_names,label_names):
        label_name = thick_label_name_rule(label_file_name)
        if img_name != label_name:
            print(img_name, '!=', label_name)

    '''
    img_name_set = set(img_names)
    label_name_set = set(map(lambda name:thick_label_name_rule(name), label_names))

    print(img_name_set - label_name_set)
    '''


def look_n_feel():
    print('-------------- look & feel check ----------------')
    error_paths = []
    for dirname in label_dirnames:
        for img_path, label_path in zip(img_paths,label_paths_dic[dirname]):
            print('now showing image:', img_path)
            print('now showing label:', label_path)
            print('----')
            while True:
                cv2.imshow('img', cv2.imread(img_path))
                cv2.imshow('label', cv2.imread(label_path))
                key = cv2.waitKey(1) & 0xFF
                if key == ord('o'):
                    break
                elif key == ord('x'):
                    error_paths.append([img_path, label_path])
                    break
                elif key == ord('q'):
                    return error_paths

error_pairs = look_n_feel()
print('=================================')
print('==== error image:label pairs ====')
print('=================================')
for img_path, label_path in error_pairs:
    print('now showing image:', img_path)
    print('now showing label:', label_path)
    print('----')
    cv2.imshow('img', cv2.imread(img_path))
    cv2.imshow('label', cv2.imread(label_path))
    cv2.waitKey(0)

import yaml
if len(error_pairs) != 0:
    with open('error_pairs.yml','w') as f:
        f.write(yaml.dump(error_pairs))
    print('error_pairs.yml are saved!')
