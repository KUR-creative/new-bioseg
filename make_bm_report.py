import os,re,yaml
import xlsxwriter
from collections import namedtuple
import numpy as np
from math import isnan

def chk_nan2zero(x):
    return 0 if isnan(x) else x

def filename_ext(path):
    name_ext = namedtuple('name_ext','name ext')
    return name_ext( *os.path.splitext(os.path.basename(path)) )

def filename(path):
    return filename_ext(path).name

def extension(path):
    return filename_ext(path).ext

def human_sorted(iterable):
    ''' Sorts the given iterable in the way that is expected. '''
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(iterable, key = alphanum_key)

root = '.'
is_result = lambda p:('[b' in p) or ('[m' in p)
result_dirpaths = human_sorted(filter(is_result, os.listdir(root)))
print(result_dirpaths,sep='\n')

expr_info_keys_col = [
    'expr name', 
    'train data', 
    '#filters',
    '#layers',
    ' ',
    'model',
    'optimizer',
    ' ',
    'img size',
    'batch size',
    '#epochs',
    '#SPE',
]

workbook = xlsxwriter.Workbook('test.xlsx')
#summay_sh = workbook.add_worksheet('summary')
key_format = workbook.add_format({'bold':True})

for name in result_dirpaths:
    valid_fname = name.replace('[','__').replace(']','__')
    expr_sh = workbook.add_worksheet(valid_fname)

    # Open data ymls
    config_fname = '[config]' + name + '.yml'
    config_fpath = os.path.join(name,config_fname)
    result_fname = '[result]' + name + '.yml'
    result_fpath = os.path.join(name,result_fname)
    with open(config_fpath) as expr_file, open(result_fpath) as result_file:
        expr = yaml.load(expr_file)
        result = yaml.load(result_file)

    # Write experiment info
    expr_info_vals_col = []
    expr_info_vals_col.append(name)                 # expr name
    dataset_name = filename(expr['DATASET_YML'])
    if dataset_name == 'b':
        train_data = 'Benign'
    elif dataset_name == 'm':
        train_data = 'Malignant'
    else:
        train_data = 'All'
    expr_info_vals_col.append(train_data)           # train data
    expr_info_vals_col.append(expr['NUM_FILTERS'])  # number of filters
    if expr['NUM_MAXPOOL'] == 4:
        num_layers = 34
    elif expr['NUM_MAXPOOL'] == 5:
        num_layers = 42
    expr_info_vals_col.append(num_layers)           # number of layers
    expr_info_vals_col.append(' ')
    expr_info_vals_col.append(expr['MODEL'])        # model
    expr_info_vals_col.append(expr['OPTIMIZER'])    # optimizer
    expr_info_vals_col.append(' ')
    expr_info_vals_col.append(expr['IMG_SIZE'])     # img size
    expr_info_vals_col.append(expr['BATCH_SIZE'])   # batch size
    expr_info_vals_col.append(expr['NUM_EPOCHS'])   # number of epochs
    expr_info_vals_col.append(expr['STEPS_PER_EPOCH']) # steps/epoch

    expr_sh.write_column(0,0, expr_info_keys_col, key_format)
    expr_sh.write_column(0,1, expr_info_vals_col) # y,x

    # Write f1 scores and dice_objs
    # Write train/valid key row, and mean key cell
    train_keys_row = ['train', 'f1 score', 'dice_obj']
    valid_keys_row = ['valid', 'f1 score', 'dice_obj']
    # Calculate ys 
    train_keys_y = len(expr_info_keys_col) + 1
    train_beg_y = train_keys_y + 1
    train_mean_y = train_beg_y + len(result['train_imgs']) 
    valid_keys_y = train_mean_y + 2 
    valid_beg_y = valid_keys_y + 1
    valid_mean_y = valid_beg_y + len(result['valid_imgs']) 

    expr_sh.write_row(train_keys_y,0, train_keys_row, key_format)
    expr_sh.write_row(valid_keys_y,0, valid_keys_row, key_format)

    # Write data columns
    train_names = [filename(p) for p in result['train_imgs']]
    valid_names = [filename(p) for p in result['valid_imgs']]
    train_f1s = [chk_nan2zero(x) for x in result['train_f1']]
    valid_f1s = [chk_nan2zero(x) for x in result['valid_f1']]
    train_dice_objs = [chk_nan2zero(x) for x in result['train_dice_obj']]
    valid_dice_objs = [chk_nan2zero(x) for x in result['valid_dice_obj']]

    expr_sh.write_column(train_beg_y,0, train_names) 
    expr_sh.write_column(train_beg_y,1, train_f1s) 
    expr_sh.write_column(train_beg_y,2, train_dice_objs) 

    expr_sh.write_column(valid_beg_y,0, valid_names) 
    expr_sh.write_column(valid_beg_y,1, valid_f1s) 
    expr_sh.write_column(valid_beg_y,2, valid_dice_objs) 

    # Calculate mean values
    mean_train_f1 = np.asscalar(np.mean(train_f1s))
    mean_valid_f1 = np.asscalar(np.mean(valid_f1s))
    mean_train_dice_obj = np.asscalar(np.mean(train_dice_objs))
    mean_valid_dice_obj = np.asscalar(np.mean(valid_dice_objs))
    # Write mean values
    expr_sh.write(train_mean_y,0, 'train mean', key_format)
    expr_sh.write(train_mean_y,1, mean_train_f1)
    expr_sh.write(train_mean_y,2, mean_train_dice_obj)
    expr_sh.write(valid_mean_y,0, 'valid mean', key_format)
    expr_sh.write(valid_mean_y,1, mean_valid_f1)
    expr_sh.write(valid_mean_y,2, mean_valid_dice_obj)
    
workbook.close()
