import os,re,yaml
import xlsxwriter
from collections import namedtuple
import numpy as np
from math import isnan

def mean(values):
    if len(values) == 0:
        return None
    else:
        return np.asscalar(np.mean(values))

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

benigns =\
['testA_1', 'testA_10', 'testA_11', 'testA_18', 'testA_19',
'testA_2', 'testA_20', 'testA_21', 'testA_25', 'testA_27',
'testA_28', 'testA_30', 'testA_31', 'testA_33', 'testA_35',
'testA_36', 'testA_37', 'testA_4', 'testA_40', 'testA_43',
'testA_44', 'testA_46', 'testA_49', 'testA_5', 'testA_50',
'testA_52', 'testA_54', 'testA_55', 'testA_58', 'testA_6',
'testA_60', 'testA_7', 'testA_9', 'testB_17', 'testB_4',
'testB_5', 'testB_7', 'train_12', 'train_15', 'train_2',
'train_20', 'train_21', 'train_22', 'train_25', 'train_30',
'train_32', 'train_33', 'train_34', 'train_38', 'train_4',
'train_45', 'train_46', 'train_47', 'train_51', 'train_52',
'train_53', 'train_55', 'train_56', 'train_59', 'train_6',
'train_61', 'train_62', 'train_63', 'train_64', 'train_65',
'train_67', 'train_69', 'train_72', 'train_79', 'train_8',
'train_80', 'train_84', 'train_85', 'train_9',]
malignants =\
['testA_12', 'testA_13', 'testA_14', 'testA_15', 'testA_16',
'testA_17', 'testA_22', 'testA_23', 'testA_24', 'testA_26',
'testA_29', 'testA_3', 'testA_32', 'testA_34', 'testA_38',
'testA_39', 'testA_41', 'testA_42', 'testA_45', 'testA_47',
'testA_48', 'testA_51', 'testA_53', 'testA_56', 'testA_57',
'testA_59', 'testA_8', 'testB_1', 'testB_10', 'testB_11',
'testB_12', 'testB_13', 'testB_14', 'testB_15', 'testB_16',
'testB_18', 'testB_19', 'testB_2', 'testB_20', 'testB_3',
'testB_6', 'testB_8', 'testB_9', 'train_1', 'train_10',
'train_11', 'train_13', 'train_14', 'train_16', 'train_17',
'train_18', 'train_19', 'train_23', 'train_24', 'train_26',
'train_27', 'train_28', 'train_29', 'train_3', 'train_31',
'train_35', 'train_36', 'train_37', 'train_39', 'train_40',
'train_41', 'train_42', 'train_43', 'train_44', 'train_48',
'train_49', 'train_5', 'train_50', 'train_54', 'train_57',
'train_58', 'train_60', 'train_66', 'train_68', 'train_7',
'train_70', 'train_71', 'train_73', 'train_74', 'train_75',
'train_76', 'train_77', 'train_78', 'train_81', 'train_82',
'train_83',]

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
    mean_train_f1 = mean(train_f1s)
    mean_valid_f1 = mean(valid_f1s)
    mean_train_dice_obj = mean(train_dice_objs)
    mean_valid_dice_obj = mean(valid_dice_objs)
    # Write mean values
    expr_sh.write(train_mean_y,0, 'train mean', key_format)
    expr_sh.write(train_mean_y,1, mean_train_f1)
    expr_sh.write(train_mean_y,2, mean_train_dice_obj)
    expr_sh.write(valid_mean_y,0, 'valid mean', key_format)
    expr_sh.write(valid_mean_y,1, mean_valid_f1)
    expr_sh.write(valid_mean_y,2, mean_valid_dice_obj)

    # Make img_name:(f1,dice_obj) dict
    train_dic = {k:tup for k,tup in zip(train_names, zip(train_f1s,train_dice_objs))}
    valid_dic = {k:tup for k,tup in zip(valid_names, zip(valid_f1s,valid_dice_objs))}
    # Calculate benign/malignant mean values
    def values(keys, k_tup_dic, val_idx):
        return list(\
            map(lambda tup:tup[val_idx],
                filter(lambda v:v is not None,
                       (k_tup_dic.get(k) for k in keys)))
        )

    train_benigns_f1s = values(benigns, train_dic, 0)
    valid_benigns_f1s = values(benigns, valid_dic, 0)
    train_malignants_f1s = values(malignants, train_dic, 0)
    valid_malignants_f1s = values(malignants, valid_dic, 0) 
    train_benigns_dice_objs = values(benigns, train_dic, 1)
    valid_benigns_dice_objs = values(benigns, valid_dic, 1)
    train_malignants_dice_objs = values(malignants, train_dic, 1)
    valid_malignants_dice_objs = values(malignants, valid_dic, 1) 
    mean_train_b_f1 = mean(train_benigns_f1s)
    mean_valid_b_f1 = mean(valid_benigns_f1s)
    mean_train_m_f1 = mean(train_malignants_f1s)
    mean_valid_m_f1 = mean(valid_malignants_f1s)
    mean_train_b_dice_obj = mean(train_benigns_dice_objs)
    mean_valid_b_dice_obj = mean(valid_benigns_dice_objs)
    mean_train_m_dice_obj = mean(train_malignants_dice_objs)
    mean_valid_m_dice_obj = mean(valid_malignants_dice_objs)

    # Write valid/train-benign/malignant-f1/dice_obj key cells
    expr_sh.write( 'E9',           'means'); expr_sh.write( 'F9',      'f1 score'); expr_sh.write( 'G9',            'dice_obj')
    expr_sh.write('E10',    'train benign'); expr_sh.write('F10', mean_train_b_f1); expr_sh.write('G10', mean_train_b_dice_obj)
    expr_sh.write('E11',    'valid benign'); expr_sh.write('F11', mean_valid_b_f1); expr_sh.write('G11', mean_valid_b_dice_obj)
    expr_sh.write('E12', 'train malignant'); expr_sh.write('F12', mean_train_m_f1); expr_sh.write('G12', mean_train_m_dice_obj)
    expr_sh.write('E13', 'valid malignant'); expr_sh.write('F13', mean_valid_m_f1); expr_sh.write('G13', mean_valid_m_dice_obj)

    print(train_benigns_f1s)
    print(valid_benigns_f1s)
    print(train_malignants_f1s)
    print(valid_malignants_f1s)

    print(name)
    
workbook.close()
