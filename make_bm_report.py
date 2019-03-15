import xlsxwriter
from utils import file_paths, human_sorted, filename
import os
import yaml

root = '.'
is_result = lambda p:('[b' in p) or ('[m' in p)
result_dirpaths = human_sorted(filter(is_result, os.listdir(root)))
print(result_dirpaths,sep='\n')
#print(result_dirpaths)

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

name = result_dirpaths[0]
valid_fname = name.replace('[','__').replace(']','__')
expr_sh = workbook.add_worksheet(valid_fname)

config_fname = '[config]' + name + '.yml'
config_fpath = os.path.join(name,config_fname)
with open(config_fpath) as expr_file:
    expr = yaml.load(expr_file)
#print(expr)

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

expr_sh.write_column(0,0, expr_info_keys_col)
expr_sh.write_column(0,1, expr_info_vals_col) # y,x
key_format = workbook.add_format({'bold':True})
expr_sh.set_column(0,0, cell_format=key_format)

score_keys_row = ['train', 'f1 score', 'dice_obj']
score_keys_y = len(expr_info_keys_col) + 1
score_beg_y = score_keys_y + 1
expr_sh.write_row(score_keys_y,0, score_keys_row)
expr_sh.set_row(score_keys_y,0, cell_format=key_format)

result_fname = '[result]' + name + '.yml'
result_fpath = os.path.join(name,result_fname)
with open(result_fpath) as result_file:
    result = yaml.load(result_file)
print(result)
'''
for fname in result_dirpaths:
    valid_fname = fname.replace('[','__').replace(']','__')
    expr_sh = workbook.add_worksheet(valid_fname)
'''
workbook.close()
