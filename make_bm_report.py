import os,re,yaml
import xlsxwriter
from collections import namedtuple
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

key_format = workbook.add_format({'bold':True})
expr_sh.write_column(0,0, expr_info_keys_col, key_format)
expr_sh.write_column(0,1, expr_info_vals_col) # y,x

train_keys_row = ['train', 'f1 score', 'dice_obj']
train_keys_y = len(expr_info_keys_col) + 1
expr_sh.write_row(train_keys_y,0, train_keys_row, key_format)

result_fname = '[result]' + name + '.yml'
result_fpath = os.path.join(name,result_fname)
with open(result_fpath) as result_file:
    result = yaml.load(result_file)
train_names = [filename(p) for p in result['train_imgs']]
valid_names = [filename(p) for p in result['valid_imgs']]

valid_keys_row = ['valid', 'f1 score', 'dice_obj']
train_beg_y = train_keys_y + 1
valid_keys_y = train_beg_y + len(train_names) + 2 
expr_sh.write_row(valid_keys_y,0, valid_keys_row, key_format)
valid_beg_y = valid_keys_y + 1

expr_sh.write_column(train_beg_y, 0, train_names) 
expr_sh.write_column(valid_beg_y, 0, valid_names) 
'''
for fname in result_dirpaths:
    valid_fname = fname.replace('[','__').replace(']','__')
    expr_sh = workbook.add_worksheet(valid_fname)
'''
workbook.close()
