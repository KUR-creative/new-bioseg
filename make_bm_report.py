import xlsxwriter
from utils import file_paths, human_sorted
import os
import yaml

root = '.'
is_result = lambda p:('[b' in p) or ('[m' in p)
result_dirpaths = human_sorted(filter(is_result, os.listdir(root)))
print(result_dirpaths,sep='\n')
#print(result_dirpaths)

expr_info_keys_col = ['expr-name', 'train-data', '#filters', '#layers']

workbook = xlsxwriter.Workbook('test.xlsx')
summay_sh = workbook.add_worksheet('summary')
'''
fpath = result_dirpaths[0]
valid_fname = fpath.replace('[','__').replace(']','__')
expr_info_vals_col = []
expr_info_vals_col.append(fpath) # expr-name
expr_info_vals_col.append(fpath) # train-data
expr_info_vals_col.append(fpath) # number of filters
expr_info_vals_col.append(fpath) # number of layers
'''
for fname in result_dirpaths:
    valid_fname = fname.replace('[','__').replace(']','__')
    expr_sh = workbook.add_worksheet(valid_fname)
workbook.close()
