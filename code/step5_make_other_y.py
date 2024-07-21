import numpy as np
import pandas as pd
import os, natsort, re, sys
from tqdm import tqdm

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
save_dir = 'processed/model_input'
dir_path = os.path.join(PRJCT_PATH,save_dir)

sys.path.append(PRJCT_PATH)

from module_aladin.file_io import *

data_type = 'whole'
ver=1.0
strat= False
X_train=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_X_train.pkl'.format(data_type,ver,strat)))
X_val=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_X_val.pkl'.format(data_type,ver,strat)))
X_test=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_X_test.pkl'.format(data_type,ver,strat)))
y_train=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_y_train.pkl'.format(data_type,ver,strat)))
y_val=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_y_val.pkl'.format(data_type,ver,strat)))
y_test=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_y_test.pkl'.format(data_type,ver,strat)))

RSLT_DIR = PRJCT_PATH + 'processed/'
ver, strat = 1.0, False
file_name = 'data_splitted_ver{}_strat-{}.pkl'.format(ver,strat)
file_path = os.path.join(RSLT_DIR,file_name)
data = load_pkl(file_path)

file_name = 'bookinfo_ver{}.csv'.format(1.0)
file_path = os.path.join(RSLT_DIR,file_name)
bookinfo = pd.read_csv(file_path)

book_dict=dict()
for mode,sample in data.items():
    item_list = list(sample['X']['ItemId'].values)
    book_dict[mode]= bookinfo[bookinfo['ItemId'].isin(item_list)]

book_cols = ['BName', 'BName_sub', 'Author', 'Author_mul', 'Publshr', 'Pdate',
           'RglPrice', 'SlsPrice', 'SalesPoint', 'Category']
for mode,sample in tqdm(data.items()):
    X_mode,bookinfo = sample['X'], book_dict[mode].set_index('ItemId')
    for col in tqdm(book_cols):
        X_mode[col] = X_mode['ItemId'].apply(lambda x: bookinfo.loc[x,col])
    data[mode]['X'] = X_mode

for mode,sample in data.items():
  X_mode,y = sample['X'], sample['y']
  X_mode['SellPrice'] = y
#  X_mode['RglPrice'] = X_mode['RglPrice'].apply(erase_num_comma).astype(int)
  X_mode['DicntRate'] = X_mode[['RglPrice','SellPrice']].apply(lambda x : (x[0]-x[1])/x[0]*100, axis=1)
  data[mode]['X'] = X_mode

y_dscnt = dict()
for mode,sample in data.items():
  y_dscnt[mode] = sample['X']['DicntRate'].values
  
for mode,y in y_dscnt.items():
    file_name = '{}.v{}_st-{}_y_discount_{}.pkl'.format(data_type,ver,strat,mode)
    save_pkl(dir_path,file_name,y)