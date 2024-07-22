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

y_type='_discount'
X_train=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_X_train.pkl'.format(data_type,ver,strat)))
X_val=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_X_val.pkl'.format(data_type,ver,strat)))
X_test=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_X_test.pkl'.format(data_type,ver,strat)))
y_train=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_y{}_train.pkl'.format(data_type,ver,strat,y_type)))
y_val=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_y{}_val.pkl'.format(data_type,ver,strat,y_type)))
y_test=load_pkl(os.path.join(dir_path,'{}.v{}_st-{}_y{}_test.pkl'.format(data_type,ver,strat,y_type)))

sampling_rate = 0.1
sample_X_train=X_train[::int(1/sampling_rate),:]
sample_X_val=X_val[::int(1/sampling_rate),:]
sample_X_test=X_test[::int(1/sampling_rate),:]
sample_y_train=y_train[::int(1/sampling_rate)]
sample_y_val=y_val[::int(1/sampling_rate)]
sample_y_test=y_test[::int(1/sampling_rate)]

ver = 0
save_pkl(dir_path,'sample{}X_{}_sr{}.pkl'.format(0,'train',sampling_rate),sample_X_train)
save_pkl(dir_path,'sample{}X_{}_sr{}.pkl'.format(0,'val',sampling_rate),sample_X_val)
save_pkl(dir_path,'sample{}X_{}_sr{}.pkl'.format(0,'test',sampling_rate),sample_X_test)
save_pkl(dir_path,'sample{}y_{}_sr{}.pkl'.format(0,'train',sampling_rate),sample_y_train)
save_pkl(dir_path,'sample{}y_{}_sr{}.pkl'.format(0,'val',sampling_rate),sample_y_val)
save_pkl(dir_path,'sample{}y_{}_sr{}.pkl'.format(0,'test',sampling_rate),sample_y_test)