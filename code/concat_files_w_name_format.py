import numpy as np
import pandas as pd
import natsort
import os, sys, pickle, argparse
import functools
import ipdb

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook'
sys.path.append(PRJCT_PATH)

from module_aladin.file_io import save_pkl
from module_aladin.data_process import load_n_concat
    
def prjct_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path','-f',default=None)
    parser.add_argument('--concat_mode','-c',default='harsh')
    parser.add_argument('--save_path','-s',default=None)
    args = parser.parse_args()
    
    file_path = args.file_path
    concat_mode = args.concat_mode
    save_path = args.save_path
    return file_path,concat_mode,save_path

def detect_file_cand(file_path, concat_mode='harsh'):
    dir_path, file_name = os.path.split(file_path)
    all_file = natsort.natsorted(os.listdir(dir_path))
    common_str = file_name.split('_step')[0]
    if concat_mode == 'harsh' : abstract_func = lambda x : x.split('_')[-1]
    else :  abstract_func = lambda x : x.split('.')[-1]
    file_type = abstract_func(file_name)
    file_cand = list(filter(lambda x: common_str in x,all_file))
    return dir_path,list(filter(lambda x : abstract_func(x)==file_type,file_cand))

def save_file(file,save_dir,file_name):
    if type(file) == pd.DataFrame :
        print(file.info())
        file.to_csv(os.path.join(save_dir,file_name+'.csv'),index=False)
    else :
        print(type(file),len(file))
        save_pkl(save_dir,file_name+'.pkl',file)        

if __name__=='__main__':
    file_path, concat_mode, save_path = prjct_config()
    dir_path,cand_list = detect_file_cand(file_path,concat_mode)
    concatted_file = load_n_concat(dir_path,cand_list)
    print(f'{len(cand_list)} files concatted')
    common_str = cand_list[0].split('_step')[0]
    if save_path is None :
        if 'intermid' in dir_path : base_path = dir_path.split('intermid')[0]
        else : base_path = os.path.join(dir_path,'../')
        save_path = os.path.join(base_path,'concatted')
    save_file(concatted_file,save_path,common_str+'_concatted') 
    