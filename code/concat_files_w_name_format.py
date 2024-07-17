import numpy as np
import pandas as pd
import natsort
import os, sys, pickle, argparse
import functools
import ipdb

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook'
sys.path.append(PRJCT_PATH)

from module_aladin.file_io import save_pkl
from module_aladin.data_process import load_n_concatt
    
def prjct_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path','-f',default=None)
    args = parser.parse_args()
    
    file_path = args.file_path
    return file_path

def detect_file_cand(file_path):
    dir_path, file_name = os.path.split(file_path)
    all_file = natsort.natsorted(os.listdir(dir_path))
    common_str = file_name.split('_step')[0]
    file_type = file_name.split('_')[-1]
    file_cand = list(filter(lambda x: common_str in x,all_file))
    return dir_path,list(filter(lambda x : x.split('_')[-1]==file_type,file_cand))

def save_file(file,save_dir,file_name):
    if type(file) == pd.DataFrame :
        print(file.info())
        file.to_csv(os.path.join(save_dir,file_name+'.csv'),index=False)
    else :
        print(type(file),len(file))
        save_pkl(save_dir,file_name+'.pkl',file)        

if __name__=='__main__':
    file_path = prjct_config()
    dir_path,cand_list = detect_file_cand(file_path)
    concatted_file = load_n_concatt(dir_path,cand_list)
    common_str = cand_list[0].split('_step')[0]
    save_file(concatted_file,os.path.join(dir_path,'concatted'),common_str+'_concatted') 
    