import os, sys
import pickle
import pandas as pd

## FUNCTIONS - FILE I/O

def check_and_mkdir(func):
    def wrapper(*args,**kwargs):
        if not os.path.exists(args[0]): os.mkdir(args[0])
        return func(*args,**kwargs)
    return wrapper

@check_and_mkdir
def save_pkl(save_dir,file_name,save_object):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    file_path = os.path.join(save_dir,file_name)
    with open(file_path,'wb') as f:
        pickle.dump(save_object,f)

def load_pkl(file_path):
    with open(file_path,'rb') as f:
        data = pickle.load(f)
    return data
    
def load_files(dir_path,file_list):
    file_type = list(set(map(lambda x : x.split('.')[-1],file_list)))
    if len(file_type) != 1 :
        print(file_type)
        assert len(file_type) == 1
    file_type = file_type[0]
    if file_type == 'pkl':
        return [load_pkl(os.path.join(dir_path,file)) for file in file_list]
    if file_type == 'csv':     
        return [pd.read_csv(os.path.join(dir_path,file)) for file in file_list]