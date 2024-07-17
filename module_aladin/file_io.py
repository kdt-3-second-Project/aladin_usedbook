import os, sys
import pickle

## FUNCTIONS - FILE I/O

def chcek_and_mkdir(func):
    def wrapper(*args,**kwargs):
        if not os.path.exists(args[0]): os.mkdir(args[0])
        return func(*args,**kwargs)
    return wrapper

def save_pkl(save_dir,file_name,save_object):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    file_path = os.path.join(save_dir,file_name)
    with open(file_path,'wb') as f:
        pickle.dump(save_object,f)
        
        
    