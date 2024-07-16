import numpy as np
import pandas as pd
import natsort
import os, sys, pickle, argparse
import functools
import ipdb

def save_pkl(save_dir,file_name,save_object):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    file_path = os.path.join(save_dir,file_name)
    with open(file_path,'wb') as f:
        pickle.dump(save_object,f)

    
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

def load_files(dir_path,file_list):
    file_type = list(set(map(lambda x : x.split('.')[-1],file_list)))
    if len(file_type) != 1 :
        print(file_type)
        assert len(file_type) == 1
    file_type = file_type[0]
    if file_type == 'pkl':
        rslt = list()
        for file in file_list:
            file_path = os.path.join(dir_path,file)
            with open(file_path,'rb') as f :
                rslt.append(pickle.load(f))
        return rslt
    if file_type == 'csv':     
        return [pd.read_csv(os.path.join(dir_path,file)) for file in file_list]

def concat_dict(dict1,dict2):
    key1, key2 = set(dict1.keys()), set(dict2.keys())
    common_keys = key1.intersection(key2)
    key1_only = key1.difference(key2)
    key2_only = key2.difference(key1)
    rslt = dict()
    if len(common_keys)==0:
        for key,val in dict1.items():
            rslt[key] = val
        for key,val in dict2.items():
            rslt[key] = val
    else :
        for key in key1_only:
            rslt[key] = [dict1[key]] 
        for key in key2_only:
            rslt[key] = [dict2[key]]
        for key in common_keys : 
            rslt[key] = [dict1[key],dict2[key]]
    return rslt 

def concat_files(file_list):
    file_type = list(set(map(type,file_list))) 
    assert len(file_type) == 1
    file_type = file_type[0]
    if file_type == pd.DataFrame :
        rslt = pd.concat(file_list)
    elif file_type == list :
        rslt = functools.reduce(lambda x,y : x+y,file_list)
    elif file_type == dict :
        rslt = functools.reduce(concat_dict,file_list)
    return rslt

def load_n_concatt(dir_path,file_list):
    loaded_files = load_files(dir_path,file_list)
    return concat_files(loaded_files)    

def save_file(file,save_dir,file_name):
    if type(file) == pd.DataFrame :
        print(file.info())
        file.to_csv(os.path.join(save_dir,file_name+'.csv'),index=False)
    else :
        print(type(file),len(file))
        save_pkl(save_dir,file_name+'.pkl',file)        

if __name__=='__main__':
    file_path = prjct_config()
#    ipdb.set_trace()
    dir_path,cand_list = detect_file_cand(file_path)
    concatted_file = load_n_concatt(dir_path,cand_list)
    common_str = cand_list[0].split('_step')[0]
    save_file(concatted_file,os.path.join(dir_path,'concatted'),common_str+'_concatted') 
    