import numpy as np
import pandas as pd
import functools

from module_aladin.file_io import load_files

def check_pvtb_of_list(df,cols):
    len_pvtb = df[cols].apply(lambda x : list(map(len,x)))
    return np.sum(np.sum(~(len_pvtb == 1))) == 0

def nested_dict_to_df(data:dict,sep='$'):
    df_in = pd.json_normalize(data,sep=sep)
    df_in.columns = df_in.columns.str.split(sep, expand=True)
    df_reform = df_in.loc[0]
    return df_reform.reset_index()
    
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

def load_n_concatt(dir_path,file_list):
    loaded_files = load_files(dir_path,file_list)
    return concat_files(loaded_files)

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