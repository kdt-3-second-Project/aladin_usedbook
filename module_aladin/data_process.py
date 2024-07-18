import numpy as np
import pandas as pd
import functools

from module_aladin.file_io import load_files

def check_pvtb_of_list(df,cols):
    #pvtb로 바꿀 때 각 cell 마다 하나의 데이터만 반영되었는지 체크
    len_pvtb = df[cols].apply(lambda x : list(map(len,x)))
    return np.sum(np.sum(~(len_pvtb == 1))) == 0

def nested_dict_to_df(data:dict,sep='$'):
    # dict 안에 dict 구조가 여러겹으로 되어있는 dict를 df로 변환
    df_in = pd.json_normalize(data,sep=sep)
    df_in.columns = df_in.columns.str.split(sep, expand=True)
    df_reform = df_in.loc[0]
    return df_reform.reset_index()
    
def concat_dict(dict1,dict2):
    # 두 dict를 통합. 
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

def concat_data(data_list):
    data_type = list(set(map(type,data_list))) 
    assert len(data_type) == 1
    data_type = data_type[0]
    if data_type in [type(pd.DataFrame()),type(pd.Series())] :
        rslt = pd.concat(data_list)
    elif data_type == type(np.array([])) :
        func1 = lambda x : x.reshape(len(x),-1)
        func = lambda x : np.vstack(map(func1,x))
        rslt = func(data_list)
    elif data_type == list :
        rslt = functools.reduce(lambda x,y : x+y,data_list)
    elif data_type == dict :
        rslt = functools.reduce(concat_dict,data_list)
    else : raise Exception("unknown type")
    return rslt

def load_n_concat(dir_path,file_list):
    loaded_files = load_files(dir_path,file_list)
    return concat_data(loaded_files)

def pd_datetime_2_datenum(times):
    # datetime 자료형을 숫자로 변환
    times = times.to_numpy(dtype=np.int64)
    return times/1e11 