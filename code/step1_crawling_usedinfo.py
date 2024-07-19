import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os, natsort,sys
from tqdm import tqdm
import time, random
import argparse

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook'

sys.path.append(PRJCT_PATH)
from module_aladin.config import url_usedinfo, base_table, selector_dict
from module_aladin.util import class_name
from module_aladin.data_process import nested_dict_to_df, check_pvtb_of_list
from module_aladin.file_io import save_pkl

save_dir = 'processed'

def check_time_2_sleep(idx,num,rest_count,rest_total):
    if idx % (num+random.uniform(0,5)) == 1 :
        rest_count,sleeping_time = rest_count+1, (random.uniform(1,3))/10
        print('time to rest **^^** : ',rest_count," | ",sleeping_time)
        time.sleep(sleeping_time)
        rest_total += sleeping_time
    return rest_count,rest_total

def get_usedinfo(book_id):
    url = url_usedinfo.format(book_id) 
    r = requests.get(url)
    if r.status_code != 200: raise Exception('bad request')
    html=r.text
    soup=BeautifulSoup(html, 'lxml') #  BeautifulSoup 클래스의 인스턴스 생성
    table = soup.select_one(base_table)

    used_list = table.find_all('tr')
    data,error_count = dict(), 0
    if len(used_list) <= 1 : return data, error_count
    for i in range(1,len(used_list)):
        content = used_list[i]
        try :
            data[i] = {
               key : func(content.select_one(selector))
               for key,(selector,func) in selector_dict.items()
            }
        except: error_count += 1
    if not data : raise Exception('all product in table raised error')
    return data, error_count
    

def crawl_usedifo(id_list,work_type='step',num0=0, num1=None):
    data_dict,errored_item = dict(), dict()
    null_used, rest_count, rest_total = list(), 0, 0 
    if work_type=='range':
        if num1 is None : num1 = len(id_list)
        work_target = id_list[num0:num1]
    elif work_type=='step':
        if num1 is None : num1 = 1
        work_target = id_list[num0::num1]

    for n,book_id in enumerate(tqdm(work_target)):
        rest_count,rest_total = check_time_2_sleep(n,30,rest_count,rest_total)
        try:
            used_data, error_count = get_usedinfo(book_id)
            if used_data:
                data_dict[book_id] = used_data
                if error_count : raise Exception('some selector raised error')
            else : null_used.append(book_id)
        except Exception as e:
            errored_item[book_id] = f'{class_name(e)}/{e}'
    
    print('work done : {} | {} | {}'.format(*work_info))
    print('rested time : {} times, {}(sec)'.format(rest_count,rest_total))
    
    return data_dict, errored_item, null_used

#def uncover_list_pvtb(df,cols):
#    df[cols] = df[cols].apply(lambda x : list(map(lambda y: y[0],x)))
#    return df

def process_datadict(data_dict,cols):
    #수집한 datadict를 정해진 형식의 df로 바꿈
    data_df = nested_dict_to_df(data_dict)
    pvtb = pd.pivot_table(data=data_df,values=0,index=['level_0','level_1'],columns='level_2',aggfunc=list).reset_index(level=[1])
    if not check_pvtb_of_list(pvtb,cols): return pvtb, False
    pvtb[cols] = pvtb[cols].apply(lambda x : list(map(lambda y: y[0],x)))
    return pvtb, True

def prjct_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_type','-t',default='step')
    parser.add_argument('--work_feat','-n',default=None)
    parser.add_argument('--start_idx','-i',default=0)
    parser.add_argument('--file_path','-f',default=None)
    args = parser.parse_args()
    
    work_type, start_idx, work_feat = args.work_type,int(args.start_idx),int(args.work_feat)
    file_path = args.file_path
    return file_path,work_type,start_idx,work_feat

if __name__ =='__main__':
    file_path,work_type,start_idx,work_feat = prjct_config()
    df_unused = pd.read_csv(file_path)
    file_dir,file_name = os.path.split(file_path)
    
    id_list = list(df_unused.ItemId.values)
    
    #work_type : 'step' 1/n씩 하기 / 'range' 특정 구간에 대해서 하기
    work_info = work_type,start_idx,work_feat
    data_dict, errored_item, null_used = crawl_usedifo(id_list,*work_info)
    
    cols = ['delivery_fee','price','quality','url','store']
    pvtb,flag = process_datadict(data_dict,cols)
    if not flag : print('error occured when making pivot table')
    pvtb = pvtb.rename(columns={"level_1":"used_idx"})

    #save rslts
    work_name = '.'.join(file_name.split('.')[:-1])
    save_dir = os.path.join(PRJCT_PATH,'processed','usedbook_data','intermid',work_name)
    if not os.path.exists(save_dir) : os.mkdir(save_dir)
    #usedproduct
    pvtb_name = 'usedproduct_{}_{}_{}_{}.csv'.format(file_name[:-4],*work_info)
    save_path = os.path.join(save_dir,pvtb_name)
    pvtb.to_csv(save_path,index=True,index_label='ItemId')
    print("file saved : usedproduct_data | {} | {} | {} | {} | len : {}".format(file_name[:-4],*work_info,len(pvtb)))
    #errored IthemId
    errored_name = 'errored_{}_{}_{}_{}.pkl'.format(file_name[:-4],*work_info)
    null_name = 'nullused_{}_{}_{}_{}.pkl'.format(file_name[:-4],*work_info)
    save_pkl(save_dir,errored_name,errored_item)
    print("file saved : errored_item | {} | {} | {} | {} | len : {}".format(file_name[:-4],*work_info,len(errored_item)))
    #usdeinofo down't exist
    save_pkl(save_dir,null_name,null_used)
    print("file saved : null_used | {} | {} | {} | {} | len : {}".format(file_name[:-4],*work_info,len(null_used)))    