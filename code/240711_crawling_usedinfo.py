import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import requests
import re
import os, natsort
from tqdm import tqdm
import time, random
import argparse

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook'
save_dir = 'processed'

url_usedinfo = 'https://www.aladin.co.kr/shop/UsedShop/wuseditemall.aspx?ItemId={}&TabType=3&Fix=1'

base_table ='#Ere_prod_allwrap_box > div.Ere_prod_middlewrap > div.Ere_usedsell_table > table' 
selector_dict = {
   'quality': ('td:nth-child(3) > span > span',lambda x : x.__dict__['contents'][0].strip()),
   'price': ('td:nth-child(4) > div > ul > li.Ere_sub_pink > span',lambda x : x.get_text().strip().replace(',','')),
   'delivery_fee': ('td:nth-child(4) > div > ul > li:nth-child(3)',lambda x : x.get_text().strip().split(' : ')[1][:-1].replace(',','')),
   'url': ('td.sell_tableCF1 > a',lambda x : x['href']),
}

def class_name(clss):
  name = str(type(clss)).strip()
  name = name[1:-1].split(' ')
  return name[1]

def crwal_usedifo(id_list,work_type='range',num0=0, num1=None):
    data_dict,errored_item = dict(), dict()
    null_used, rest_count = list(), 0
    if work_type=='range':
        if num1 is None : num1 = len(id_list)
        work_target = id_list[num0:num1]
    elif work_type=='worker':
        if num1 is None : num1 = 1
        work_target = id_list[num0::num1]

    for n,book_id in enumerate(tqdm(work_target)):
        url = url_usedinfo.format(book_id) 
        data = dict()
        if n % (100+random.uniform(0,5)) == 1 :
            rest_count,sleeping_time = rest_count+1, (random.uniform(1,3))/2
            print('time to rest **^^** : ',rest_count," | ",sleeping_time)
            time.sleep(sleeping_time)
        try:
            r = requests.get(url)
            if r.status_code != 200: raise Exception('bad request')
            html=r.text
            soup=BeautifulSoup(html, 'lxml') #  BeautifulSoup 클래스의 인스턴스 생성
            table = soup.select_one(base_table)

            used_list = table.find_all('tr')
            if len(used_list) <= 1 : null_used.append(book_id)
            data,error_count = dict(), 0
            for i in range(1,len(used_list)):
                content = used_list[i]
                try :
                    data[i] = {
                       key : func(content.select_one(selector))
                       for key,(selector,func) in selector_dict.items()
                    }
                except: error_count += 1
            if data : data_dict[book_id] = data
            elif len(used_list) > 1 : raise Exception('all product in table raised error')
            if error_count : raise Exception('some selector raised error')
        except Exception as e:
            errored_item[book_id] = f'{class_name(e)}/{e}'
    
    return data_dict, errored_item, null_used
    
def nested_dict_to_df(data:dict,sep='$'):
    df_in = pd.json_normalize(data,sep=sep)
    df_in.columns = df_in.columns.str.split(sep, expand=True)
    df_reform = df_in.loc[0]
    return df_reform.reset_index()

def check_pvtb_of_list(df,cols):
    len_pvtb = df[cols].apply(lambda x : list(map(len,x)))
    return np.sum(np.sum(~(len_pvtb == 1))) == 0
    
def uncover_list_pvtb(df,cols):
    df[cols] = df[cols].apply(lambda x : list(map(lambda y: y[0],x)))
    return df

def process_datadict(data_dict,cols):
    data_df = nested_dict_to_df(data_dict)
    pvtb = pd.pivot_table(data=data_df,values=0,index=['level_0','level_1'],columns='level_2',aggfunc=list).reset_index(level=[1])
    if not check_pvtb_of_list(pvtb,cols): return pvtb
    pvtb[cols] = pvtb[cols].apply(lambda x : list(map(lambda y: y[0],x)))
    return pvtb

def chcek_and_mkdir(func):
    def wrapper(*args,**kwargs):
        if not os.path.exists(args[0]): os.mkdir(args[0])
        return func(*args,**kwargs)
    return wrapper

import pickle

def save_pkl(save_dir,file_name,save_object):
    if not os.path.exists(save_dir): os.mkdir(save_dir)
    file_path = os.path.join(save_dir,file_name)
    with open(file_path,'wb') as f:
        pickle.dump(save_object,f)

def prjct_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_type',default='worker')
    parser.add_argument('--work_feat',default=None)
    parser.add_argument('--start_idx',default=0)
    parser.add_argument('--file_path',default=None)
    args = parser.parse_args()
    
    work_type, start_idx, work_feat = int(args.work_type),int(args.start_idx),int(args.work_feat)
    file_path = args.file_path
    return file_path,work_type,start_idx,work_feat

if __name__ =='__main__':
    file_path,work_type,start_idx,work_feat = prjct_config()
    df_unused = pd.read_csv(file_path)
    file_dir,file_name = os.path.split(file_path)
    
    id_list = list(df_unused.ItemId.values)
    
    #work_type : 'worker' 1/n씩 하기 / 'range' 특정 구간에 대해서 하기
    work_type = 'worker'
    feat_0, feat_1 = 0, 100
    data_dict, errored_item, null_used = crwal_usedifo(id_list,work_type,feat_0,feat_1)
    
    cols = ['delivery_fee','price','quality','url']
    pvtb = process_datadict(data_dict)
    pvtb = pvtb.rename(columns={"level_1":"used_idx"})
    
    save_dir = os.path.join(PRJCT_PATH,'processed','usedbook_data')
    
    pvtb_name = 'usedproduct_{}_{}_{}_{}.csv'.format(file_name[:-4],'range',0,100)
    save_path = os.path.join(save_dir,pvtb_name)
    pvtb.to_csv(save_path,index=True,index_label='ItemId')
    errored_name = 'errored_{}_{}_{}_{}.pkl'.format(file_name[:-4],'range',0,100)
    null_name = 'nullused_{}_{}_{}_{}.pkl'.format(file_name[:-4],'range',0,100)
    save_pkl(save_dir,errored_name,errored_item)
    save_pkl(save_dir,null_name,null_used)
    
    
    
    
    