import numpy as np
import pandas as pd
import seaborn as sns
import os, sys

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
RSLT_DIR = PRJCT_PATH + 'processed/'

sys.path.append(PRJCT_PATH)
from module_aladin.util import train_test_split_strat
from module_aladin.file_io import save_pkl

quality_dict= {
    '최상':4,
    '상':3,
    '중':2,
    '균일가':1,
    '하':1
}

def pd_datetime_2_datenum(times):
    times = times.to_numpy(dtype=np.int64)
    return times/1e11 

if __name__=='__main__':
    save_dir = 'processed/usedbook_data/concatted'
    date = 240711
    file_name = f'usedproduct_unused_filtered_{date}_concatted.csv'

    file_path = os.path.join(PRJCT_PATH,save_dir,file_name)
    usedinfo = pd.read_csv(file_path)
    usedinfo['quality'] = usedinfo['quality'].map(quality_dict)
    assert len(usedinfo['delivery_fee'].unique()) == 1
    usedbook_data = usedinfo [['ItemId','quality','store','price']]

    file_name = 'usedinfo_ver{}.csv'.format(0)
    save_path = os.path.join(PRJCT_PATH,'processed',file_name)
    usedinfo.to_csv(save_path,index=False)
    
    bookinfo_name = 'bookinfo_ver{}.csv'.format(0.8)
    bookinfo_path = os.path.join(RSLT_DIR,bookinfo_name)
    bookinfo = pd.read_csv(bookinfo_path)
    
    book_cols = ['BName', 'BName_sub', 'Author', 'Author_mul', 'Publshr', 'Pdate',
           'RglPrice', 'SlsPrice', 'SalesPoint', 'Category']
    for col in book_cols:
        usedinfo[col] = usedinfo['ItemId'].apply(lambda x: bookinfo.loc[x,col])
    #pd concat 이나 join을 이용하는 것으로 바꿔야 함
    
    usedinfo['Pdate']= pd.to_datetime(usedinfo.Pdate,format='%Y-%m-%d')
    usedinfo['Pdate'] = pd_datetime_2_datenum(usedinfo['Pdate']) 
    
    y_col = 'price'
    x_cols = list(filter(lambda x : x != y_col,list(usedinfo.columns)))
    X, y = usedinfo[x_cols], usedinfo[y_col]
    
    #split strat by ItemId
    X_dev,X_test,y_dev,y_test = train_test_split_strat(X,y,strat=X['ItemId'],test_size=0.2,random_state=329)
    X_train,X_val,y_train,y_val = train_test_split_strat(X_dev,y_dev,strat=X_dev['ItemId'],test_size=0.2,random_state=329)
    
    data_dict= {
        'train':{'X': X_train, 'y': y_train},
        'val':{'X': X_val, 'y': y_val},
        'test':{'X': X_test, 'y': y_test},
    }
    
    file_name = 'data_splitted_ver{}.pkl'.format(0.8)
    save_pkl(RSLT_DIR,file_name,data_dict)
    
    
    
    
    
    
    
# encoding 파트로 옮기기    
#    stores= usedbook_data['store'].value_counts().sort_values(ascending=False)
#    store_encoding = {
#        place : n+1
#        for n,place in enumerate(stores.index)
#    }
#    rslt['store'] = usedbook_data['store'].map(store_encoding)
#