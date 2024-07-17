import numpy as np
import pandas as pd
import copy, os, sys
from tqdm import tqdm
PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
RSLT_DIR = PRJCT_PATH + 'processed/'

bookinfo_name = 'bookinfo_ver{}.pkl'.format(0.75)
usedinfo_name = 'encoded_usedinfo_ver{}.csv'.format(0)
bookinfo_path = os.path.join(RSLT_DIR,bookinfo_name)
usedinfo_path = os.path.join(RSLT_DIR,usedinfo_name)

sys.path.append(PRJCT_PATH)
from module_aladin.file_io import load_pkl, save_pkl
from module_aladin.data_process import pd_datetime_2_datenum

from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.sequence import pad_sequences
import itertools
from sklearn.preprocessing import MinMaxScaler

def set_maxlen(data,maxlen,mode):
    #maxlen 이름 정리 필요
    if maxlen == None : maxlen = len(data)
    if maxlen > len(data) : maxlen = len(data)
    if mode == 'uniform':
        cond = data['counts']>=data['counts'].iloc[maxlen]
        maxlen = np.sum(cond)
    elif mode =='ths':
        cond = data[data['counts'] > maxlen]
        maxlen = np.sum(cond)
    return maxlen

def make_encoding_by_freq(corpus,null_val='[PAD]',maxlen=None,mode=None):
    df_corpus = pd.DataFrame(corpus).T
    df_corpus = df_corpus.rename(columns={0:'token',1:'counts'})
    temp = df_corpus.sort_values(by='counts',ascending=False)
    maxlen = set_maxlen(temp,maxlen,mode)
    temp = temp.iloc[:maxlen]
    temp['val'] = np.arange(maxlen)+1
    temp2 = temp.set_index('token').to_dict()
    map_token_encode = temp2['val']
    map_token_encode[null_val]=0
    return map_token_encode

def encode_tokens(map_token,x,oov=True):
    oov_val =max(map_token.values())+1 if oov else 0
    return map_token[x] if x in map_token else oov_val

if __name__=='__main__':
    file_name = 'data_splitted_ver{}.pkl'.format(0.8)
    file_path = os.path.join(RSLT_DIR,file_name)
    data = load_pkl(file_path)
    
    file_name = 'bookinfo_ver{}.csv'.format(0.8)
    file_path = os.path.join(RSLT_DIR,file_name)
    bookinfo = pd.read_csv(file_path)
    
    book_dict=dict()
    for mode,sample in data.items():
        item_list = list(sample['X']['ItemId'].values)
        book_dict[mode]= bookinfo[bookinfo['ItemId'].isin(item_list)]
    
    mecab = Mecab()
    tokenizer_basic = lambda x : mecab.morphs(x)
    #apply tokenizer
    cols_tknz = ['Category','BName','BName_sub']
    cols_freq = ['Author','Publshr','store']
    for mode, bookinfo in book_dict.items():
        for col in cols_tknz:
            bookinfo[col] = bookinfo[col].fillna('').apply(tokenizer_basic)
        book_dict[mode] = bookinfo    
    
    #make encoding map
    book_tknzed = book_dict['train'][cols_tknz].to_dict('series')
    book_name, book_subname, category = book_tknzed['BName'], book_tknzed['BName_sub'],book_tknzed['Category']
    tokens = np.array(list(itertools.chain(*book_name.values,*book_subname.values,*category.values)))
    corpus = np.unique(tokens,return_counts=True)
    map_token_encode = make_encoding_by_freq(corpus,maxlen=32000)
    encode_1line =lambda x: list(map(lambda y : encode_tokens(map_token_encode,y),x))
    
    ths_author = np.round(len(book_name)/4500)*500
    ths_publshr = np.round(len(book_name)/4500)*30

    pvtb = pd.pivot_table(data=bookinfo,index='Author',values='SalesPoint',aggfunc=np.sum)
    pvtb = pvtb.sort_values(by='SalesPoint',ascending=False)
    author_top_slspnt= pvtb[pvtb['SalesPoint']>=ths_author].index
    encode_author = pd.DataFrame({'author' : author_top_slspnt.values,'val':np.arange(1,len(author_top_slspnt)+1)})
    encode_author = encode_author.set_index('author')
    map_author_encode=encode_author.to_dict()['val']
    
    encode_maps = {
        'Author' : lambda x : encode_tokens(map_author_encode,x,oov=False),
        'Publshr' : lambda x : encode_tokens(map_publshr_encode,x,oov=False),
        'store' : lambda x : encode_tokens(map_store_encode,x,oov=False)
    }

    maxlens={
        'Category' : 5,
        'BName' : 30,
        'BName_sub' : 25
    }
    x_cols = ['ItemId', 'quality', 'store', 'BName', 'BName_sub', 'Author',
       'Author_mul', 'Publshr', 'Pdate', 'RglPrice', 'SalesPoint',
       'Category']
    
    book_cols = ['BName', 'BName_sub', 'Author', 'Author_mul', 'Publshr', 'Pdate',
           'RglPrice', 'SlsPrice', 'SalesPoint', 'Category']
    xcols_scalar = list(filter(lambda x : x not in cols_tknz+['ItemId'],x_cols)) 
    
    #attach bookinfo to usedifo
    for mode,sample in data.items():
        X_mode,bookinfo = sample['X'], book_dict[mode]
        for col in book_cols:
            X_mode[col] = X_mode['ItemId'].apply(lambda x: bookinfo.loc[x,col])
        data[mode]['X'] = X_mode
    #pd concat 이나 join을 이용하는 것으로 바꿔야 함
    
    #encode X 
    X_encoded=dict()
    for mode,sample in data.items():
        X_mode = sample['X']
        #padding and encoding
        encoded = pd.DataFrame(X_mode['ItemId']) 
        for col in cols_tknz :
            padded = pad_sequences(X_mode[col],padding='post',
                                   maxlen=maxlens[col],
                                   value='[PAD]',dtype=object)
            encoded[col] = list(np.apply_along_axis(encode_1line,0,padded))
        #concat encoded
        for col in cols_freq:
            encoded[col] = X_mode['Author'].map(encode_maps[col])
        encoded['Pdate']= pd.to_datetime(X_mode['Pdate'],format='%Y-%m-%d')
        encoded['Pdate']= pd_datetime_2_datenum(encoded['Pdate'])
        cols_else = list(filter(lambda x : x not in encoded.columns,x_cols))
        encoded[cols_else] = X_mode[cols_else]

        concat_tknzed =np.apply_along_axis(np.hstack,1,encoded[cols_tknz].values)
        x_scalar = encoded[xcols_scalar].values
        X = np.hstack((concat_tknzed,x_scalar))
        x_id = encoded['ItemId'].values.reshape(-1,1)
        X = np.hstack((x_id,X))
        X_encoded[mode] = X    

    #scaling
    len_tknzed, len_scalar = sum(maxlens.values()), len(xcols_scalar)
    scale_partition = [(1,len_tknzed+1)]+[(len_tknzed+1+i,len_tknzed+2+i) for i in range(len_scalar)]
    scaler_list = [copy.deepcopy(MinMaxScaler()) for _ in scale_partition]
    scalers = list(zip(scale_partition,scaler_list))
    
    X_scaled = dict()
    for mode in ['train','val','test']:
        intermid = []
        for (cols,scaler) in scalers:
            sliced = X_encoded[mode][:,cols[0]:cols[1]]
            sliced = sliced.astype(np.float64)
            if mode == 'train': sliced = scaler.fit_transform(sliced)
            else : sliced = scaler.transform(sliced)
            intermid.append(sliced)

        X_scaled[mode] = np.hstack(intermid)
    
    data_type = 'sample'
    ver=0.8
    dir_path = os.path.join(RSLT_DIR,'model_input')
    for mode, x_scaled in X_scaled.items():
        save_pkl(dir_path,'{}.v{}_X_{}.pkl'.format(data_type,ver,mode),x_scaled)
    for mode,sample in data.items(): 
        save_pkl(dir_path,'{}.v{}_y_{}.pkl'.format(data_type,ver,mode),sample['y'])