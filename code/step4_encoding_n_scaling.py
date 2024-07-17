import numpy as np
import pandas as pd
import os, natsort, re,sys
from tqdm import tqdm
PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
RSLT_DIR = PRJCT_PATH + 'processed/'

bookinfo_name = 'bookinfo_ver{}.pkl'.format(0.75)
usedinfo_name = 'encoded_usedinfo_ver{}.csv'.format(0)
bookinfo_path = os.path.join(RSLT_DIR,bookinfo_name)
usedinfo_path = os.path.join(RSLT_DIR,usedinfo_name)

sys.path.append(PRJCT_PATH)
from module_aladin.file_io import load_pkl, save_pkl

from konlpy.tag import Mecab
from tensorflow.keras.preprocessing.sequence import pad_sequences
import itertools

def load_data_dict(file_path):
    data_splitted = load_pkl(file_path)
    
    X_train, y_train = data_splitted['train'].values()
    X_val, y_val = data_splitted['val'].values()
    X_test, y_test = data_splitted['test'].values()
    return X_train,y_train,X_val,y_val,X_test,y_test
    

def make_encoding_by_freq(corpus,null_val='[PAD]',maxlen=None):
    df_corpus = pd.DataFrame(corpus).T
    df_corpus = df_corpus.rename(columns={0:'token',1:'counts'})
    
    if maxlen == None : maxlen = len(df_corpus)
    if maxlen > len(df_corpus) : maxlen = len(df_corpus)
    
    temp = df_corpus.sort_values(by='counts',ascending=False)
    temp = temp.iloc[:maxlen]
    temp['val'] = np.arange(len(temp))+1
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
    X_train,y_train,X_val,y_val,X_test,y_test = load_data_dict(file_path)
    file_name = 'bookinfo_ver{}.csv'.format(0.8)
    file_path = os.path.join(RSLT_DIR,file_name)
    bookinfo = pd.read_csv(file_path)
    item_list = list(X_train['ItemId'].values)
    book_train = bookinfo[bookinfo['ItemId'].isin(item_list)]
   
   # corpus는 bookinfo를 보고 만들되 replace는 usedinfo에 해야함을 주의해서 아래 부분 다시 수정해야 함
    
    #encoding train
    #using tokenizer : BName, BName_sub, Category
    mecab = Mecab()
    tokenizer_basic = lambda x : mecab.morphs(x)
    
    book_name,book_subname, category = book_train.BName, book_train.BName_sub, book_train.Category
    book_name = book_name.apply(tokenizer_basic)
    book_subname = book_subname.fillna('').apply(tokenizer_basic)
    category = category.apply(tokenizer_basic)
    
    tokens = np.array(list(itertools.chain(*book_name.values,*book_subname.values,*category.values)))
    corpus = np.unique(tokens,return_counts=True)
    map_token_encode = make_encoding_by_freq(corpus,maxlen=32000)
    #padding
    padded_bookname = pad_sequences(book_name,padding='post',maxlen=30,value='[PAD]',dtype=object)
    padded_subname = pad_sequences(book_subname,padding='post',maxlen=25,value='[PAD]',dtype=object)
    padded_category = pad_sequences(category,padding='post',maxlen=5,value='[PAD]',dtype=object)
    #encode
    encode_1line =lambda x: list(map(lambda y : encode_tokens(map_token_encode,y),x))
    encoded_bookname = np.apply_along_axis(encode_1line,0,padded_bookname)
    encoded_subname = np.apply_along_axis(encode_1line,0,padded_subname)
    encoded_category = np.apply_along_axis(encode_1line,0,padded_category)
    
    #only by frequency : Author,Publshr,store
    
    ths_author = np.round(len(book_train)/4500)*500
    
    pvtb = pd.pivot_table(data=book_train,index='Author',values='SalesPoint',aggfunc=np.sum)
    pvtb = pvtb.sort_values(by='SalesPoint',ascending=False)
    author_top_slspnt= pvtb[pvtb['SalesPoint']>=ths_author].index
    encode_author = pd.DataFrame({'author' : author_top_slspnt.values,'val':np.arange(1,len(author_top_slspnt)+1)})
    encode_author = encode_author.set_index('author')
    encode_map_author=encode_author.to_dict()['val']
    encode_map_author['기타 저자'] = 0
    authors = book_train.Author
    cond = authors.isin(author_top_slspnt)
    authors[~cond] = '기타 저자'
    encoded_author = authors.map(encode_map_author)

    publshrs = book_train.Publshr
    map_publs_encode = make_encoding_by_freq(publshrs.value_counts(),maxlen=32000)
    encoded_publshr = publshrs.map(lambda x: encode_tokens(map_publs_encode,x,oov=False)) 
    
    X_train['BName'] = list(encoded_bookname)
    X_train['BName_sub'] = list(encoded_subname)
    X_train['Publshr'] = encoded_publshr
    X_train['Author'] = encoded_author
    X_train['Category'] = list(encoded_category)
    
    cols_arry = ['Category','BName','BName_sub']
    temp = X_train[cols_arry].values
    temp_concat =np.apply_along_axis(np.hstack,1,temp)
    
    x_cols = ['ItemId', 'quality', 'store', 'BName', 'BName_sub', 'Author',
       'Author_mul', 'Publshr', 'Pdate', 'RglPrice', 'SlsPrice', 'SalesPoint',
       'Category']
    xcols_scalar = list(filter(lambda x : x not in cols_arry+['ItemId'],x_cols))
    x_scalar = X_train[xcols_scalar].values
    X = np.hstack((temp_concat,x_scalar))
    x_id = X_train['ItemId'].values.reshape(-1,1)
    X = np.hstack((x_id,X))
    
    
    #encoding val
    
    #encoding test

    from sklearn.preprocessing import MinMaxScaler

    scale_partition = [(1,61)]+[(61+i,62+i) for i in range(9)]
    import copy
    scaler_list = [copy.deepcopy(MinMaxScaler()) for _ in scale_partition]
    scalers = list(zip(scale_partition,scaler_list))

    intermid = []
    for (cols,scaler) in scalers:
        sliced = X_train[:,cols[0]:cols[1]]
        sliced = sliced.astype(np.float64)
        sliced = scaler.fit_transform(sliced)
        intermid.append(sliced)

    X_train_scaled = np.hstack(intermid)

    intermid = []
    for (cols,scaler) in scalers:
        sliced = X_val[:,cols[0]:cols[1]]
        sliced = sliced.astype(np.float64)
        sliced = scaler.transform(sliced)
        intermid.append(sliced)

    X_val_scaled = np.hstack(intermid)

    intermid = []
    for (cols,scaler) in scalers:
        sliced = X_test[:,cols[0]:cols[1]]
        sliced = sliced.astype(np.float64)
        sliced = scaler.transform(sliced)
        intermid.append(sliced)

    X_test_scaled = np.hstack(intermid)

    data_type = 'sample'
    ver=0.8
    dir_path = os.path.join(RSLT_DIR,'model_input')
    save_pkl(dir_path,'{}.v{}_X_train.pkl'.format(data_type,ver),X_train_scaled)
    save_pkl(dir_path,'{}.v{}_X_val.pkl'.format(data_type,ver),X_val_scaled)
    save_pkl(dir_path,'{}.v{}_X_test.pkl'.format(data_type,ver),X_test_scaled)
    save_pkl(dir_path,'{}.v{}_y_train.pkl'.format(data_type,ver),y_train)
    save_pkl(dir_path,'{}.v{}_y_val.pkl'.format(data_type,ver),y_val)
    save_pkl(dir_path,'{}.v{}_y_test.pkl'.format(data_type,ver),y_test)