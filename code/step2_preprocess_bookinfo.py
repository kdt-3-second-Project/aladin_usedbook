import numpy as np
import pandas as pd
import os,re
from tqdm import tqdm
import sys

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
RSLT_DIR = PRJCT_PATH + 'processed/'

sys.path.append(PRJCT_PATH)

from module_aladin.config import col_name_dict, roman_number, special_chr, paren_patterns
from module_aladin.nlp import erase_num_comma, change_num2year, translate_hanja, find_patterns, clear_patterns
from module_aladin.nlp import replace_by_dict, extract_author1, erase_role

def process_bookname(titles):
    titles = titles.apply(erase_num_comma)
    titles = titles.apply(change_num2year)
    titles = titles.apply(lambda x : replace_by_dict(x,roman_number))
    titles = titles.apply(lambda x : replace_by_dict(x,special_chr))
    titles = titles.apply(translate_hanja)

    temp = titles.apply(lambda x : find_patterns(paren_patterns,x))
    temp2 = pd.DataFrame(data = [np.nan]*len(temp),index = temp.index)
    for parens in paren_patterns.keys():
        temp2[f'con_{parens}'] = temp.apply(lambda x: ', '.join(list(x[parens].values())))

    paren_cols = list(map(lambda x : f'con_{x}',paren_patterns.keys()))
    rslt2 = temp2[paren_cols].apply(lambda x : ', '.join(
        (filter(lambda y : y !='',x))),axis=1)
    rslt1 = titles.apply(lambda x : clear_patterns(paren_patterns,x))
    return rslt1, rslt2
    
def process_authors(authors):
    cond_mul = authors.str.split(',').apply(len) > 1
    authors = authors.apply(extract_author1)
    authors_mul = cond_mul

    authors = authors.apply(erase_role)
    authors = authors.apply(lambda x : re.sub(r'\s\d+[인명]$','',x))

    authors_splitted = authors.str.split(' ')
    end_word = authors_splitted.apply(lambda x : x[-1])
    cond = end_word == '외'
    authors[cond] = authors_splitted.apply(lambda x : ' '.join(x[:-1]))
    authors_mul[cond] = True
    
    return authors, authors_mul    

if __name__ == '__main__':
    date = 240711
    file_name = f'unused_filtered_{date}.csv'

    bookdata_path = os.path.join(RSLT_DIR,file_name)
    bookinfo = pd.read_csv(bookdata_path)
    bookinfo = bookinfo.rename(columns=col_name_dict)

    cols = ['Rank','BName','ItemId','Author',
           'Publshr','Pdate','RglPrice','SlsPrice','SalesPoint',
           'Category','Sorce'] 
    rslt = bookinfo.copy()[cols]

    #도서명
    rslt['BName'],rslt['BName_sub'] = process_bookname(bookinfo['BName'])

    #저자명
    rslt['Author'],rslt['Author_mul'] = process_authors(bookinfo['Author'])

    #순서 정리
    new_cols = cols.copy()
    new_cols.insert(4,'Author_mul')
    new_cols.insert(2,'BName_sub')
    rslt = rslt[new_cols]

    file_name = 'bookinfo_ver{}.csv'.format(0.8)
    save_path = os.path.join(RSLT_DIR,file_name)
    rslt.to_csv(save_path,index=False)

#encoding으로 옮겨야
##출판사
#k = 50
#stats = bookinfo['Publshr'].value_counts().sort_values(ascending=False)
#top_k_val = stats.iloc[k]
#publs_top_k = list(stats[stats >= top_k_val].index)
#cond_etc = ~(bookinfo['Publshr'].isin(publs_top_k))
#bookinfo_processed.loc[cond_etc,'Publshr'] =  '기타 출판사'
##top k 정하는 것은 완전 합본 후에 정해야 함