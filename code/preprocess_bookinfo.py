import numpy as np
import pandas as pd

import os,re
from tqdm import tqdm

PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
save_dir = 'processed'

col_name_dict= {
    '순번/순위' : 'Rank',
    '구분' : 'Partition',
    '상품명' : 'BName',
    '부가기호' : 'AddCode',
    '저자/아티스트' : 'Author',
    '출판사/제작사' : 'Publshr',
    '출간일' : 'Pdate',
    '정가' : 'RglPrice',
    '판매가' : 'SlsPrice',
    '마일리지' : 'Mileage',
    '세일즈포인트' : 'SalesPoint',
    '대표분류(대분류명)' : 'Category',
    'source' : 'Sorce'
}

roles = [
'글', '시', '역',
#'외',
'제공',
'소설',
'극본',
'강의',
'사진',
'구술',
'정리',
'엮음',
'부록',
'편저',
'감수',
'교열',
'그림',
'역주',
'주해',
'판화',
'지음',
'연출',
'감역',
'서문',
'자문',
'옮김',
'편집',
'평역',
'만화',
'사회',
'추천',
'해제',
'각본',
'저자',
'각색',
'원안',
'역자',
'영역',
'해설',
'구성',
'요리',
'기획',
'원작',
#'머리',
'지도',
'작곡','대담','편곡',
'디자인','스토리','디렉터','풀어씀','총편집','엮고지음',
'옮겨엮음',
'일러스트',
'자원집필',
'책임편집',
'소설구성',
'시나리오',
'글.사진',
'콘티구성',
'편찬책임',
'글.그림',
'글.삽화',
'기획.제작',
'동영상강의',
'엮음.사진',
#'관리위원회',
'기획.채록',
'편집·해설',
'사진.캘리그라피'
]

def erase_role(content:str):
    temp = content.strip().split(' ')
    if temp[-1] not in roles: return content.strip()
    else : return ' '.join(temp[:-1])


def extract_author1(content:str):
    ele0 = content.split(',')[0].strip()
    return erase_role(ele0)

def clear_patterns(patterns:dict,sentence):
    for pat in patterns.values() :
        sentence = re.sub(pat,'',sentence).strip()
        sentence = re.sub(r'\s+',' ',sentence).strip()
    return sentence

patterns=dict()
patterns['['] = r'\[.+\]'
patterns['{'] = r'\{.+\}'
patterns['('] = r'\(.+\)'

def re_iter_to_rslt(re_iter):
    return {
        m.start() : m.group()
        for m in re_iter
    }
    
def find_patterns(patterns,sentence):
    return {
        key : re_iter_to_rslt(re.finditer(pat,sentence.strip()))
        for key,pat in patterns.items()
    }

import hanja

custom_hanja ={
    '洛' : '락'
}

def is_hanja_custom(chr):
    return hanja.is_hanja(chr) or chr in custom_hanja

def split_hanja_custom(text):
    """주어진 문장을 한자로 된 구역과 그 이외의 문자로 된 구역으로 분리"""
    # TODO: Can we make this a bit prettier?
    if len(text) == 0:
        yield text
    else:
        ch = text[0]
        bucket = [ch]
        prev_state = is_hanja_custom(ch)

        for ch in text[1:]:
            state = is_hanja_custom(ch)

            if prev_state != state:
                yield "".join(bucket)
                bucket = [ch]
            else:
                bucket.append(ch)

            prev_state = state

        yield "".join(bucket)

import re

parens = [
    ('',''),
    ('(',')'),
    ('{','}'),
    ('[',']'),
    ('（','）'),
    ('［','］'),
    ('｛','｝'),
    ('<','>'),
    ('〈','〉'),
    ('《','》'),
    ('「','」'),
    ('『','』'),
    ('〔','〕'),
    ('【','】'),
]

erase_space = lambda x: re.sub(r'\s+','',x)

def check_adj(word,target,sign:bool):
    #sign : position of word < position of target
    if len(target) > len(word) : return False
    if sign : sliced = word[len(word)-len(target):]
    else : sliced = word[:len(target)]
    if sliced == target : 
        return True
    return False

def is_hanja_words(segment):
    check = list(map(hanja.is_hanja,list(segment)))
    return all(check)

def translate_hanja(sentence):
    #masking
    temp = erase_space(sentence)
    segs = list(split_hanja_custom(temp))
    ishanja = list(map(is_hanja_words,segs))
    for idx,(chr,flag) in enumerate(zip(segs,ishanja)):
        if not flag : continue
        sound = hanja.translate(chr,'substitution')
        #한자 뒤에 '(독음)'의 형태로 있는지 확인
        if idx < len(segs)-1:
            check_list = list(map(lambda x : x[0]+sound+x[1],parens))
            for ele in check_list:
                flag2 = check_adj(segs[idx+1],ele,idx+1<idx)
                if flag2 : segs[idx+1] = ' '*len(ele) + segs[idx+1][len(ele):]
            segs[idx] = sound
        #독음 뒤에 '(한자)'의 형태로 있는지 확인
        if 1 <= idx: 
            for par in parens:
                if not check_adj(segs[idx-1],sound+par[0],idx-1<idx) : continue
                if len(par[1]) > 1 and len(segs) < idx+1 : continue
                if (len(par[1])==0) or (segs[idx+1][:len(par[1])] == par[1]) :
                    segs[idx] = ' '*len(chr)
                    if par[0] : segs[idx-1] = segs[idx-1][:-len(par[0])]+' '*len(par[0])
                    if par[1] : segs[idx+1] = ' '*len(par[1])+segs[idx+1][len(par[1]):]
    masked = ''.join(segs)
    #apply mask to original sentence
    idx = 0
    rslt = list(sentence)
    for i,chr in enumerate(list(sentence)):
        if re.match(r'\s',chr) : continue #have to chage condition for all type of spaces
        rslt[i] = masked[idx]
        idx += 1
    return re.sub(r'\s+',' ',''.join(rslt)).strip()

roman_number ={
   'ⅰ':'1',
   'ⅱ':'2',
   'ⅲ':'3',
   'ⅳ':'4',
   'ⅴ':'5',
   'ⅵ':'6',
   'ⅶ':'7',
   'ⅷ':'8',
   'ⅸ':'9',
   'ⅹ':'10',
   'Ⅰ':'1',
   'Ⅱ':'2',
   'Ⅲ':'3',
   'Ⅳ':'4',
   'Ⅴ':'5',
   'Ⅵ':'6',
   'Ⅶ':'7',
   'Ⅷ':'8',
   'Ⅸ':'9',
   'Ⅹ':'10',
}

special_chr = {
  '&#xFF3C;' : '\\',
  '／':'/',
}

def erase_num_comma(text):
    pat = r'\d\,\d'
    idx_list = sorted([m.start()+1 for m in re.finditer(pat,text)])
    temp = list(text)
    for i in idx_list[::-1]:
        temp.pop(i)
    return ''.join(temp)

def replace_by_dict(text,chr_dict):
    for key,val in chr_dict.items():
        text = text.replace(key,val)
    return text

def change_num2year(text):
    pat = r'\'\d\d(?!\d)'
    temp = sorted([m.start() for m in re.finditer(pat,text)])
    rslt = list(text)
    for i in temp[::-1]:
        t = text[i+1:i+3]
        digit = int(t)
        if digit > 59 : rslt[i] = '19'
        else : rslt[i] = '20'
    return ''.join(rslt)


date = 240711
file_name = f'unused_filtered_{date}.csv'

bookdata_path = os.path.join(PRJCT_PATH,save_dir,file_name)

bookinfo = pd.read_csv(bookdata_path)
bookinfo = bookinfo.rename(columns=col_name_dict)


bookinfo_processed = bookinfo.copy()
cols = ['Rank','BName','ItemId','Author',
       'Publshr','Pdate','RglPrice','SlsPrice','SalesPoint',
       'Category','Sorce'] 
bookinfo_processed = bookinfo_processed[cols]

#도서명
titles = bookinfo['BName']
titles = titles.apply(erase_num_comma)
titles = titles.apply(change_num2year)
titles = titles.apply(lambda x : replace_by_dict(x,roman_number))
titles = titles.apply(lambda x : replace_by_dict(x,special_chr))
titles = titles.apply(translate_hanja)

temp = titles.apply(lambda x : find_patterns(patterns,x))
temp2 = pd.DataFrame(data = [np.nan]*len(temp),index = temp.index)
for parens in patterns.keys():
    temp2[f'con_{parens}'] = temp.apply(lambda x: ', '.join(list(x[parens].values())))

paren_cols = list(map(lambda x : f'con_{x}',patterns.keys()))
bookinfo_processed['BName_sub'] = temp2[paren_cols].apply(lambda x : ', '.join(
    (filter(lambda y : y !='',x))),axis=1)
bookinfo_processed['BName'] = titles.apply(lambda x : clear_patterns(patterns,x))

#출판사
k = 50
stats = bookinfo['Publshr'].value_counts().sort_values(ascending=False)
top_k_val = stats.iloc[k]
publs_top_k = list(stats[stats >= top_k_val].index)
cond_etc = ~(bookinfo['Publshr'].isin(publs_top_k))
bookinfo_processed.loc[cond_etc,'Publshr'] =  '기타 출판사'
#top k 정하는 것은 완전 합본 후에 정해야 함

#저자명
authors = bookinfo['Author']
cond_mul = authors.str.split(',').apply(len) > 1
bookinfo_processed['Author'] = authors.apply(extract_author1)
bookinfo_processed['Author_mul'] = cond_mul

authors = bookinfo_processed['Author']
authors = authors.apply(erase_role)
temp = authors.apply(lambda x : re.sub(r'\s\d+[인명]$','',x))
bookinfo_processed['Author'] = temp

authors = bookinfo_processed['Author']
temp = authors.str.split(' ').apply(lambda x : x[-1])
cond = temp == '외'
temp = temp[cond]
temp = authors.str.split(' ').apply(lambda x : ' '.join(x[:-1]))
bookinfo_processed.loc[cond,'Author'] = temp
bookinfo_processed.loc[cond,'Author_mul'] = True

#정리
new_cols = cols.copy()
new_cols.insert(4,'Author_mul')
new_cols.insert(2,'BName_sub')
bookinfo_processed = bookinfo_processed[new_cols]

file_name = 'bookinfo_ver{}.csv'.format(0.75)
save_path = os.path.join(PRJCT_PATH,'processed',file_name)
bookinfo_processed.to_csv(save_path,index=False)