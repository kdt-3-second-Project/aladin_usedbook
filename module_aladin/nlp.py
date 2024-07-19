import re, hanja

from module_aladin.config import roles, parens, custom_hanja

def erase_role(content:str):
    temp = content.strip().split(' ')
    if temp[-1] not in roles: return content.strip()
    else : return ' '.join(temp[:-1])

def replace_by_dict(text,chr_dict):
    for key,val in chr_dict.items():
        text = text.replace(key,val)
    return text

def extract_author1(content:str):
    ele0 = content.split(',')[0].strip()
    return erase_role(ele0)

def clear_patterns(patterns:dict,sentence):
    #pattern에 포함된 pat들을 sentence에서 지움
    for pat in patterns.values() :
        sentence = re.sub(pat,'',sentence).strip()
        sentence = re.sub(r'\s+',' ',sentence).strip()
    return sentence

def re_iter_to_rslt(re_iter):
    return {
        m.start() : m.group()
        for m in re_iter
    }
    
def find_patterns(patterns,sentence):
    #pattern에 포함된 pat 각각에 대하여, sentence에 포함된 부분 모두 추출
    return {
        key : re_iter_to_rslt(re.finditer(pat,sentence.strip()))
        for key,pat in patterns.items()
    }

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

def erase_num_comma(text):
    pat = r'\d\,\d'
    idx_list = sorted([m.start()+1 for m in re.finditer(pat,text)])
    temp = list(text)
    for i in idx_list[::-1]:
        temp.pop(i)
    return ''.join(temp)

erase_space = lambda x: re.sub(r'\s+','',x)

#hanja

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

def is_hanja_words(segment):
    check = list(map(hanja.is_hanja,list(segment)))
    return all(check)

def check_adj(word,target,sign:bool):
    #sign : position of word < position of target
    if len(target) > len(word) : return False
    if sign : sliced = word[len(word)-len(target):]
    else : sliced = word[:len(target)]
    if sliced == target : 
        return True
    return False

def translate_hanja(sentence):
    #한자를 한글로 변환. 독음이 병기된 경우 독음만 남김
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
