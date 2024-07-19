## settings
PRJCT_PATH = '/home/doeun/code/AI/ESTSOFT2024/workspace/2.project_text/aladin_usedbook/'
RSLT_DIR = PRJCT_PATH + 'processed/'

## URL, HTML
url_usedinfo = 'https://www.aladin.co.kr/shop/UsedShop/wuseditemall.aspx?ItemId={}&TabType=3&Fix=1'

base_table ='#Ere_prod_allwrap_box > div.Ere_prod_middlewrap > div.Ere_usedsell_table > table' 
selector_dict = {
   'quality': ('td:nth-child(3) > span > span',lambda x : x.__dict__['contents'][0].strip()),
   'price': ('td:nth-child(4) > div > ul > li.Ere_sub_pink > span',lambda x : x.get_text().strip().replace(',','')),
   'delivery_fee': ('td:nth-child(4) > div > ul > li:nth-child(3)',lambda x : x.get_text().strip().split(' : ')[1][:-1].replace(',','')),
   'url': ('td.sell_tableCF1 > a',lambda x : x['href']),
   'store':('td:nth-child(5) > div > ul > li.Ere_store_name > a',lambda x : x.get_text()),
}

## CONSTANTS 
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
    'source' : 'Source',
    '날짜' : 'Source'
}

#NLP

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

paren_patterns={
    '[': r'\[.+\]',
    '{': r'\{.+\}',
    '(': r'\(.+\)',
}

roles = [
'글', '시', '역',
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
'기획.채록',
'편집·해설',
'사진.캘리그라피'
]

custom_hanja ={
    '洛' : '락'
}
