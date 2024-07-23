
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import tqdm, glob

# 그래프의 폰트 출력을 선명하게 (svg, retina 등이 있음)
matplotlib_inline.backend_inline.set_matplotlib_formats("png2x")
# 테마 설정: "default", "classic", "dark_background", "fivethirtyeight", "seaborn"
mpl.style.use("default")
# 이미지가 레이아웃 안으로 들어오도록 함
# https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html
mpl.rcParams.update({"figure.constrained_layout.use": True})

#font, line, marker 등의 배율 설정: paper, notebook, talk, poster
sns.set_context("paper") 
#배색 설정: tab10, Set2, Accent, husl
sns.set_palette("Set2") 
#눈금, 배경, 격자 설정: ticks, white, whitegrid, dark, darkgrid
# withegrid: 눈금을 그리고, 각 축의 눈금을 제거
sns.set_style("whitegrid") 

import requests
import time
year = range(2000,2024)
month = range(1,13)
week = range(1,6)
# Define the URL to download the CSV file
url_format = 'http://www.aladin.co.kr/shop/common/wbest_excel.aspx?BestType=Bestseller&BranchType=1&CID=0&Year={}&Month={}&Week={}'
for y in year:
     for m in month:
          for w in week:
               print(url_format.format(y,m,w))
               url = url_format.format(y,m,w)
               response = requests.get(url)
               if response.status_code == 200:
                    file_path = '../csv/{}년{}월{}주_20240710.csv'                            
                    with open(file_path.format(y,m,w), 'wb') as file:
                        file.write(response.content)
                    print(f'File downloaded successfully and saved as {file_path}')
               else:
                    print(f'Failed to download the file. Status code: {response.status_code}')
               
               time.sleep(5)

import glob

# CSV 파일이 저장된 디렉토리 경로
file_path = "../csv"
files = glob.glob(file_path + "**/*.csv")

# 데이터프레임을 저장할 리스트
dataframes = []

# 각 파일을 반복 처리
for file in files:
    # 파일 이름에서 Year, Month, Week 추출
    filename = file.split('_')[-2].split('\\')[-1]   
    # CSV 파일을 데이터프레임으로 읽기, 마지막 3줄은 광고문구이므로 제외
    df = pd.read_csv(file,skipfooter=3, engine='python',on_bad_lines='skip')
    # 데이터프레임에 날짜 열 추가
    df['날짜'] = filename
    # ItemId 컬럼을 int 타입으로 변환
    df['ItemId'] = df['ItemId'].astype(int)
    # 데이터프레임을 리스트에 추가
    dataframes.append(df)

# 모든 데이터프레임을 결합
combined_df = pd.concat(dataframes, ignore_index=True)

# 결합된 데이터프레임을 새로운 CSV 파일로 저장
combined_df.to_csv(file_path + "combined_bestseller_data.csv", index=False)

combined_df = pd.read_csv('../csvcombined_bestseller_data.csv')
#다시 다운받지 않고 전에 병합한 csv 파일 불러옴

# ItemId가 중복된 행은 한개 행만 남기고 제거
bestseller_df_unique = combined_df.drop_duplicates(subset=['ItemId'], keep='first')
# 국대도서만 선택
bestseller_df_unique = bestseller_df_unique[bestseller_df_unique['구분']== '국내도서']
# 필요하지 않은 열 삭제
bestseller_df_unique.drop(columns=['ISBN13','부가기호','구분','순번/순위','마일리지'],inplace=True)

# 결측치 확인
null_info =bestseller_df_unique.isnull().sum()
rows_with_missing_values = bestseller_df_unique[bestseller_df_unique.isnull().any(axis=1)]
print('결측치 정보',null_info)
print('결측치 해당하는 행의 갯수:',rows_with_missing_values)

bestseller_df_unique_cleaned = bestseller_df_unique.dropna()
bestseller_df_unique_cleaned.to_csv('../bestseller_cleaned.csv',index=False)