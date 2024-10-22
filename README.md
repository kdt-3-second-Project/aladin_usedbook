# 알라딘 중고 도서 데이터셋 구축 및 그에 기반한 중고 서적 가격 예측 모델

프로젝트 구성원: 오도은, 박예림, 이준성, 정홍섭 / [발표 슬라이드](https://docs.google.com/presentation/d/15EIOMGpadZQf3cT2k0pfClS9DVICLmmf5ZTH1k4XnKc/edit?usp=sharing)

**사용된 스킬 셋**: NumPy, Pandas, Matplotlib, Beautifulsoup, re, Scikit-learn, xgboost, [Mecab](https://pypi.org/project/python-mecab-ko/)

## 1. 프로젝트 개요

### 배경

- 중고 도서의 상태, 저자, 장르 등 다양한 요소를 기반으로 중고 도서의 판매 가격이 형성됨
- 중고 도서 가격은 상품의 종류 및 상태 별로 가격의 편차가 있음
- 가격에 중요한 요소들의 내용은 쉽게 확인 가능하고, 상품 페이지에 정리가 잘 되어있는 편
  - 크롤링을 통해 데이터셋을 구축하기에 접근성이 좋음

### 목표

- 크롤링을 통해 알라딘 중고도서 데이터 셋 구축
- 알라딘 중고 매장에서 판매하는 중고 도서 가격을 예측하는 회귀 모델 개발
  - 알라딘 홈페이지에서 판매하는 중고도서 중 중고 매장에서 판매하는 상품을 우선적 대상으로 함
  - Random Forest Regressor, XGBoost 등의 다양한 모델을 이용
- 각 모델의 성능을 여러 성능 지표 및 실험을 통하여 적절히 평가

## 2. 데이터 셋

### 1) 개요

- 알라딘의 중고 도서 상품과 새 책 사이에 url 구조 등으로는 알라딘의 중고 도서와 새 책 사이에 구별할 수 없음
- 알라딘에서 도서별로 중고 상품을 정리해 놓은 페이지를 이용하여 데이터 셋을 체계적으로 구축

### 2) 구성

#### [알라딘 주간 베스트셀러 데이터](https://www.aladin.co.kr/shop/common/wbest.aspx?BranchType=1)

- 알라딘의 주간 베스트셀러 페이지에서 제공한 1~1000위에 대한 xls 파일 데이터를 이용하여 구성
- 2000년 1월 1주차 ~ 2024년 7월 2주차까지의 데이터를 포괄하며, 24-07-00에 수집 진행

![image](https://github.com/user-attachments/assets/e330ca44-893c-4fad-8d91-4a2f520c13af)

*도표. 알라딘 주간 베스트셀러 페이지 예시*

- 총 1,415,586개의 row와 랭킹, 구분, 도서 명, ItemId, ISBN13, 부가기호, 저자, 출판사, 출판일, 정가, 판매가, 마일리지, 세일즈 포인트, 카테고리, 날짜 12개의 column
  - **구분** : 국내도서, 외국도서 등으로 구분되어 있음
  - **ItemId** : 알라딘에서 부여한 해당 도서의 id. 숫자로만 구성
    - 새 책 기준의 id 값이 기재됐고, 한정판, 개정판 등의 경우도 별도의 id가 부여 됨
    - raw data에는 도서 외에도, 당시 베스트셀러였던 MD 굿즈, 강연 등도 포함되어 있음
  - **날짜, 랭킹** : 해당 도서가 어떤 주차의 주간 베스트셀러 목록에 몇 위로 올랐는지
    - 하나의 도서가 다양한 주 차에서 다양한 랭킹의 베스트셀러로 등장
  - [**ISBN13, 부가기호**](https://blog.aladin.co.kr/ybkpsy/959340) : ISBN13은 전세계에서 공통적으로 사용하는 도서에 대한 id. 발행자 등의 정보가 포함되어 있음. 부가기호는 한국 문헌 보호 센터에서 부여하는 번호로, 예상 독자층에 대한 정보 등이 포함 되어 있음
  - **카테고리** : 도서가 어떤 장르에 속하는지에 대한 정보. 외국어, 종교, 사회과학, 건강/취미 등 총 24개 유형으로 분류
  - **세일즈 포인트**
    - 판매량과 판매기간에 근거하여 해당 상품의 판매도를 산출한 알라딘만의 판매지수이며, 매일 업데이트 됨
    - 최근 판매분에 가중치를 두어, 팔릴수록 올라가고 덜 팔리면 내려감
    - 최근 베스트셀러는 점수가 높으며, 꾸준히 팔리는 스테디셀러들도 어느 정도 점수를 유지
- 날짜 및 랭킹을 제외하고, 판매가, 세일즈 포인트 등은 크롤링 시점에서의 값이 저장됨

![image](https://github.com/user-attachments/assets/8d74d9a6-3423-4bd3-b0a0-27817761de9c)

*도표. 알라딘 주간 베스트 셀러*

#### [알라딘 중고 도서 데이터](https://www.aladin.co.kr/shop/UsedShop/wuseditemall.aspx?ItemId=254468327&TabType=3&Fix=1)

![image](https://github.com/user-attachments/assets/e8840608-96f8-47e6-954b-5d6e08f47df9)

*도표. 도서 별 중고 매물 목록 페이지 예시*

<!--위의 탭을 포함하는 이미지로 업데이트 필요-->

- 알라딘 중고 도서는 판매자 별로 셋으로 분류 됨
  1. 알라딘 직접 배송 : 중고매장에서 판매하고 있지 않은 중고 도서
  2. 알라딘 온라인 중고매장 : 오프라인의 알라딘 중고 매장에서 판매되고 있는 중고 도서
  3. 판매자 중고 : 판매자가 알라딘이 아닌 중고 도서
- [알라딘 온라인 중고매장(광활한 우주점)](https://www.aladin.co.kr/usedstore/wonline.aspx?start=we)에 등록 된 중고 도서 매물로 한정
  - 도서멸 중고 매물 목록 페이지 url 구조 상, 새 책의 ItemId를 이용하여 접근할 수 있음
  - 판매자 분류를 기준으로 탭이 나눠져 있으며, 알라딘 중고 매장도 그 중 하나의 탭
- 위의 베스트셀러 데이터에 포함된 도서(ItemId)를 기준으로 크롤링한 중고도서 매물 자료

![image](https://github.com/user-attachments/assets/6bc6657e-cc45-4830-baaa-fca240733d6e)

*도표. 알라딘 중고 도서 데이터*

- 총 784,213개의 row, 7개의 column으로 구성.
  - 각 row 당 중고 도서 매물 하나에 해당
    - 103,055 종의 도서에 대한 중고도서 매물 784,213건
  - ItemId (새 책 기준), 중고 번호, 중고 등급, 판매지점, 배달료, 중고가, 판매 url
  - **ItemId** : ItemId는 중고 도서를 포함하여 모든 상품에 각각 부여되기 때문에, 책 종류를 구별하려면 새 책 기준 ItemId를 사용해야 함
  - **중고 번호** : 해당도서의 중고도서 목록 페이지에 있었던 순서
  - **중고가, 품질**
    - 중고가는 품질(중고 등급)의 큰 영향을 받으며, '균일가' 및 '하', '중', '상', '최상'으로 구분되어 있음
    - 품질이 높을수록 중고가가 높은 경향이 있음
    - 같은 품질이라도 가격이 다르거나, 낮은 품질의 매물보다 더 가격이 싼 경우가 종종 있음
  - **판매 url** : 해당 중고 매물에 대한 판매 페이지. 해당 중고 매물의 ItemId가 url에 포함되어 있음

- 위 2개의 데이터를 종합하여 데이터셋을 만들어 프로젝트를 진행
- 종합된 데이터 셋에서는 ItemId, 도서 명, 중고 등급, 판매 지점, 저자, 출판사, 출판일, 정가, 판매가, 세일즈포인트, 카테고리, 중고가 총 12개를 column으로 사용

![image](https://github.com/user-attachments/assets/caa98ef5-b5be-47d9-a9c4-9ff236ecdb48)

*도표. 데이터 셋들에 포함된 주요 column 및 그에 대한 개요*

## 3. 문제 설정

**목표**: 품질, 판매 지점, 저자, 출판사, 출판일, 정가 등의 값을 이용하여 알라딘에서 공식으로 판매하는 중고 서적 가격을 예측 하고 분석하고자 함

### 1) 종속 변수/ 독립 변수

- 종속 변수를 제외한 항목 중에서 총 11개의 독립변수 선정
  - BName_sub (도서명에서 괄호 안의 내용), Author_mul (저자 등이 여러 명으로 표기되었는지 여부) 등 파생 항목 포함. 해당 내용은 전처리 파트에서 후술

| 종속 변수 | 독립 변수 |
|---------|---------|
| Price | quality, store, BName, BName_sub, Author, Author_mul, Publshr, Pdate, RglPrice, Category, SalesPoint |

*도표. 모델의 종속 변수 및 독립 변수*

### 2) 실험 설계

- sklearn을 이용하여 train 64%, validation 16%, test 20% 비율로 분리
  - train : 95,061종의 도서에 대한 중고도서 501,896건
  - valid : 62,995종의 도서에 대한 중고도서 125,474건
  - test : 69,385종의 도서에 대한 중고도서 156,843건
- 크게 세 가지 측면으로 실험 진행
  - Random Forest Regressor, XGBoost 모델 간의 성능을 비교
    - Grid search를 이용해 각 모델 별로 가장 높은 성능을 내는 hyper parameter 탐색
  - 판매가와 SalesPoint를 학습에서 제외시켜도 안정적인 성능이 나오는지 탐색
  - train set에 포함되지 않았던 도서들에 대한 중고 매물로 test 대상을 한정지었을 때, 성능이 어떻게 달라지는지 탐색
- RMSE, MAPE, R2 Score 등의 회귀 평가 지표를 사용하여 성능을 각 모델 별로 분석

## 4. [전처리](./code/)

### 1) 전체 과정

#### 베스트 셀러 목록 전처리

- 결측치 처리
  - 저자 명, 구분, 출판사, 카테고리 등에 결측치가 있는 행의 개수 1,214개
    - 실제 도서도 있지만, MD 굿즈, 강연등 도서가 아닌 데이터 다수 존재
- 중복 도서 처리 : 베스트 셀러 목록에 여러 번 오른 도서는 하나의 행만 남김
- [도서 명](./research/240716_check_bookinfo.ipynb)
  - 한자 처리
    - [hanja](https://github.com/suminb/hanja)을 이용해 한자를 한글로 변환. 한글 독음이 이미 있는 경우 중복되지 않게 처리
  - 숫자 처리
    - 숫자 사이 구분자 "," 정리 : ex) "1,000" -> "1000"
    - 로마 숫자를 아랍 숫자로 변환
    - 연도 표기 정리 : "\`00"의 형태로 표기된 년도를 정리
      - ex) "\`98 ~ \`07 기출문제 모음" -> "1998 ~ 2007 기출문제 모음"
  - 특수한 unicode로 기입된 문자를 흔히 쓰이는 특수문자로 변환
    - "&#"가 들어가는 token들이 있는지 확인 후 별도 처리
    - ex) "세 명의 삶 ＼ Q. E. D." -> "세 명의 삶 \ Q. E. D."
  - 괄호속 내용 추출 후 BName_sub column에 정리
    - ex) "전지적 루이 &후이 시점(양장본)" -> "(양장본)"만 BName_sub에 분리
- [저자 명](./research/240716_check_bookinfo2.ipynb)
  - 여러 명이 제작자로 기재된 경우, 맨 앞의 제작자만 남김
    - 여러 명이 기재되어 있었는지 여부를 Author_mul에 bool형태로 기록
      - ex) "정홍섭 글 이준성 그림" -> "정홉섭 글", True
  - 이름 뒤에 붙은 기타 문자열 처리
    - 역할에 대한 단어 : "글", "시", "역", "지음", "평역" 등 총 72가지
    - 다수의 사람이 참여했다는 의미의 단어
      - ex) "외 13인", "외 5명", "외"
- 출간일 : DateTime 타입으로 파싱
- ItemId, 정가, 판매가 : 정수 형태로 변환

#### 중고 도서 목록 전처리

- 이상치 처리:
  - 알라딘 페이지에서 '최상', '상', '중' 등급이 있는 것으로 안내하지만, '균일가', '하' 등급도 소수 존재
  - '균일가', '하'는 [하]로 통일
- 배달료 : 2500원으로 통일되어 있어 삭제

#### [인코딩 및 스케일링](./research/240716_encoding_bookinfo.ipynb)

- validation 및 test set의 데이터가 전처리에 영향을 주지 않도록 주의하여 진행
  - train set을 전처리 하면서 결정된 함수 및 관련 내용들을 validation 및 test set에 일괄적으로 적용
- Mecab을 사용해 Category, BName,BName_sub 컬럼을 토큰화
  - [Mecab](https://pypi.org/project/python-mecab-ko/)은 원문 내 띄어쓰기에 의존하기보다 사전을 참조해 어휘를 구분하여 안정적인 결과값을 보여줌
- 도서 명(BName, BName_sub)과 카테고리는 하나의 코퍼스로 통합하여 정수 인코딩
  - 글의 내용이 되는 문장이 아닌 제목이므로, train set의 해당 열에 포함 된 최대한 모든 토큰을 데이터 셋에 포함
  - TF-IDF를 이용한 토큰 정리, 품사나 길이를 기준으로 정리하는 방법 등은 적용하지 않음
- 출판사, 판매 지점, 저자 명에 대해서는 빈도 수 혹은 SalesPoint를 고려한 인기를 반영하여 정수 인코딩
- 날짜 관련 데이터 정수형으로 인코딩
- MinMaxScaling 진행
  - 도서 명과 카테고리 관련 열은 일괄적으로 진행
  - 이외의 열은 개별적으로 진행

![image](https://github.com/user-attachments/assets/f4a98000-345b-4695-a2e8-0fbfff784d68)

*그림. 전처리,스케일링후 최종 데이터 예시*

![image](https://github.com/user-attachments/assets/251df1b2-ce2d-41a2-a1bb-4b8517ba8771)

*그림. train셋의 평균을 baseline으로 했을 때, Random Forest Regressor 적용 결과 분석 예시*

## 5. 모델 학습 결과 및 평가

<!--내용 검토 및 이미지 업데이트-->
<!--실험 설계 부분과 유기적으로 구성하기-->

- 모델 성능은 RMSE, MAPE, R2 Score 등을 활용하여 평가
- Random Forest Regressor, XGBoost 모델 간의 성능을 비교
  - XGBoost에 대해서는 GridSearchCV를 이용해 각 모델 별로 가장 높은 성능을 내는 hyper parameter 탐색
- 각 모델별로 4 종류의 상황에 대한 실험을 진행
  - Expt. 1 : 모든 종속변수를 이용해 중고도서 가격 예측
    - 종속변수 : Category, BName, BName_sub, quality, store, Author, Author_mul, Publshr, Pdate, RglPrice, SalesPoint
  - Expt. 2 : 세일즈포인트를 제외한 종속변수를 이용해 중고도서 가격 예측
  - Expt. 3 : 세일즈포인트와 정가를 제외한 종속변수를 이용해 중고도서 가격 예측
  - Expt. 4 : 세일즈포인트와 정가를 제외한 종속변수를 이용해 중고도서 할인율 예측
- 모델 평가는 두 가지 방법으로 진행
  - test1 : 초기에 test dataset으로 설정된 데이터셋
    - 69,385종의 도서에 대한 중고도서 156,843건
  - test2 : train set에 포함된 적 없는 도서에 대한 중고 매물로 제한한 데이터셋
    - valid set에서 4,084종의 도서에 대한 중고도서 4,677건
    - test set에서 4,984종의 도서에 대한 중고도서 5,968건
    - 총 7,994종의 도서에 대한 중고도서 10,645 건
- 판매가와 SalesPoint를 학습에서 제외시켜도 안정적인 성능이 나오는지 탐색

### 결과

- Random Forest Regressor (이하 RFR)
  - hyperparameter 별 결과

     ![image](https://github.com/user-attachments/assets/fce0e86d-818d-4a15-a659-b9eae4fce201)

      *그림. RFR 모델 hyperparmeter: default인 경우의 feature importance*

  - salespoint 혹은 정가를 제외한 경우

     ![image](https://github.com/user-attachments/assets/5a47472e-c124-44a7-92ca-9abdaac7fc95)

      *그림. RFR 모델, salespoint를 제외한 경우*

     ![image](https://github.com/user-attachments/assets/abd08aad-d821-4979-9829-3080b950c32a)

      *그림. RFR 모델, 정가 제외한 경우*

  - train set에 포함된 적 없는 도서에 대한 중고 매물로 한해서 평가한 경우

- XGBoost Regressor (이하 XGB)
  - 각 실험 환경에 대해 GridSearch를 진행한 후, 가장 성적이 높았던 7개의 hyperparameter에 대한 모델 평가 결과를 정리
  - GridSearchCV
    - fold = 3
    - 대상 hyperparamter 및 범위
      - *num_boost_round* : [100,1500,2500]
      - *learning_rate* : [0.5,0.3,0.1]
      - *max_depth* : [4,5,6]
      - *min_child_weight* : [1,4,7]
      - *colsample_bytree* : [0.5,1]
      - *subsample* : [0.4,0.7,1]
    - GridSearch 결과 우수했던 hyperparmeter 및 성적
      - *Expt. 1*

        ||h1|**h2**|h3|h4|h5|h6|h7|
        |-|-:|-:|-:|-:|-:|-:|-:|
        |num_boost_round|1500|*2500*|2500|2500|2500|2500|2500|
        |learning_rate|0.3|*0.3*|0.3|0.3|0.3|0.3|0.3|
        |max_depth|6|*6*|6|6|6|6|6|
        |min_child_weight|4|*1*|1|4|4|7|7|
        |colsample_bytree|1|*0.5*|1|0.5|1|0.5|1|
        |subsample|1|*1*|1|1|1|1|1|
        |mean valid score|0.97100|**_0.97207_**|0.97172|0.97141|0.97163|0.97145|0.97143|

        *도표. 제외한 종속 변수 없는 상황에서 best parameter 및 $R^2$ score*

      - *Expt. 2*

        ||**h2**|h3|h4|h7|
        |-|-:|-:|-:|-:|
        |num_boost_round|2500|2500|2500|2500|
        |learning_rate|0.3|0.3|0.3|0.3|
        |max_depth|6|6|6|6|
        |min_child_weight|1|1|1|7|
        |colsample_bytree|0.5|1|1|1|
        |subsample|1|1|1|1|
        |mean valid score|**0.97139**|0.97110|0.97058|0.97049|

        *도표. SalesPoint 제외한 상황에서 best parameter 및 $R^2$ score*

      - *Expt. 3*

        ||h1|h2|
        |-|-:|-:|
        |mean test score|0.9048|0.9014|
        |std test score|0.0060|0.0059|

        *도표. SalesPoint, RglPrice 제외한 상황에서 best parameter 및 score*

      - *Expt. 4*

        ||h2|h3|
        |-|-:|-:|
        |mean test score|0.8277|0.8262|
        |std test score|0.0022|0.0021|

        *도표. SalesPoint, RglPrice 제외하고 할인율 예측할 때 best parameter 및 score*

    - XGB 평가에 최종적으로 사용 된 hyperparmeter

      ||h0|h1|h2|h3|
      |-|-:|-:|-:|-:|
      |colsample_bytree|1|1|1|0.5|
      |learning_rate|0.3|0.5|0.3|0.3|
      |max_depth|6|6|6|6|
      |n_estimators|100|2100|2100|2100|

      *도표. XGB 평가에서 사용한 hyperparmeter*

    ![image](https://github.com/user-attachments/assets/e2f0a3e1-0c9f-4b34-b65a-43c0eedf0ad7)

    *그림. XGB 모델, hyperparmeter: default*



- 제외한 column이 없는 경우

||h0|h1|h2|h3|
|-|-:|-:|-:|-:|
|RMSE|||||
|MAPE|||||
|R2 score|||||

*표. hyperparmeter 별 성적*

- column 중 세일즈포인트를 제외한 경우

||h0|h1|h2|h3|
|-|-:|-:|-:|-:|
|RMSE|||||
|MAPE|||||
|R2 score|||||

*표. hyperparmeter 별 성적*

- column 중 세일즈포인트, 정가를 제외한 경우

||h0|h1|h2|h3|
|-|-:|-:|-:|-:|
|RMSE|||||
|MAPE|||||
|R2 score|||||

*표. hyperparmeter 별 성적*


      ![image](https://github.com/user-attachments/assets/851c1c21-6e9c-4aec-9910-fe892b22700e)

        *그림. XGB*

## 6. 결과 분석

- 가장 성능이 좋았던 분석 모델은 default hyperparmter 설정의 XGBoost 모델
  - R2_score=0.95, MAPE=0.08, RMSE=811
  - GridSearch 결과, validation 및 test set에서 성적이 더 잘 나오는 hyperparmeter 설정도 있었음
  - 하지만 train set에서 등장하지 않았던 도서의 경우엔 hyperparameter가 default인 경우에 더 안정적인 성능을 보임
  - 어떤 경우에서도 base line(train set에서의 평균값)에 비해 월등히 좋은 성능을 보임
- feature importance 분석 결과를 바탕으로 중고가 예측에 정가, 도서 명, 중고 등급 등이 주요한 역할을 하는 것을 확인
- 정가를 학습 데이터에 포함하지 않았을 때, train set에 등장 한 적 없는 종류의 도서에 대해서는 중고 판매가 예측 성능이 많이 떨어지는 것을 발견

## 7. 결론 및 한계

### 결론

- Random Forest Regressor 및 XGBoost 등 간단한 모델로도 높은 성능의 모델 개발 가능
- 도서 명, 중고 등급, 정가, 출판일, 저자 등 실물 중고 도서에서 간단히 확인 가능한 특징만으로도 높은 성능이 충분히 가능
- 중요하게 본 칼럼이 일반적인 직관에서 크게 벗어나지 않는 모델로 학습됨
- train set에서 중고 시세를 학습한 적 없는 도서에 대한 중고 가격에 대해서도 큰 차이 없는 성능으로 예측한 것으로 보아, NLP한 결과가 모델에 충분히 반영되었음을 알 수 있음
- 간단한 모델과 default hyperparmeter로도 높은 성능이 나오는 것으로 보아, 알라딘 중고매장에서 중고 도서 판매 가격을 산정하는 가이드라인이 있을 것이라 추측 가능

### 한계 평가

- 정가를 데이터 셋에 포함하지 않는 모델 개발의 어려움
  - 정가를 포함하지 않았을 때, train set에 없는 데이터에 대해서는 중고 판매가 예측 성능이 많이 떨어지는 것을 발견
  - 보완하기 위해 중고 도서의 할인율을 예측하는 모델 개발 시도
    - RFR, XGB는 hyperparameter에 따라 R2 score 0.75~0.82 정도로 학습됨
    - 데이터셋에 포함되지 않았던 도서의 중고 매물도 안정적으로 예측하나, 성능을 더 올리기는 어려웠음
- 저자명, 출판사를 인코딩 중 기타 항목으로 처리할 때 threshold 기준의 구체적인 근거를 제시하지 못 함
  - 알라딘의 Sales Point 및 개인적 경험에서의 인지도를 바탕으로 결정
  - 추가적인 조사를 통해 더 객관적이고 제시 가능한 근거 확립 가능

## 8. 추후 과제

- RNN 등 Neural Network를 이용한 회귀 모델 개발
- 배포 가능한 알라딘 중고도서 데이터 셋으로 확장
  - 중고 판매가 예측 모델 외에도 다양한 모델 개발 가능
    - 도서 정보 및 중고 시장에서의 가격을 바탕으로 알라딘의 SalesPoint 산정법 추정
    - 카테고리와 도서 명, 출판사 등의 정보로 출간 연도 예측
- 베스트 셀러 이외의 도서, 공식 매점에서 판매하지 않는 도서 등으로 데이터 셋 및 프로젝트 확장
  - 베스트 셀러에 포함된 적 없는 도서도 대상으로 하기 위한 크롤링 방법 개발 필요
