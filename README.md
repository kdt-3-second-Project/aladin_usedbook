# 알라딘 중고책 데이터 기반으로 한 국내 중고서적 가격예측 AI모델
프로젝트 구성원: 오도은, 박예림, 이준성, 정홍섭 / [발표 슬라이드](https://docs.google.com/presentation/d/15EIOMGpadZQf3cT2k0pfClS9DVICLmmf5ZTH1k4XnKc/edit?usp=sharing)

**사용된 스킬 셋**: NumPy, Pandas, Matplotlib, Beautifulsoup, re, Scikit-learn, xgboost, [Mecab](https://pypi.org/project/python-mecab-ko/)

## 1. 프로젝트 개요

### 프로젝트 배경

- 중고 책의 상태, 저자, 장르 등 다양한 요소들을 기반으로한 중고 책 가격 예측
- 상품 별로 편차가 큰 중고 도서의 가격을, 종이 책에서 직접 확인 가능한  정보를 위주로 하여 가격 예측

## 2. 목적

- 크롤링을 통해 알라딘 중고도서 데이터 셋 구축
- Random Forest Regressor, XGBoost 등의 모델을 이용한 알라딘 공식 중고 도서 가격 회귀 예측
- 각 모델의 성능을 여러 성능 지표 및 실험을 통하여 적절히 평가

## 3. 데이터셋

### 1) 데이터 개요

- 알라딘의 중고 상품과 새 책 사이에 url 구조 등을 바탕으로 구별할 수 없지만, 알라딘에서 도서별로 중고 상품을 정리해 놓은 페이지를 제공하는 것을 이용하여 데이터 셋을 체계적으로 구축

#### [알라딘 국내도서 주간베스트셀러](https://www.aladin.co.kr/shop/common/wbest.aspx?BranchType=1)
- 알라딘 사이트 내 주간 베스트셀러 1~1000위에 대한 데이터를 xls 파일로 제공

![image](https://github.com/user-attachments/assets/e330ca44-893c-4fad-8d91-4a2f520c13af)

#### [알라딘 온라인 중고서적](https://www.aladin.co.kr/shop/UsedShop/wuseditemall.aspx?ItemId=254468327&TabType=3&Fix=1)
- url 구조 상, 새 책의 ItemId를 이용하여 중고 매물 목록 페이지에 접근할 수 있음
- 판매자 분류 별로 탭이 나눠져 있으며, 알라딘 중고서점 매장에서 판매하는 중고도서의 정보가 그 중 하나의 탭

![image](https://github.com/user-attachments/assets/e8840608-96f8-47e6-954b-5d6e08f47df9)

### 2) 데이터 셋 구조

1. 2000년 1월 1주차 ~ 2024년 7월 2주차의 주간별 베스트셀러 자료
  - 총 1,415,586개의 row로 구성됨

![image](https://github.com/user-attachments/assets/8d74d9a6-3423-4bd3-b0a0-27817761de9c)

2. 위 베스트셀러에 포함된 서적(ItemId)을 기준으로 크롤링한 중고도서 매물 자료
  - 총 784,213개의 row로 구성

![image](https://github.com/user-attachments/assets/6bc6657e-cc45-4830-baaa-fca240733d6e)

- 위 2개의 데이터를 종합하여 하나의 raw data 파일을 얻을수 있었음
  - column은 도서 명, [ISBN13, 부가기호](https://blog.aladin.co.kr/ybkpsy/959340), quality(중고 등급), store(지점), ItemId, 저자,출판사, 출판일, 정가, 판매가, 중고가, 마일리지, 세일즈포인트, 카테고리, 날짜(주간 베스트셀러에 포함된 주)로 총 12개가 있음.
    - 카테고리는 외국어, 종교, 사회과학, 건강/취미 등 총 24개 유형으로 분류됨
    - **세일즈포인트**
      - 판매량과 판매기간에 근거하여 해당 상품의 판매도를 산출한 알라딘만의 판매지수
      - 최근 판매분에 가중치를 두어, 팔릴수록 올라가고 덜 팔리면 내려감
      - 최근 베스트셀러는 점수가 높으며, 꾸준히 팔리는 스테디셀러들도 어느 정도 점수를 유지함
      - ‘SalesPoint'는 매일매일 업데이트 되고, 크롤링 시점에서의 값이 저장되어 있음
    - **중고가 및 품질**
      - 중고가는 품질(중고 등급)의 큰 영향을 받으며, '균일가' 및 '하', '중', '상', '최상'으로 구분되어있음
      - 품질이 높을수록 중고가가 높은 경향이 있음
      - 같은 품질이라도 가격이 다르거나, 낮은 품질의 매물보다 더 가격이 싼 경우가 종종 있음

![image](https://github.com/user-attachments/assets/caa98ef5-b5be-47d9-a9c4-9ff236ecdb48)
*참고 항목*

## 4. 문제 설정

**목표**: 품질, 판매 지점, 저자, 출판사, 출판일, 정가 등의 값을 이용하여 알라딘에서 공식으로 판매하는 중고 서적 가격을 예측 하고 분석하고자 함

### 1) 종속 변수/ 독립 변수

- 종속 변수를 제외한 항목 중에서 독립변수 선정
  - BName_sub (도서명에서 괄호 안의 내용), Author_mul (복수의 저자가 참여했는지 여부) 등 파생 항목 포함

| 종속 변수 | 독립 변수 |
|---------|---------|
| Price (중고가)| ItemId, quality, store, BName, BName_sub, Author, Author_mul, Publshr (출판사), Pdate (출판일), RglPrice (정가), SlsPrice (새 책 판매가), Category |

### 1) 실험 설계

- sklearn.train_test_split을 사용하여 train 64%, validation 16%, test 20% 비율로 분리
- 모델 종류 별 비교
  - Random Forest, XGBoost 모델 간의 성능을 비교
  - Random Forest, XGBoost, MLP 학습을 할 때 grid search를 이용해 각 모델 별로 가장 높은 성능을 내는 hyper parameter 탐색
  - RMSE, MAPE, MASE, R2 Score 등의 회귀 평가 지표를 사용하여 성능을 각 모델 별로 분석

## 5. [전처리](./code/)

### 1) 전체 과정

#### 베스트 셀러 목록 전처리
- 결측치 처리
  - 저자 명, 구분, 출판사, 카테고리 등에 결측치가 있는 행의 개수 1,214개
    - 실제 도서도 있지만, MD 굿즈, 강연등 도서가 아닌 데이터 다수 존재
- 중복 도서 처리 : 베스트 셀러 목록에 여러 번 오른 도서는 하나의 행만 남김
- [도서명 전처리](./research/240716_check_bookinfo.ipynb)
  - 한자 처리
    - [hanja](https://github.com/suminb/hanja)을 이용해 한자를 한글로 변환. 한글 독음이 이미 있는 경우 중복되지 않게 처리
  - 숫자 사이 구분자 "," 정리 : ex) "1,000" -> "1000"
  - 로마 숫자를 아랍 숫자로 변환
  - 특수한 unicode 문자 detect 및 유사한 일반적 문자로 변환
    - '&#'가 들어가는 token들이 있는지 확인 후 별도 처리
  - 괄호속 내용 추출 후 BName_sub column에 정리
    - ex) "전지적 루이 &후이 시점(양장본)" -> "(양장본)"만 BName_sub에 분리
- [저자명 전처리](./research/240716_check_bookinfo2.ipynb)
  - 여러 명이 제작에 참여한 경우, 맨 앞의 참여자만 남김
    - 여러 명이 제작에 참여했는지 여부를 Author_mul에 bool형태로 기록
      - ex) 정홍섭 글 이준성 그림 -> 정홉섭 글, True
  - 이름 뒤에 붙은 기타 문자열 처리
    - 역할에 대한 단어 : '글', '시', '역' 등 총 72가지
    - 다수의 사람이 참여했다는 의미의 단어
      - ex) "외 13인", "외 5명", "외"

#### 중고 도서 목록 전처리

- 이상치 처리: 균일가, 하는 [하]로 통일
  - 총 등급은 [최상,상,중,하]로 구분
- 배달료 처리: 2500원으로 통일되어 있어 삭제

#### [인코딩 및 스케일링](./research/240716_encoding_bookinfo.ipynb)

- validation 및 test set의 데이터가 전처리에 영향을 주지 않도록 주의하여 진행 함
  - train set을 전처리 하면서 결정된 함수 및 관련 내용들을 validation 및 test set에 일괄적으로 적용
- Mecab을 사용해 Category, BName,BName_sub 컬럼을 토큰화
  - [Mecab](https://pypi.org/project/python-mecab-ko/)은 원문 내 띄어쓰기에 의존하기보다 사전을 참조해 어휘를 구분하여 안정적인 결과값을 보여줌
- 도서 명(BName, BName_sub)과 카테고리는 하나의 코퍼스로 통합하여 정수 인코딩
  - 글의 내용이 되는 문장이 아닌 제목이므로, train set의 해당 열에 포함 된 최대한 모든 토큰을 데이터 셋에 포함
  - TF-IDF를 이용한 토큰 정리, 품사나 길이 별로 정리하는 방법 등은 적용하지 않음
- 출판사, 판매 지점, 저자 명에 대해서는 빈도 수 혹은 SalesPoint를 고려한 인기를 반영하여 정수 인코딩
- 날짜 관련 데이터 정수형으로 인코딩
- MinMaxScaling 진행
  - 하나의 코퍼스에 해당하는 도서 명과 카테고리 관련 열은 일괄적으로 진행
  - 이외의 열은 개별적으로 진행

![image](https://github.com/user-attachments/assets/f4a98000-345b-4695-a2e8-0fbfff784d68)  *그림. 전처리,스케일링후 최종 데이터 예시*

![image](https://github.com/user-attachments/assets/251df1b2-ce2d-41a2-a1bb-4b8517ba8771) *그림. train셋의 평균을 baseline으로 했을 때, Random Forest Regressor 적용 결과 분석 예시(실제 및 예측값 분포/오차의 분포/성능 지표)*

## 5. 모델 학습 결과 및 평가

- 모델 성능은 RMSE, MAPE, score 등을 활용하여 평가
  - Random Forest Regressor
     ![image](https://github.com/user-attachments/assets/fce0e86d-818d-4a15-a659-b9eae4fce201)
      *그림. RFR 모델 hyperparmeter: default, sample data의 feature importance*
    - test set을 train set에 포함된 적 없는 책에 대한 중고매물로 꾸린 경우
     ![image](https://github.com/user-attachments/assets/5a47472e-c124-44a7-92ca-9abdaac7fc95)
      *그림. RFR 모델 salespoint를 제외한 경우
     ![image](https://github.com/user-attachments/assets/abd08aad-d821-4979-9829-3080b950c32a)
      *그림. RFR 모델 정가 제외한 경우
  - XGBoost
    ![image](https://github.com/user-attachments/assets/e2f0a3e1-0c9f-4b34-b65a-43c0eedf0ad7)
    *그림. XGBoost 모델 hyperparmeter: default*
    - test set을 train set에 포함된 적 없는 책에 대한 중고매물로 꾸린 경우
      ![image](https://github.com/user-attachments/assets/851c1c21-6e9c-4aec-9910-fe892b22700e)
        *그림. XGBoost*

## 6. 결과 분석

- 가장 성능이 좋았던 분석 모델은 default hyperparmter 설정의 XGBoost 모델
  - 성능지표가 R2_score=0.95, mape=0.08, rmse=811
  - validation 및 test set에서 성적이 더 잘 나오는 hyperparmeter 설정도 있었지만, 
- feature importance 분석 결과를 바탕으로 중고가 예측에 정가, 도서 명, 중고 등급 등이 주요한 역할을 하는 것을 확인
- 정가를 포함하지 않았을 때 예측 성능이 많이 떨어지는 것을 발견
- RMSE, MAPE, R2_score 등 다양한 성능 지표의 공통적인 경향 및 각각의 특징에 대해 생각해볼 수 있었음

## 7. 결론 및 향후 연구

### 결론

- Random Forest regressor 및 XGBoost 등 간단한 모델로도 높은 성능의 모델 개발 가능
- 도서 명, 중고 등급, 정가, 출판일, 저자 등 중고 도서에서 직접 확인 가능한 특징만으로도 충분히 가능
- train set에서 중고 시세를 학습한 적 없는 중고 매물에 대해서도 높은 성능으로 예측
- 알라딘에서 중고 도서의 공식 판매 가격을 산정하는 가이드라인이 있을 것이라 추측 가능

### 추후과제

- 배포 가능한 알라딘 중고도서 데이터 셋으로 확장

- 중고 판매가 외에 다른 값을 예측하는 다양한 모델 개발 가능
    • 도서 정보 및 중고 시장에서의 가격을 바탕으로 알라딘이 산정한 SalesPoint 추정
    • 카테고리와 도서 명, 출판사 등의 정보로 출간 연도 예측

- 베스트 셀러 이외의 도서, 공식 매점에서 판매하지 않는 도서 등으로 프로젝트 확장

## 8. 한계점

- 베스트 셀러 목록에 포함된 도서를 대상으로 제한
  - 베스트 셀러에 포함된 적 없는 도서도 대상으로 하기 위한 크롤링 방법 연구 필요
- RNN 등을 이용한 모델 개발의 실패
  - hyperparameter, optimizer 등을 조정 해봤지만 일정한 값을 출력하는 모델 밖에 얻지 못함
  - 모델 구조 등의 개선 및 추가적인 실험 필요
- 정가를 포함하지 않는 모델 개발의 어려움
  - 정가를 포함하지 않았을 때 예측 성능이 많이 떨어지는 것을 발견, 중고 도서 할인율을 예측하는 모델 개발 시도
  - 모델과 hyperparameter에 따라 R2 score 0.75~0.82 정도로 예측하는 모델은 개발 완료
  - 데이터셋에 포함되지 않았던 도서의 중고 매물도 안정적으로 예측하나, 성능을 더 올리기는 어려웠음
  - RNN, LSTM 등의 고도화된 모델을 이용해 성능을 더욱 높히는 것을 기대하고 있음
- 저자명, 출판사를 인코딩 중 기타 항목으로 처리할 때 threshold 기준의 구체적인 근거를 제시하지 못 함
  - 알라딘의 Sales Point 및 개인적 경험에서의 인지도를 바탕으로 결정
  - 추가적인 조사를 통해 더 객관적이고 제시 가능한 근거 확립 가능
