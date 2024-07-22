# 알라딘 중고책 데이터 기반으로 한 국내 중고서적 가격예측 AI모델
--------
프로젝트 구성원: 오도은, 박예림, 이준성, 정홍섭

**사용된 스킬 셋**: NumPy, Pandas, Matplotlib, Scikit-learn, xgboost, mecab

## 1. 프로젝트 개요

### 프로젝트 배경
- 중고 책의 상태, 저자, 장르 등 다양한 요소들을 기반으로한 중고 책 가격 예측
- 편차가 큰 중고 책 시장에서 특정 책의 인기도와 희귀성을 통한 가격예측
--------
## 2. 목적

-  Random Forest Regressor, XGBoost 및 Muliti layer perceptron, RNN 모델을 이용하여 예측
-  적절한 성능지표를 이용하여 회귀 예측에 적합한 모델링 개발
-------
## 3. 데이터셋

### 1) 데이터 개요
  - [알라딘 국내도서 주간베스트셀러](https://www.aladin.co.kr/shop/common/wbest.aspx?BranchType=1)
  - 알라딘 사이트 내 주간별 베스트셀러에대한 정보들을 xls 파일로 제공
![image](https://github.com/user-attachments/assets/e330ca44-893c-4fad-8d91-4a2f520c13af)

  - [알라딘 온라인 중고서적]
  - 중고서적에 대한 정보들을 크롤링으로 얻을 수 있음.
    ![image](https://github.com/user-attachments/assets/e8840608-96f8-47e6-954b-5d6e08f47df9)

### 2) 데이터 셋 구조
- 1. 2000년 1월 1주차 ~ 2024년 7월 2주차의 주간별 베스트셀러 자료
     - 총 1,415,586개의 row로 구성됨
       ![image](https://github.com/user-attachments/assets/8d74d9a6-3423-4bd3-b0a0-27817761de9c)
- 2. 위 베스트셀러의 해당하는 서적(ItemId)을 기준으로 중고책 크롤링 자료
     - 총 784,213개의 row로 구성됨
     - ![image](https://github.com/user-attachments/assets/6bc6657e-cc45-4830-baaa-fca240733d6e)

- 위 2개의 데이터를 바탕으로 하나의 raw data 파일을 얻을수 있었음
  - column은 책이름,ISBN13,부가기호, quality, store(지점) , ItemId, 저자 ,출판사, 출판일, 정가, 판매가, 중고가,마일리지, 세일즈포인트, 카테고리,날짜로 총 12개가 있음.
    - 카테고리는 외국어, 종교,사회과학, 건강/취미 등 총 24개 유형으로 분류됨
    - **세일즈포인트**
      - **판매량과 판매기간에 근거하여 해당 상품의 판매도를 산출한 알라딘만의 판매지수법**
      - 최근 판매분에 가중치를 준 판매점수. 팔릴수록 올라가고 덜 팔리면 내려감
      - 최근 베스트셀러는 높은 점수이며, 꾸준히 팔리는 스테디셀러들도 어느 정도 포인트를 유지함
      - ‘SalesPoint'는 매일매일 업데이트 됨
    - **중고가 및 품질**
      - 중고가는 품질별로 다르며 품질은 균일, 하,중,상,최상으로 구분되어있음
      - 통상 품질이 높을수록 중고가가 높음
      - 같은 품질이라도 다른 가격이 존재함

![image](https://github.com/user-attachments/assets/caa98ef5-b5be-47d9-a9c4-9ff236ecdb48)


------

## 4. 문제 설정
**문제**: 품질, 지점, 저자, 출판사, 출판일, 정가등 관련있는 항목 데이터중 중고서적 가격에 영향끼치는 것을 분석 및 예측 하고자 함

### 1) 실험 설계
- sklearn.train_test_split을 사용하여 train 80%, test 20% 비율로, ItemId값의 범 기준으로 층화해 분리
- 모델 종류 별 비교
    - Random Forest, XGBoost, MLP, RNN 모델 간의 성능을 비교
    - Random Forest, XGBoost, MLP, RNN
      - 독립변수로 사용한 기타 통계 항목(이하 참고 항목)은 데이터 셋에 포함된 다른 통계 항목.
    - Random Forest, XGBoost, MLP 학습을 할 때 grid search를 이용해 각 모델 별로 가장 높은 성능을 내는 hyper parameter 탐색
    - 모든 모델의 성능을 비교하고, rmse,mape,mase,r2_score의 회귀 평가 지표를 사용하여 성능을 분
### 2) 참고 창목 설정
- 종속변수를 제외한 통계 항목 중에서 독립변수 선정
- 도메인 지식을 활용하여 종속 변수 별로 해당하는 통계 항목 선별
- BName_sub(도서명에서 괄호안에 있는 내용 ), Author_mul(저자가 여러명 유무)등 파생 변수 생성

| 종속 변수 | 참고 항목 |
|---------|---------|
| Price(중고가)| ItemId, quality, store, BName, BName_sub, Author, Author_mul, Publshr, Pdate, RglPrice, SlsPrice, Category | 

------
## 4. 전처리

### 1)전체 과정
- raw data에서 결측치 삭제
  - 결측치가 있는 행의 개수 1,214개
  - 카테고리 데이터에서 MD 굿즈, 강연등 연관없는 데이터 존재 -> 제거 결정
    
-  [도서명 전처리](https://github.com/kdt-3-second-Project/aladin_usedbook/blob/e40008549c28f741bf72596828fa9913f60399fd/research/240716_check_bookinfo.ipynb) 
  - 한자 처리 : 만약 한자와 똑같은 발음의 한글이 앞 혹은 뒤에 반복된 경우 제거, 이외의 경우는 번역
  - 숫자 사이 , 정리 : ex) 1,000 -> 1000
  - 로마 숫자를 아랍 숫자로 변환
  - 특수문자 detect 및 가까운 특수 문자로 변환 : ex) '&#'
  - 괄호속 내용 추출 후 BName_sub column에 정리 : ex) 전지적 루이 &후이 시점(양장본) -> 양장본만 BName_sub에 분리
 
- [저자명 전처리](https://github.com/kdt-3-second-Project/aladin_usedbook/blob/e40008549c28f741bf72596828fa9913f60399fd/research/240716_check_bookinfo2.ipynb)
  - 여러 명이 제작에 참여한 경우, 맨 앞의 참여자만 남김
    - 여러 명이 제작에 참여했는지 여부를, Author_mul에 bool형태로 기록: ex) 정홍섭 글 이준성 그림 -> 정홉섭 글, True
  - 이름 뒤에 붙은 기타 문자열 처리
    - 기타 문자열: 역할('글', '시', '역', '외' 등 총 72가지), 외 n인, 외

- 중고 목록 전처리
  - 이상치 처리: 균일가, 하는 [하]로 통일
    - 총 등급은 [최상,상,중,하]로 구분
  - 배달료 처리: 2500원으로 통일되어 있어 삭

- [인코딩 및 스케일링](https://github.com/kdt-3-second-Project/aladin_usedbook/blob/e40008549c28f741bf72596828fa9913f60399fd/research/240716_encoding_bookinfo.ipynb)
  - Mecab을 사용해 Category, BName,BName_sub 컬럼을 토큰화
    - Mecab은 원문 내 띄어쓰기에 의존하기보다 사전을 참조해 어휘를 구분하여 안정적인 결과값을 보여줌
  - 저자, 출판사, 지점 인코딩 / 텍스트 열 패딩후 출판 날짜 인코딩
  - train 데이터는 scaler 학습 및 변환, val 및 test 데이터는 학습된 scaler로 변환
-----
## 5. 모델 학습 결과 및 평가
------

## 6. 결과 분석
------

## 7. 결론 및 향후 연구
------



