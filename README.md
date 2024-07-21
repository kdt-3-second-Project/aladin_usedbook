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
    - Random Forest, XGBoost, MLP 모델 간의 성능을 비교
    - Random Forest, XGBoost, MLP
      - 독립변수로 사용한 기타 통계 항목(이하 참고 통계 항목)은 데이터 셋에 포함된 다른 통계 항목.
    - Random Forest, XGBoost, MLP 학습을 할 때 grid search를 이용해 각 모델 별로 가장 높은 성능을 내는 hyper parameter 탐색
 
### 2) 참고 feature 설정
- 종속변수를 제외한 통계 항목 중에서 독립변수 선정
- 도메인 지식을 활용하여 종속 변수 별로 해당하는 통계 항목 선별
  

------

## 5. 모델 학습 결과 및 평가
------

## 6. 결과 분석
------

## 7. 결론 및 향후 연구
------



