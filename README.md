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

------

## 4. 문제 설정
------

## 5. 모델 학습 결과 및 평가
------

## 6. 결과 분석
------

## 7. 결론 및 향후 연구
------



