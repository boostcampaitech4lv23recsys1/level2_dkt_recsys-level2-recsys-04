![image](https://user-images.githubusercontent.com/79916736/210174229-e5832e4d-e904-4f14-81a4-978e879f298d.png)

# Deep Knowledge Tracing

# 팀원 소개

| <img src="https://user-images.githubusercontent.com/79916736/207600031-b46e76d2-cba3-4c94-9fc3-d9f29cd3bef8.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207600420-dd537303-d69d-439f-8cc8-5af648fe8941.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207601023-bbf9e64f-1447-41d8-991f-677593094592.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/207600724-c140a102-39fc-4c03-8109-f214773a64fc.png" width=200> | <img src="https://user-images.githubusercontent.com/79916736/208005357-e98d106d-a207-4acd-ab4b-1abf7dbcb69f.png" width=200> | <img src="https://user-images.githubusercontent.com/65999962/210237522-72198783-f40c-491b-b8a7-6e6badf6cc24.jpg" width=200> |
| :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------: |
|                                           [김성연](https://github.com/KSY1526)                                            |                                           [배성재](https://github.com/SeongJaeBae)                                            |                                            [양승훈](https://github.com/Seunghoon-Schini-Yang)                                            |                                         [조수연](https://github.com/Suyeonnie)                                          |                                            [황선태](https://github.com/HSUNEH)                                            |                                            [홍재형](https://github.com/secrett2633)                                            |



# 🏆️ 프로젝트 목표
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/65529313/168472960-0eac76e2-4fe3-4ebc-b093-f9c0aab59859.png" /></p> -->
- 사용자의 문제 풀이 기록을 보고 다음 문제의 정답 여부를 맞추는 모델 설계
- 사용자의 지식 상태를 추적하는 딥러닝 모델 설계

<br /> 
<br /> 

# 💻 활용 장비
- GPU Tesla V100-PCIE-32GB

<br /> 
<br /> 

# 🙋🏻‍♂️🏻‍♀️ 프로젝트 팀 구성 및 역할
- **김성연**: EDA, 전반적인 팀 프로젝트 타임라인 잡기, 부스팅 모델 적용 작업
- **배성재**: 협업을 매끄럽게 하기 위한 코드 정리 및 Git 담당하기, ELO 등 피처엔지니어링 진행
- **양승훈**: 전반적인 그래프 모델 탐색, GKT, Saint+ 모델 적용 작업
- **조수연**: EDA 진행, 부스팅 모델과 MF 모델 적용 작업
- **홍재형**: GKT 모델 적용 작업, Optuna, K-Fold Cross Validation 등을 이용한 모델 고도화
- **황선태**: 전반적인 시퀀셜 모델 탐색, Saint+와 LightGCN 모델 적용 작업


<br /> 
<br /> 

# 🏗️ Model Architecture
<!-- <p align="center"><img src="https://user-images.githubusercontent.com/65529313/168473170-938e1ce0-395f-40be-9118-ea127668b11d.png" /></p> -->

- 범주형 데이터 처리에 좋은 성능을 내는 Catboost Model 사용
- 피처 엔지니어링 후 유저 단위로 LGBM 사용
- Surprise 라이브러리를 이용한 심플한 MF Model 사용
- 시퀀셜 정보를 담는 딥러닝 모델 Saint_plus 사용
- 유저와 문제 간 깊은 상호작용을 잘 표현하는 그래프 기반 LightGCN Model 사용
- 그래프와 시퀀셜 두 형태를 모두 가지고 있는 GKT Model 사용
~~~
EDA 내 개인 별 EDA와 피처엔지니어링이 담겨있습니다.
requirements.txt 내 라이브러리만 다운받으면 실행시키는데 무리 없습니다.
단, LightGCN 모델의 경우 폴더 내 install.sh 을 실행시킨 가상환경에서만 작동합니다.
자세한 설명은 발표자료 및 레포트를 참고해주세요.
~~~

<br /> 
<br /> 



# 💯 프로젝트 수행 결과 - 최종 Private 2등

|리더보드| auroc  |     순위     |
|:--------:|:------:|:----------:|
|public| 0.8362 |  **2위**   |
|private| 0.8549 | **최종 2위** |

![image](https://user-images.githubusercontent.com/79916736/210174782-89b8297a-dd02-4585-aff2-873a4717fb2c.png)
