# AutoGluon을 활용한 통신사 고객 이탈 예측

이 프로젝트는 **Telco Customer Churn** 데이터셋을 분석하고 **AutoGluon**을 사용하여 이탈 위험이 있는 고객을 식별하는 예측 모델을 구축합니다. 워크플로우에는 포괄적인 데이터 분석, 전처리, 특성 공학(Feature Engineering) 및 AutoML 모델링이 포함됩니다.

## 📂 프로젝트 구조
- `TelcoCustomerChurn_with_autogluon_ver2.ipynb`: 전체 분석 및 모델링 파이프라인이 포함된 메인 Jupyter 노트북입니다.
- `WA_Fn-UseC_-Telco-Customer-Churn.csv`: 분석에 사용된 데이터셋입니다.
- `assets/`: 생성된 시각화 이미지가 저장된 디렉토리입니다.

## 📊 데이터셋 개요
이 데이터셋은 통신사의 고객 정보를 포함하고 있습니다. 타겟 변수는 `Churn` (이탈 여부: Yes/No)입니다.
주요 특성:
- **인구통계 정보**: Gender(성별), SeniorCitizen(고령자 여부), Partner(배우자 유무), Dependents(부양가족 유무).
- **서비스 가입 정보**: PhoneService, MultipleLines, InternetService, OnlineSecurity 등.
- **계정 정보**: Contract(계약 형태), PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges, Tenure(가입 기간).

## 🛠 전처리 및 특성 공학 (Preprocessing & Feature Engineering)

### 1. 데이터 클리닝
- **TotalCharges**: 빈 문자열 값이 포함되어 있어 `NaN`으로 변환한 후, 수치 분석이 가능하도록 `0`으로 채웠습니다.
- **Churn**: 상관관계 분석을 위해 타겟 컬럼을 인코딩(Yes=1, No=0) 했습니다 (AutoGluon은 원본 라벨을 자동으로 처리합니다).

### 2. 특성 공학 (Feature Engineering)
모델 성능 향상을 위해 다음과 같은 파생 변수를 추가했습니다:
- **`TenureGroup`**: 가입 기간에 따른 행동 패턴을 파악하기 위해 고객을 기간별 그룹(예: '0-1 Year', '1-2 Years', '5+ Years')으로 분류했습니다.
- **`ServiceCount`**: 각 고객이 가입한 부가 서비스(OnlineSecurity, TechSupport, StreamingTV 등)의 총 개수를 계산했습니다. 이는 고객의 서비스 관여도를 정량화하는 데 도움이 됩니다.

## 📈 탐색적 데이터 분석 (EDA)

### 1. 이탈 여부 분포 (Churn Distribution)
전체 이탈률을 시각화하여 클래스 불균형을 파악합니다.
![이탈 여부 분포](assets/churn_distribution.png)

### 2. 성별에 따른 이탈률 (Churn by Gender)
성별에 따른 이탈률의 차이를 분석합니다.
![성별에 따른 이탈률](assets/churn_by_gender.png)

### 3. 계약 형태별 서비스 이용 현황 (Service Usage by Contract Type)
계약 기간(Month-to-month, One year, Two year)에 따라 가입한 서비스 수가 어떻게 달라지는지 조사합니다.
![계약 형태별 서비스 이용 수](assets/service_count_by_contract.png)

## 🤖 AutoGluon을 이용한 모델링

이 프로젝트는 자동화된 머신러닝을 위해 **AutoGluon의 `TabularPredictor`**를 사용합니다. AutoGluon은 자동으로 다음을 수행합니다:
- 데이터 타입을 추론하고 특성을 처리합니다.
- 여러 모델(GBM, CatBoost, RandomForest, Neural Networks 등)을 학습시킵니다.
- 성능을 극대화하기 위해 최적의 모델들을 스태킹(Stacking)하고 앙상블(Ensemble)합니다.
- 홀드아웃 검증 세트를 사용하여 모델을 평가합니다.

### 사용법
```python
from autogluon.tabular import TabularPredictor

predictor = TabularPredictor(label='Churn', eval_metric='accuracy').fit(train_data)
predictor.leaderboard()
```

## 🏆 결과
노트북은 정확도(및 ROC-AUC와 같은 기타 지표) 순으로 정렬된 모델 리더보드를 생성합니다. 최종 앙상블 모델은 테스트 세트에서 강력한 예측 성능을 제공합니다.

## 🚀 실행 방법
1. 의존성 패키지 설치: `pip install pandas matplotlib seaborn autogluon`
2. `TelcoCustomerChurn_with_autogluon_ver2.ipynb` 노트북을 실행합니다.
   - *참고: 이 README의 이미지를 다시 생성하려면 노트북(또는 제공된 헬퍼 스크립트 `generate_assets.py`)을 실행해야 합니다.*
