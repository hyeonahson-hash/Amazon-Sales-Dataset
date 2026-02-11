# 📱 Telco Customer Churn Prediction: 이탈 방지를 위한 정밀 타격 모델링

> **"고객은 왜 떠나는가?"** 본 프로젝트는 통신사 고객 데이터를 심층 분석하여 이탈 원인을 데이터로 증명하고, 최신 AutoML(AutoGluon)과 하이퍼파라미터 최적화(Optuna)를 활용하여 최적의 이탈 예측 솔루션을 구축합니다.

---

## 1. 프로젝트 주제
- **목표**: 고객 이탈(Churn) 패턴 분석 및 예측 모델을 통한 리텐션 전략 수립
- **핵심 과제**: 
    - 가입 기간, 결제 수단, 인구통계학적 특성에 따른 이탈 트리거 발굴
    - 특성 공학(Feature Engineering)을 통한 모델 예측력 극대화
    - AutoGluon 앙상블과 Optuna-LightGBM 간의 성능 비교 분석

## 2. 데이터 셋 개요
- **Source**: Kaggle Telco Customer Churn Dataset
- **Size**: 7,043 rows, 21 columns
- **Target**: `Churn_Numeric` (0: 유지, 1: 이탈)

---

## 3. 탐색적 데이터 분석 (EDA) & Deep Insights
노트북 분석 과정에서 도출된 5가지 핵심 유의미 컬럼별 인사이트입니다.

### ⏳ 3.1 가입 기간 (Tenure): "초기 6개월의 높은 장벽"
* **Fact**: 이탈 고객의 상당수가 가입 초기 **1~6개월**에 집중되어 있으며, 24개월 이상 유지 시 이탈률이 급격히 낮아짐.
* **Insight**: 신규 고객 대상의 웰컴 프로모션과 초기 서비스 적응 지원(On-boarding)이 장기 고객 전환의 핵심임.

### 💳 3.2 결제 수단 (Payment Method): "수동 결제의 심리적 저항"
* **Fact**: **Electronic check** 결제 고객의 이탈률이 자동 결제(신용카드/은행이체) 고객보다 약 2.5배 높음.
* **Insight**: 매달 직접 결제하는 방식은 요금 체감도를 높여 이탈을 유발함. 자동이체 전환 시 혜택을 주는 전략적 유도가 필요함.

### 👴 3.3 연령대 (Senior Citizen): "고령층의 취약한 연결성"
* **Fact**: 고령층(Senior Citizen)의 비중은 낮으나 이탈률은 젊은 층보다 높으며, 특히 기술 지원 서비스 부재 시 이탈률이 상승함.
* **Insight**: 고령층 고객을 위한 전담 기술 지원이나 단순화된 서비스 패키지 제안이 이탈 방지에 효과적임.

### 🛠 3.4 부가 서비스 (Services): "결합할수록 강력해지는 락인(Lock-in)"
* **Fact**: Online Security, Tech Support 등 **부가 서비스를 3개 이상** 사용하는 고객은 1개 이하 사용 고객보다 이탈률이 현저히 낮음.
* **Insight**: 단순 인터넷 판매를 넘어 보안 및 지원 서비스를 결합한 패키지 판매가 고객의 스위칭 비용(Switching Cost)을 높임.

### 👨‍👩‍👧‍👦 3.5 가족 형태 (Partner & Dependents): "가구 단위 결합의 안정성"
* **Fact**: 배우자(Partner)와 부양가족(Dependents)이 모두 있는 가구는 1인 가구 대비 가입 기간이 길고 요금 민감도가 낮음.
* **Insight**: 1인 가구에게는 가격 혜택을, 가족 단위 고객에게는 가족 결합 공유 데이터를 강조하는 차별화된 타겟팅이 필요함.

---

## 4. 특성 공학 (Feature Engineering)
데이터의 숨겨진 의미를 찾기 위해 다음 파생 변수를 생성하여 모델에 투입했습니다.
- **`Fiber_Price_Impact`**: 광랜 사용자 중 평균 대비 요금 부담 정도를 수치화 (모델 기여도 Top 5)
- **`ServiceCount`**: 총 부가 서비스 이용 개수 (고객 고착도 측정)
- **`Fiber_Premium_Ratio`**: 전체 요금 중 인터넷 요금이 차지하는 비중 계산
- **`LongTerm_Complexity_Risk`**: 장기 고객 중 서비스 이용이 단순하여 이탈 징후가 보이는 군 식별

---

## 5. 모델링 분석 및 성능 비교표

| 평가 항목 | **Optuna (LightGBM)** | **AutoGluon (Ensemble)** | 비고 |
| :--- | :---: | :---: | :--- |
| **ROC-AUC** | 0.8407 | **0.8465** | **AutoGluon 우세** |
| **Accuracy** | **0.8070** | 0.8048 | Optuna 우세 |
| **Precision** | 0.6300 | **0.6713** | AutoGluon 승 (오진율 낮음) |
| **Recall** | **0.5300** | 0.5187 | Optuna 승 (탐지율 높음) |

---

## 6. 최종 결과 및 시사점
- **최종 모델**: AutoGluon의 `WeightedEnsemble_L2` (AUC 0.8465) 채택
- **비즈니스 결론**: 
    1. 이탈 위험 상위 100명 명단 추출 결과, 대부분 **월 단위 계약 + 광랜 사용자**임을 확인.
    2. 이들을 대상으로 '장기 약정 전환 시 요금 할인' 캠페인 우선 집행 제언.
    3. 데이터 기반의 `Fiber_Price_Impact` 변수가 이탈 예측에 결정적 역할을 함을 입증.

---
### 🛠 기술 스택 (Tech Stack)
- **Python**, **Pandas**, **Scikit-learn**
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: **AutoGluon**, **Optuna**, **LightGBM**