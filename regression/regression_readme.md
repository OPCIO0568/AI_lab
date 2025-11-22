# 🇬🇧 London Property Price Prediction (런던 부동산 가격 예측)

PyTorch를 활용하여 런던의 부동산 목록 데이터를 분석하고, 집값(Price)을 예측하는 딥러닝 회귀(Regression) 모델을 구축한 프로젝트입니다.

## 📂 Dataset Info
- **Source**: [Kaggle - London Property Listings Dataset](https://www.kaggle.com/datasets/sezermehmetemre/london-property-listings-dataset)
- **Description**: 런던 지역의 부동산 매물 정보 (가격, 주택 유형, 침실/욕실 수, 크기, 위치 등)
- **Target Variable**: `Price` (집값, GBP £)

## 🛠 Tech Stack
- **Language**: Python
- **Deep Learning**: PyTorch (NN Module, DataLoader)
- **Data Processing**: Pandas, NumPy, Scikit-learn
- **Visualization**: Matplotlib (Optional)

## 🚀 Key Modifications & Workflow (수정 및 개선 사항)

기존 캘리포니아 집값 예측 코드를 기반으로, 런던 데이터셋의 특성에 맞춰 다음과 같은 **전처리 및 트러블슈팅**을 진행했습니다.

### 1. Data Loading & Preprocessing
- **데이터 로드 방식 변경**: KaggleHub 경로 다운로드 방식에서 로컬 CSV 파일(`pd.read_csv`) 로드 방식으로 변경하여 데이터 접근성 확보.
- **범주형 데이터 처리**: `Property Type` 등 문자열(String)로 된 컬럼이 모델 학습 시 에러(`ValueError: could not convert string to float`)를 유발함에 따라, 이를 수치형으로 변환(One-Hot Encoding)하거나 수치형 컬럼 위주로 데이터를 재구성.

### 2. Feature Scaling (정규화)
- **Input Scaling (X)**: `StandardScaler`를 사용하여 데이터의 평균을 0, 표준편차를 1로 맞추어 학습 안정성 확보.
- **Target Scaling (y) [중요]**:
    - 초기 학습 시 집값의 단위가 너무 커서(수억~수십억) Loss 값이 비정상적으로 폭발하는 현상 발생.
    - **해결**: Target(y) 값에도 `StandardScaler`를 적용하여 학습을 진행하고, 예측 후 `inverse_transform`을 통해 원래 가격으로 복원하는 파이프라인 구축.

### 3. Tensor Transformation
- **차원 불일치 해결**: Pandas Series를 PyTorch Tensor로 변환하는 과정에서 발생한 Shape 에러 해결.
    - `.values`를 사용하여 NumPy 배열로 변환 후, `.reshape(-1, 1)`을 통해 `(N, 1)` 형태의 2차원 텐서로 명시적 변환.

## 🧠 Model Architecture
- **Input Layer**: 데이터 Feature 개수에 맞춘 입력층
- **Hidden Layers**:
    - FC Layer (64 units) + ReLU
    - FC Layer (32 units) + ReLU
- **Output Layer**: 1 unit (집값 예측)
- **Optimizer**: Adam (`lr=0.001`)
- **Loss Function**: MSELoss (Mean Squared Error)

## 📊 Performance (최종 결과)

테스트 데이터셋(Test Set)에 대한 최종 평가 결과입니다.

| Metric | Value (Scaled) | Interpretation |
|--------|---------------:|----------------|
| **MSE** | 0.1149 | 평균 제곱 오차 |
| **RMSE** | 0.3390 | 오차의 표준편차 |
| **MAE** | 0.1932 | 평균 절대 오차 |
| **R² Score** | **0.8894** | **모델 설명력 (약 89%)** |

---
