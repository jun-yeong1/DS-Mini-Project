# ESS 배터리 수명 예측

초기 100사이클 데이터만으로 배터리 셀의 총 수명(`cycle_life`, 방전 용량이 공칭의 80%에 도달하는 사이클 수)을 **회귀**로 예측하고, 원논문 성능(MAPE 9.1%)과 비교한다. ESS에서 셀 교체 시점을 사전에 계획하는 데 쓰는 것을 목표로 한다.

## 프로젝트 개요

- 데이터셋: MIT-Stanford Battery Dataset (Severson et al., _Nature Energy_, 2019)
- 학습 데이터: Batch 1 (2017-05-12)
- 평가 데이터: Batch 2 (2018-02-20) / 추가 검증: Batch 3 (2018-04-12)
- 태스크: **Regression** (Cycle Life 예측)

## 파일 구조

```
├── data/
│   ├── README.md          # 데이터 출처, 폴더 구성, 분석 대상 셀
│   ├── *.mat              # 원본 데이터 3개 (직접 넣는다, 저장소에는 올리지 않음)
│   └── processed/         # preprocess.py가 만드는 표 (저장소에는 올리지 않음)
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   └── 03_modeling.ipynb
├── src/
│   ├── preprocess.py      # .mat → 셀 단위 표 (data/processed/cells.pkl)
│   ├── features.py        # feature 정의, feature 세트(A/B/C), 모델링 대상 셀
│   └── train.py           # 학습·검증·테스트, 결과 저장
├── results/
│   ├── model_performance.csv
│   └── predictions_b2.csv # 최종 모델의 Batch 2 셀별 예측 (오류 분석용)
├── requirements.txt
└── README.md
```

## 환경 설정

```bash
git clone https://github.com/jun-yeong1/DS-Mini-Project
cd DS-Mini-Project
pip install -r requirements.txt
```

원본 `.mat` 3개를 `data/` 폴더에 그대로 넣는다(다운로드 방법은 [data/README.md](data/README.md) 참고). 실행 순서:

```bash
python src/preprocess.py   # 원본 .mat → data/processed/cells.pkl (약 1초)
python src/train.py        # 모델 학습·평가 → results/ (약 5초)
```

## EDA

- **Cycle Life 분포**
  - 모델링 대상 119셀(EOL 도달) 중 단수명(<500) 28개, 장수명(>1000) 31개
  - 핵심 발견: Batch마다 분포가 다르고(평균 b1 788, b2 566, b3 1,060), 단수명은 전부 b2다. b2 정답 39개 중 30개가 학습 Batch(b1)의 최솟값(534) 미만이다.
- **열화 곡선 분석**
  - 대부분의 셀에서 수명의 앞쪽 절반은 거의 평평하고 후반에 급격히 가속되며, knee는 수명의 약 77%(b1 0.75, b2 0.74, b3 0.80)에서 나타난다.
  - 핵심 발견: 초기 방전 용량(QD)만으로는 셀이 구분되지 않는다. knee는 후반 정보라 입력으로 쓸 수 없다.
- **ΔQ(V) 곡선 분석**
  - Q100(V) − Q10(V)는 단수명 셀에서 더 깊다(셀별 최솟값 평균 −0.061 대 −0.021).
  - 핵심 발견: 초기 열화가 전압별 곡선에서는 보이고, `log10 Var[ΔQ]`와 수명의 그룹 내 상관은 −0.84(b1), −0.76(b3)이다. 사이클 간격이 좁으면(5 사이클 안) 신호가 사라진다.
- **충전 속도(C-rate)와 수명의 관계**
  - b2·b3는 0→80% 충전이 모두 10분이라 평균 C-rate가 같고, b1은 8.9~13.3분이다.
  - 핵심 발견: b1에서는 같은 (C1, C2)에서 고전류 구간이 길수록 수명이 짧다. b2·b3에서는 단계 간 전류 차이 |C1−C2|가 클수록 수명이 짧다(상관 −0.53~−0.58).
- **추가 확인**
  - 초기 QD·IR·온도의 절대 수준은 Batch·셀 구조(legacy/newstructure)의 지문이라 통합 상관이 그룹 안에서 부호가 반대로 나타난다.
  - 같은 충전 정책도 Batch·구조에 따라 수명이 최대 약 3.6배 차이가 난다.

## Modeling

### 피처 엔지니어링 전략

EDA에서 그룹 안에서도 일관된 신호만 고르고, 중복과 누수를 피했다.

| 세트 | feature                                                       | 근거                                       |
| ---- | ------------------------------------------------------------- | ------------------------------------------ |
| A    | `dq_logvar` = log10 Var[Q100(V) − Q10(V)]                     | ΔQ 단독의 설명력 (핵심)                    |
| B    | A + `contrast`(\|C1−C2\|), `avg_c`(0→80% 평균 C-rate), `soc1` | 충전 정책(사이클 시작 전 정보)의 추가 기여 |
| C    | B + `tavg_mean`(초기 100사이클 평균 온도)                     | 온도의 추가 기여                           |

- 제외: 초기 QD·IR(Batch 지문), knee·후반 열화 속도(수명 후반 정보라 누수), ΔQ의 min·mean(`dq_logvar`와 r≈1)
- Target: `log10(cycle_life)`. 평가는 `10^ŷ`로 되돌려 MAPE로 계산한다.
- 정제: 별도 실험 셀(`VarCharge`, `SLOWCYCLE`)과 EOL 미도달 셀(12개)은 제외한다.

### 모델 선택 및 근거

- 후보 모델: Baseline(`dq_logvar` 단일 선형 회귀), Elastic Net, Ridge, PLS, 가우시안 과정(GPR), Random Forest
  - 학습 셀이 36개뿐이고 feature 간 상관이 높아 규제·저차원 모델을 포함하고, 학습 범위 밖 정답이 많아 선형·GPR을 포함했다. Random Forest는 비선형 비교군이다.
- 최종 모델: **세트 B / Random Forest** (사전에 정한 규칙: Valid MAPE 최소)
- 선택 이유: 후보 16개 중 Valid MAPE가 가장 낮다(5.86%).
  - **한계**: Valid hold-out이 7셀뿐이라 선택이 불안정하다. 실제로 Test(Batch 2)에서는 선형 Baseline(28.56%)과 거의 같고(29.74%), Valid 순위가 Test 순위와 맞지 않는다.

검증 설계: Batch 1을 **정책 단위**로 학습 29셀 / hold-out 7셀로 나눈다(같은 정책의 셀이 양쪽에 걸치지 않게). Train은 학습 구간의 GroupKFold(4) CV 평균, Valid는 hold-out, Test는 Batch 1 전체로 다시 학습한 모델의 Batch 2 성능이다.

## 성능 결과

최종 모델(세트 B / Random Forest), 노션 Reporting 형식:

| MAPE (%) | 비고                                      | 구분                     |
| -------- | ----------------------------------------- | ------------------------ |
| 11.75    |                                           | Train (Batch 1 CV)       |
| 5.86     |                                           | Valid (Batch 1 Hold-out) |
| 29.74    |                                           | Test (Batch 2)           |
| −5.89    | (+) : 과적합 의심 → 해당 없음             | Gap (Train−Valid)        |
| +23.89   | (+) : 배치 간 일반화 저하 의심 → **해당** | Gap (Valid−Test)         |
| +20.64   | Target : 원논문 9.1%                      | Gap (Target−Test)        |

Batch 3 (추가):

| MAPE (%) | 비고                           | 구분                            |
| -------- | ------------------------------ | ------------------------------- |
| 18.47    |                                | Test (Batch 3)                  |
| +11.27   | Batch 2 대비 Batch 3가 더 낮음 | Gap (Batch 2−Batch 3)           |
| +9.37    | Target : 원논문 9.1%           | Gap (Target−Test), Batch 3 기준 |

전체 모델·feature 세트 비교는 [results/model_performance.csv](results/model_performance.csv)에 있다. 대표 값:

| 세트 / 모델              | Train(CV) | Valid | Test (B2) | Test (B3) |
| ------------------------ | --------- | ----- | --------- | --------- |
| A / Baseline             | 9.14      | 7.80  | 28.56     | 12.81     |
| A / GPR                  | 9.71      | 8.25  | 28.41     | 12.59     |
| B / Random Forest (최종) | 11.75     | 5.86  | 29.74     | 18.47     |

## 오류 분석

- **가장 크게 틀린 셀**: Batch 2 legacy 구조의 **가장 짧은 셀들**이다(`b2_c06`, `b2_c15` 정답 393·396 사이클을 약 600~640으로 예측, 오차 53~62%). 최종 모델의 Batch 2 MAPE는 legacy 34.8%, newstructure 12.8%이다.
- **공통점**: 정답이 학습 Batch 1의 최솟값(534) 미만인 셀들이다. Batch 2 정답 39개 중 30개가 여기에 속한다.
- 원인 가설(검증 전):
  1. 학습 정답 범위(534~1,074) 밖이라 외삽이 안 된다(트리 계열은 특히).
  2. 같은 ΔQ 분산에서도 Batch 2 legacy는 Batch 1 추세보다 수명이 짧다. 선형 Baseline도 Batch 2 legacy 수명을 평균 1.32배로 과대 예측한다. 같은 충전 정책에서도 Batch 2 legacy가 Batch 1보다 35~45% 짧은 것과 같은 현상으로, Batch 간 기준 차이(셀 제조·수집 시기 등)가 의심된다.
- 개선 방향: Batch 효과 보정, 반복 정책 단위 분할로 Valid 안정화(hold-out 확대), 외삽에 강한 모델의 선택 규칙 재검토

## ESS 도메인 해석

- **활용**: 초기 100사이클의 전압별 곡선 변화(ΔQ)로 후반 열화를 일찍 감지할 수 있다면, 셀 교체와 점검의 우선순위를 사전에 정하는 데 쓸 수 있다. 같은 충전 시간이라도 단계 간 전류를 균일하게 가져가는 편이 수명에 유리한 경향이 있어 급속충전 운영 정책 설계의 참고가 된다(상관 기반이며 인과는 아니다).
- **한계**: 현재 Batch 2 성능은 원논문보다 크게 낮고(Gap +20.6%p), 특히 학습 범위보다 짧은 수명의 셀에서 크게 틀린다. 즉 **새로운 Batch·구조의 셀에는 그대로 적용하기 어렵다.** 가장 긴 수명 구간(EOL 미도달 셀)은 검증하지 못했고, 학습 셀이 36개뿐이다.
- **실 배포에 필요한 것**: 대상 셀 묶음의 일부로 재보정(Batch 효과 보정), 더 많은 셀로 학습·검증, 수명 범위가 다른 셀에 대한 별도 검증.

## 참고문헌

- Severson et al. (2019). Data-driven prediction of battery cycle life before capacity degradation. _Nature Energy_, 4, 383–391.

## 팀 구성

- 이준영 : EDA, 피처 엔지니어링, 모델 개발, 성능 평가
