# data

원본 데이터는 용량이 커서(약 7GB) 저장소에 포함하지 않는다.

## 원본 (Batch 1~3)

- 출처: MIT-Stanford Battery Dataset (Severson et al., *Nature Energy*, 2019)
- 다운로드: https://www.kaggle.com/datasets/itshpark/data-driven-prediction-of-battery-cycle
- 사용 파일 (MATLAB v7.3 / HDF5):

| Batch | 파일 | 용도 |
|---|---|---|
| Batch 1 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 학습·검증 |
| Batch 2 | `2018-02-20_batchdata_updated_struct_errorcorrect.mat` | 테스트 |
| Batch 3 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 추가 검증(선택) |

`2018-04-03_varcharge_...mat`는 사용하지 않는다.

## 경로 설정

원본 `.mat` 3개를 **이 폴더(`data/`)에 그대로** 넣는다. 파일이 없으면 `preprocess.py`가 필요한 파일 이름을 알려 준다.

```
data/
├── README.md
├── 2017-05-12_batchdata_updated_struct_errorcorrect.mat
├── 2018-02-20_batchdata_updated_struct_errorcorrect.mat
├── 2018-04-12_batchdata_updated_struct_errorcorrect.mat
└── processed/        ← preprocess.py가 생성
```

다른 폴더에 두려면 환경 변수로 지정할 수 있다.

```bash
export ESS_DATA_DIR=/path/to/mat_files
python src/preprocess.py
```

## 생성되는 파일 (`data/processed/`, 저장소에는 올리지 않음)

| 파일 | 내용 |
|---|---|
| `cells.pkl` | 셀 단위 표: 정책, `cycle_life`, 방전 용량(QD) 시계열, 초기 100사이클 평균 온도, ΔQ(V) = Q100(V) − Q10(V), EOL 도달 여부 |
| `v_grid.npy` | ΔQ(V)의 전압 격자 (1,000점, 3.5 → 2.0V) |

## 분석 대상 셀

| Batch | 전체 | 별도 실험(제외) | EOL 도달(모델링 대상) | EOL 미도달(제외) |
|---|---|---|---|---|
| Batch 1 | 46 | 0 | 36 | 10 |
| Batch 2 | 47 | 8 | 39 | 0 |
| Batch 3 | 46 | 0 | 44 | 2 |

- 별도 실험: 정책명에 `VarCharge`, `SLOWCYCLE`이 들어간 셀(`cycle_life` 정의 없음)
- EOL 미도달: 마지막 방전 용량이 0.885Ah 이하가 아닌 셀. 수명이 "기록 길이 이상"이라는 정보만 있어 학습·평가에서 제외
