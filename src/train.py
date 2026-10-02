"""Batch 1로 학습·검증하고 Batch 2(+선택: Batch 3)로 테스트한다. 실행: python src/train.py

- Train : Batch 1 학습 구간의 GroupKFold(정책 단위) CV 평균 MAPE
- Valid : Batch 1 hold-out(정책 단위로 분리) MAPE
- Test  : Batch 1 전체로 다시 학습한 모델의 Batch 2 MAPE (Batch 3는 추가 확인용)
모델 선택은 Valid MAPE로만 한다. Test는 보고용이다.
"""
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, ElasticNetCV, RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.ensemble import RandomForestRegressor

from sklearn.exceptions import ConvergenceWarning
from features import FEATURE_SETS, model_cells

warnings.filterwarnings('ignore', category=ConvergenceWarning)   # GPR 노이즈 항이 하한에 닿는 경고(결과에 영향 없음)

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
RESULTS = os.path.join(ROOT, 'results')
TARGET_MAPE = 9.1       # 원논문 회귀 성능 (MAPE, %)
SEED = 0

MODELS = {              # 이름 → 새 모델을 만드는 함수 (DAY1 후보 모델)
    'Baseline': lambda: LinearRegression(),
    'ElasticNet': lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8], cv=3)),
    'Ridge': lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 2, 20))),
    'PLS': lambda: make_pipeline(StandardScaler(), PLSRegression(n_components=1)),
    'GPR': lambda: make_pipeline(StandardScaler(), GaussianProcessRegressor(
        DotProduct() + RBF() + WhiteKernel(), normalize_y=True, random_state=SEED)),
    'RandomForest': lambda: RandomForestRegressor(300, min_samples_leaf=2, random_state=SEED),
}


def mape(y_log, pred_log):
    """log10 값을 원래 수명 단위로 되돌려 MAPE(%)를 계산한다."""
    y, p = 10 ** np.asarray(y_log), 10 ** np.ravel(pred_log)
    return float(np.mean(np.abs(y - p) / y) * 100)


def run():
    cells = model_cells()
    b1, b2, b3 = (cells[cells.batch == b] for b in ('b1', 'b2', 'b3'))
    # Batch 1을 정책 단위로 학습 80% / hold-out 20%로 나눈다 (같은 정책 셀이 양쪽에 걸치지 않게)
    tr_idx, va_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=SEED).split(b1, groups=b1.policy))
    tr, va = b1.iloc[tr_idx], b1.iloc[va_idx]
    print(f'Batch 1 학습 {len(tr)}셀 / hold-out {len(va)}셀 | Batch 2 {len(b2)}셀 | Batch 3 {len(b3)}셀')

    rows = []
    for fs_name, feats in FEATURE_SETS.items():
        for name, make in MODELS.items():
            if name == 'Baseline' and fs_name != 'A':
                continue                                   # Baseline은 dq_logvar 단일 회귀(세트 A)만
            X = lambda d: d[feats].values
            cv_pred = cross_val_predict(make(), X(tr), tr.y, groups=tr.policy, cv=GroupKFold(4))
            valid = make().fit(X(tr), tr.y).predict(X(va))
            final = make().fit(X(b1), b1.y)
            row = dict(feature_set=fs_name, model=name,
                       mape_train_cv=mape(tr.y, cv_pred), mape_valid=mape(va.y, valid),
                       mape_test_b2=mape(b2.y, final.predict(X(b2))),
                       mape_test_b3=mape(b3.y, final.predict(X(b3))))
            row['gap_train_valid'] = row['mape_valid'] - row['mape_train_cv']      # (+) 과적합 의심
            row['gap_valid_test'] = row['mape_test_b2'] - row['mape_valid']        # (+) 배치 간 일반화 저하 의심
            row['gap_target_test'] = row['mape_test_b2'] - TARGET_MAPE             # 원논문(9.1%) 대비
            rows.append(row)

    res = pd.DataFrame(rows).round(2)
    best = res.loc[res.mape_valid.idxmin()]                                       # 선택 기준: Valid MAPE
    res['final'] = (res.feature_set == best.feature_set) & (res.model == best.model)
    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(os.path.join(RESULTS, 'model_performance.csv'), index=False)

    # 최종 모델의 Batch 2 셀별 예측 (오류 분석용)
    feats = FEATURE_SETS[best.feature_set]
    final = MODELS[best.model]().fit(b1[feats].values, b1.y)
    pred = b2.assign(life_pred=10 ** np.ravel(final.predict(b2[feats].values)))
    pred['ape_pct'] = (pred.life_pred - pred.cycle_life).abs() / pred.cycle_life * 100
    pred['group'] = np.where(pred.newstruct, 'b2-new', 'b2-legacy')
    pred[['uid', 'policy', 'group', 'cycle_life', 'life_pred', 'ape_pct']].round(1).to_csv(
        os.path.join(RESULTS, 'predictions_b2.csv'), index=False)
    return res, best


if __name__ == '__main__':
    res, best = run()
    print(res.to_string(index=False))
    print(f'\n최종 모델(Valid MAPE 최소): 세트 {best.feature_set} / {best.model}')
