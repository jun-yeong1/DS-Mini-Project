"""셀 표 → 모델 입력. DAY1 전략의 feature 정의와 세트(A/B/C)를 그대로 구현한다."""
import os
import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
CELLS_PKL = os.path.join(ROOT, 'data', 'processed', 'cells.pkl')
POLICY = r'^([\d.]+)C\((\d+)%\)-([\d.]+)C'          # 예: 5.2C(58%)-4C → C1, SOC1(%), C2

FEATURE_SETS = {
    'A': ['dq_logvar'],                                         # ΔQ 단독
    'B': ['dq_logvar', 'contrast', 'avg_c', 'soc1'],            # + 충전 정책
    'C': ['dq_logvar', 'contrast', 'avg_c', 'soc1', 'tavg_mean'],   # + 온도
}


def load_cells():
    return pd.read_pickle(CELLS_PKL)


def build_features(df):
    df = df.copy()
    df[['c1', 'soc1', 'c2']] = df.policy.str.extract(POLICY).astype(float)
    t80 = 60 * (df.soc1 / 100 / df.c1 + (80 - df.soc1) / 100 / df.c2)     # 0→80% 충전 시간(분)
    df['avg_c'] = 48 / t80                                                # 0→80% 평균 C-rate
    df['contrast'] = (df.c1 - df.c2).abs()                                # 단계 간 전류 차이
    df['dq_logvar'] = [np.log10(np.nanvar(d)) for d in df.dq]             # ΔQ(V) 분산의 로그
    df['newstruct'] = df.policy.str.contains('newstructure')
    df['y'] = np.log10(df.cycle_life)                                     # Target: log10(cycle_life)
    return df


def model_cells():
    """모델링 대상: 별도 실험 제외, EOL 도달 셀(정답 확정), ΔQ 계산 가능한 셀."""
    df = build_features(load_cells())
    return df[~df.special & df.reached_eol & df.dq_logvar.notna()].reset_index(drop=True)
