"""원본 .mat(Batch 1~3)에서 필요한 값만 읽어 셀 단위 표(data/processed/cells.pkl)로 저장한다.

저장하는 값: 정책, cycle_life, 방전 용량(QD) 시계열, 초기 100사이클 평균 온도,
ΔQ(V) = Q100(V) - Q10(V), 마지막 QD(EOL 도달 판정용). 실행: python src/preprocess.py

원본 .mat 3개는 data/ 폴더에 그대로 넣어 둔다(경로를 바꾸려면 환경 변수 ESS_DATA_DIR).
"""
import os
import numpy as np
import pandas as pd
import h5py

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
DATA_DIR = os.environ.get('ESS_DATA_DIR', os.path.join(ROOT, 'data'))   # 원본 .mat 파일이 있는 폴더 (기본: data/)
OUT_DIR = os.path.join(ROOT, 'data', 'processed')
FILES = {'b1': '2017-05-12_batchdata_updated_struct_errorcorrect.mat',
         'b2': '2018-02-20_batchdata_updated_struct_errorcorrect.mat',
         'b3': '2018-04-12_batchdata_updated_struct_errorcorrect.mat'}
EOL_QD = 0.885      # 마지막 QD가 이 값 이하면 EOL(공칭 1.1Ah의 80% ≈ 0.88Ah) 도달로 본다


def _text(f, ref):
    return ''.join(chr(int(c)) for c in np.ravel(f[ref][()]))


def load_batch(bid, fname):
    rows = []
    with h5py.File(os.path.join(DATA_DIR, fname), 'r') as f:
        b = f['batch']
        v_grid = np.ravel(f[b['Vdlin'][0, 0]][()])
        
        for i in range(b['summary'].shape[0]):
            s = f[b['summary'][i, 0]]
            qd, tavg = np.ravel(s['QDischarge'][()]), np.ravel(s['Tavg'][()])
            life = np.ravel(f[b['cycle_life'][i, 0]][()])
            qdlin = f[b['cycles'][i, 0]]['Qdlin']                  # (사이클 수, 1) 참조 배열
            dq = np.full(len(v_grid), np.nan)

            if qdlin.shape[0] >= 100:                              # 사이클 n → 행 n-1
                dq = np.ravel(f[qdlin[99, 0]][()]) - np.ravel(f[qdlin[9, 0]][()])
            t = tavg[1:100]

            rows.append(dict(uid=f'{bid}_c{i:02d}', batch=bid, policy=_text(f, b['policy_readable'][i, 0]),
                             cycle_life=float(life[0]) if life.size else np.nan,
                             qd=qd, qd_last=qd[qd > 0][-1], tavg_mean=t[t > 0].mean(), dq=dq))
    return rows, v_grid

def main():
    missing = [f for f in FILES.values() if not os.path.exists(os.path.join(DATA_DIR, f))]

    if missing:
        raise SystemExit(f'원본 데이터가 없습니다. 아래 파일을 {os.path.abspath(DATA_DIR)} 에 넣어 주세요:\n  ' + '\n  '.join(missing))
    os.makedirs(OUT_DIR, exist_ok=True)

    rows = []
    for bid, fname in FILES.items():
        r, v_grid = load_batch(bid, fname)
        rows += r
        print(bid, len(r), 'cells')
    
    df = pd.DataFrame(rows)
    df['special'] = df.policy.str.contains('VarCharge|SLOWCYCLE', case=False)   # 별도 실험 셀
    df['reached_eol'] = df.qd_last <= EOL_QD
    df.to_pickle(os.path.join(OUT_DIR, 'cells.pkl'))
    np.save(os.path.join(OUT_DIR, 'v_grid.npy'), v_grid)

    print(df.groupby('batch').agg(cells=('uid', 'size'), special=('special', 'sum'), eol=('reached_eol', 'sum')))

if __name__ == '__main__':
    main()
