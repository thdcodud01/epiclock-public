"""보조 분석: 나이 마커가 남녀에게 똑같이 작동하는가?

상품 관점의 질문: 성별에 따라 다른 마커·다른 해석이 필요한가,
               아니면 모든 고객에게 같은 기준을 쓸 수 있는가?

방법 (원 분석과 동일)
  - 후보 마커(marker_selection 결과)마다 남/여 따로 나이 상관 r_male, r_female 계산
  - r_diff = r_male − r_female (노화 속도의 성별차), mean_diff = 평균 메틸화의 성별차
  - 세 코호트에서 방향이 같고 크기도 기준 이상인지 = "일관된 성별차"
  - 코호트 간 r_diff의 상관이 낮으면 성별차는 재현되지 않는 잡음으로 판단

판단 기준(결과를 보기 전에 정함)
  일관된 성별차 = 모든 코호트에서 같은 부호 + |r_diff| >= 0.10
  해당 마커가 없으면 "남녀 공통으로 작동하는 마커"로 결론

사용법:  python src/sex_difference.py --datasets blood_A blood_B blood_C
"""
import argparse

import numpy as np
import pandas as pd

from common import RESULTS, load_dataset, pearson_rows, ranked_labels

TH_RDIFF = 0.10


def per_dataset(name, markers):
    b, p = load_dataset(name, cpgs=markers)
    p = p.dropna(subset=["age", "sex"])
    out = {}
    for sx in ("M", "F"):
        idx = p.index[p.sex == sx]
        out[sx] = pearson_rows(b.loc[markers, idx].values, p.loc[idx, "age"].values)
        out[f"mean_{sx}"] = b.loc[markers, idx].mean(axis=1).values
    return pd.DataFrame({f"rdiff_{name}": out["M"] - out["F"],
                         f"mdiff_{name}": out["mean_M"] - out["mean_F"]}, index=markers), \
        (p.sex == "M").sum(), (p.sex == "F").sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["blood_A", "blood_B", "blood_C"])
    a = ap.parse_args()
    markers = list(pd.read_csv(RESULTS / "selected_markers.csv", index_col=0).index)

    parts = []
    for n in a.datasets:
        d, nm, nf = per_dataset(n, markers)
        print(f"[{n}] 남 {nm} / 여 {nf}")
        parts.append(d)
    m = pd.concat(parts, axis=1)
    rd = m[[c for c in m if c.startswith("rdiff_")]]
    consistent = (np.sign(rd).nunique(axis=1) == 1) & (rd.abs().min(axis=1) >= TH_RDIFF)
    print(f"\n후보 {len(m)}개 중 일관된 노화 속도 성별차(|r_diff|>={TH_RDIFF}, 방향 일치): "
          f"{int(consistent.sum())}개")
    print("코호트 간 r_diff 상관 (높을수록 성별차가 재현됨):\n" + rd.corr().round(2).to_string())
    print("결론:", "남녀 공통으로 작동하는 마커 — 성별 분리 해석 불필요" if consistent.sum() == 0
          else "일부 마커에 일관된 성별차 — 성별 보정 검토 필요")
    m.index = ranked_labels(len(m))
    m.round(4).to_csv(RESULTS / "sex_difference.csv", encoding="utf-8-sig")


if __name__ == "__main__":
    main()
