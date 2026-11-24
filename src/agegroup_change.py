"""보조 분석: "특정 연령대(예: 50대)에서 메틸화가 급격히 변한다"는 가설 검증.

방법 (원 분석과 동일)
  - 30~74세를 5세 구간 9개로 나눔 (세 코호트에 공통으로 있는 범위)
  - CpG마다 구간별 평균 beta → 인접 구간 사이 변화량(8개 전환)
  - ① 전환별 평균 |변화량| (전체 CpG)  ② CpG마다 변화가 가장 큰 전환이 어디인지의 분포
  - 코호트마다 반복해 같은 전환이 일관되게 튀는지 확인

판단 기준(결과를 보기 전에 정함)
  - 여러 코호트에서 같은 전환이 최대로 나와야 "특정 연령대 변화"로 인정
  - 표본 10명 미만 구간이 끼면 '불안정'으로 표시하고 결론에서 제외

사용법
  합성 데이터:  python src/agegroup_change.py
  공개 데이터:  python src/agegroup_change.py --datasets GSE40279 GSE87571 GSE55763
"""
import argparse

import numpy as np
import pandas as pd

from common import RESULTS, iter_beta, load_pheno

LABELS = [f"{e}-{e + 4}" for e in range(30, 75, 5)]
TRANS = [f"{LABELS[i]}→{LABELS[i + 1]}" for i in range(len(LABELS) - 1)]
MIN_N = 10


def run(name):
    pheno = load_pheno(name).set_index("sample_id")
    band = pd.cut(pheno["age"], bins=list(range(30, 80, 5)), labels=LABELS, right=False)
    counts = band.value_counts().reindex(LABELS).fillna(0).astype(int)
    abs_sum, n_cpg, argmax = np.zeros(len(TRANS)), np.zeros(len(TRANS)), np.zeros(len(TRANS), dtype=int)
    for chunk in iter_beta(name):
        b = band.reindex(chunk.columns)
        ok = b.notna().values
        means = chunk.loc[:, ok].T.groupby(b[ok].values, observed=False).mean().T.reindex(columns=LABELS)
        steps = np.abs(np.diff(means.values, axis=1))          # 비어 있는 구간은 NaN
        abs_sum += np.nansum(steps, axis=0)
        n_cpg += (~np.isnan(steps)).sum(axis=0)
        valid = ~np.isnan(steps).all(axis=1)
        argmax += np.bincount(np.nanargmax(steps[valid], axis=1), minlength=len(TRANS))
    stable = [(counts.iloc[i] >= MIN_N) and (counts.iloc[i + 1] >= MIN_N) for i in range(len(TRANS))]
    return pd.DataFrame({"dataset": name, "전환": TRANS, "평균_abs_변화량": np.where(n_cpg > 0, abs_sum / np.maximum(n_cpg, 1), np.nan),
                         "최대전환_CpG수": argmax, "앞구간_n": counts.values[:-1],
                         "뒤구간_n": counts.values[1:], "안정": stable})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["blood_A", "blood_B", "blood_C"])
    a = ap.parse_args()

    df = pd.concat([run(n) for n in a.datasets], ignore_index=True)
    df.to_csv(RESULTS / "agegroup_change.csv", index=False, encoding="utf-8-sig")
    print(df.pivot(index="전환", columns="dataset", values="평균_abs_변화량").reindex(TRANS)
          .round(5).to_string())

    peaks = {}
    for n, g in df.groupby("dataset"):
        st = g[g["안정"]]
        peaks[n] = st.loc[st["평균_abs_변화량"].idxmax(), "전환"] if len(st) else None
    print("\n코호트별 변화가 가장 큰 전환(안정 구간만):", peaks)
    print("결론:", "같은 전환이 모든 코호트에서 최대 → 특정 연령대 변화 재현"
          if len(set(peaks.values())) == 1 else
          "최대 전환이 코호트마다 다름 → 특정 연령대 집중 변화는 이 조건에서 재현되지 않음")


if __name__ == "__main__":
    main()
