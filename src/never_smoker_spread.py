"""4-1단계: 비흡연자에게도 줄 정보가 있는가? (간접흡연·환경노출 축 가능성)

질문: 비흡연자(never) 안에서도 흡연 마커 값이 개인별로 구분될 만큼 퍼져 있는가?
  퍼짐이 '실제 개인차'인지 '측정 잡음'인지를 가르기 위해 세 가지를 함께 본다.
  1) never 내부 퍼짐(나이·성별 보정 후 상·하위 10% 격차)이 흡연 효과의 몇 %인가
  2) never 안에서 마커들이 함께 움직이는가 (마커 간 평균 상관 vs 흡연 무관 대조 CpG)
     → 공통 원인(노출)이 있으면 함께 움직이고, 잡음이면 서로 무관하다  ★핵심★
  3) never를 패널 점수 상·하위 20%로 나눈 격차가 흡연 효과의 몇 %인가

판단 기준(결과를 보기 전에 정함)
  - 마커 간 평균 상관이 대조군보다 0.2 이상 높으면 → 공통 노출 요인 가능성(환경노출 축 검토)
  - 아니면 never 내부 변동은 대부분 잡음 → 비흡연자 개인 변별 불가
  - 3)의 비율이 30% 미만이면 소비자에게 의미 있는 변별을 주기 어려움

패널 선정 주의: 흡연 스캔 상위에는 같은 유전자 구역에 붙어 있는 CpG가 여럿 올라온다.
  바로 옆 CpG끼리는 노출과 무관하게 원래 함께 움직이므로, 그대로 쓰면 2)의 상관이 부풀려진다.
  --manifest 를 주면 서로 1Mb 이상 떨어진 위치에서만 패널을 고른다(원 분석은 문헌으로 확인한
  서로 다른 위치의 마커 5개를 사용). 매니페스트 없이 돌리면 이 점을 경고한다.

전제: smoking_axis.py 실행 후 (results/smoking_scan_<코호트>.csv)
사용법
  합성 데이터:  python src/never_smoker_spread.py --dataset smoke_A
  공개 데이터:  python src/never_smoker_spread.py --dataset GSE50660 --manifest ../../GPL13534-11288.txt
"""
import argparse

import numpy as np
import pandas as pd

from common import RESULTS, load_dataset, load_positions, pick_distinct_loci, ranked_labels, residualizer

N_CONTROL = 200


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="smoke_A")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--manifest", default=None, help="450K 매니페스트(서로 다른 위치에서 패널 선정)")
    ap.add_argument("--min-dist", type=float, default=1e6, help="패널 CpG 간 최소 거리(bp)")
    a = ap.parse_args()

    scan = pd.read_csv(RESULTS / f"smoking_scan_{a.dataset}.csv", index_col=0)
    if a.manifest:
        head = list(scan.index[:500])
        pos = load_positions(a.manifest, head)
        naive = head[: a.top]
        loci = {(pos.at[c, "CHR"], int(pos.at[c, "MAPINFO"] // a.min_dist)) for c in naive
                if c in pos.index and pd.notna(pos.at[c, "MAPINFO"])}
        print(f"p값 상위 {a.top}개의 위치 구역 수: {len(loci)}개 (같은 구역 CpG가 섞여 있으면 5보다 작음)")
        panel = pick_distinct_loci(head, pos, a.top, a.min_dist)
        print(f"→ 서로 {a.min_dist / 1e6:g}Mb 이상 떨어진 {len(panel)}개로 패널 구성 "
              f"(염색체: {[pos.at[c, 'CHR'] for c in panel]})")
    else:
        panel = list(scan.index[: a.top])
        print("⚠ --manifest 없이 p값 상위를 그대로 사용 — 인접 CpG가 섞이면 마커 간 상관이 부풀려질 수 있음")
    pool = list(scan.index[len(scan) // 2:])                      # 흡연과 무관한 쪽 절반
    ctrl = list(np.random.default_rng(0).choice(pool, min(N_CONTROL, len(pool)), replace=False))
    b, p = load_dataset(a.dataset, cpgs=panel + ctrl)
    p = p.dropna(subset=["age", "sex"])
    never, curr = p.index[p.smoking == "never"], p.index[p.smoking == "current"]

    R = residualizer(np.column_stack([p.loc[never, "age"], (p.loc[never, "sex"] == "M").astype(float)]))
    adj = pd.DataFrame(R @ b.loc[:, never].T.values.astype(float), index=never, columns=b.index)

    rows = []
    for lab, m in zip(ranked_labels(len(panel)), panel):
        signal = b.loc[m, never].mean() - b.loc[m, curr].mean()
        q10, q90 = adj[m].quantile([.1, .9])
        rows.append(dict(marker=lab, sd_never_adj=adj[m].std(), smoking_effect=signal,
                         p10_p90_gap_pct=(q90 - q10) / abs(signal) * 100))
    res = pd.DataFrame(rows).astype({"sd_never_adj": float, "smoking_effect": float, "p10_p90_gap_pct": float})

    def mean_offdiag(df):
        c = df.corr().values
        return np.nanmean(c[np.triu_indices_from(c, 1)])
    r_panel, r_ctrl = mean_offdiag(adj[panel]), mean_offdiag(adj[ctrl])

    z = (adj[panel] - adj[panel].mean()) / adj[panel].std()
    score = z.mean(axis=1)
    lo, hi = score.nsmallest(int(len(never) * .2)).index, score.nlargest(int(len(never) * .2)).index
    res["never_top_bottom20_gap_pct"] = [
        float((b.loc[m, hi].mean() - b.loc[m, lo].mean())
              / abs(b.loc[m, never].mean() - b.loc[m, curr].mean()) * 100)
        for m in panel]

    print(f"[{a.dataset}] never {len(never)}명 / current {len(curr)}명")
    print(res.round(3).to_string(index=False))
    print(f"\nnever 안에서 마커 간 평균 상관: {r_panel:+.3f}  |  대조 CpG {len(ctrl)}개: {r_ctrl:+.3f}"
          f"  → 차이 {r_panel - r_ctrl:+.3f}")
    common_factor = (r_panel - r_ctrl) >= 0.2
    big_gap = res["never_top_bottom20_gap_pct"].mean() >= 30
    print("결론:", "공통 노출 요인 가능성 → 환경노출 축 검토" if common_factor and big_gap else
          "never 내부 변동은 대부분 잡음이거나 격차가 작음 → 비흡연자 개인 변별 어려움")
    res.round(4).to_csv(RESULTS / "never_smoker_spread.csv", index=False)


if __name__ == "__main__":
    main()
