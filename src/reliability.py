"""3단계: 측정 신뢰도 — "재검사로 개인의 변화를 판정할 수 있는가?"

같은 DNA를 두 번 측정한 technical replicate 쌍으로
  ① 예측 나이의 ICC(2,1)  : 다시 재도 같은 값이 나오는 정도 (절대 일치, 단일 측정)
  ② SEM  = SD × √(1 − ICC) : 한 번 측정에 따라붙는 전형적 오차(년)
     MDC  = 2.77 × SEM      : 측정 오차만으로도 생길 수 있는 두 검사 간 차이의 상한(95%)
  ③ 마커별 ICC              : 어느 마커가 노이즈를 많이 만드는지 (|계수| × (1 − ICC) 순)
  ④ OLS / Ridge / ElasticNet: 계수를 줄이는(수축) 방법이 재현성을 높이는지
를 본다. ※ technical replicate는 측정 노이즈의 '하한'이다. 같은 사람을 시간 간격을 두고
  다시 채취하면 오차는 이보다 커진다.

판단 기준(결과를 보기 전에 정함)
  ICC: 0.9 이상 우수 / 0.75 이상 양호 / 0.5 이상 보통
  기대 변화량(--effect, 년) < MDC 이면 개인 단위 단기 재검사 판정은 통계적으로 지지되지 않음

사용법
  합성 데이터:  python src/reliability.py --train blood_A --rep blood_rep
  공개 데이터:  python src/reliability.py --train GSE87571 --rep GSE55763_rep --effect 2
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS, choose_markers, load_dataset, ranked_labels

N_BOOT = 1000


def icc_2_1(x: np.ndarray) -> float:
    """x: (대상 n, 반복 k). 이원 무작위 효과·절대 일치·단일 측정 ICC(2,1)."""
    n, k = x.shape
    gm = x.mean()
    msr = k * np.sum((x.mean(1) - gm) ** 2) / (n - 1)
    msc = n * np.sum((x.mean(0) - gm) ** 2) / (k - 1)
    sse = np.sum((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2)
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def icc_ci(x, seed=0):
    rng = np.random.default_rng(seed)
    b = [icc_2_1(x[rng.integers(0, len(x), len(x))]) for _ in range(N_BOOT)]
    return np.percentile(b, [2.5, 97.5])


def wide(values: pd.Series, pheno: pd.DataFrame) -> np.ndarray:
    t = pd.DataFrame({"v": values, "subj": pheno["subject_id"], "rep": pheno["rep"]})
    return t.pivot_table(index="subj", columns="rep", values="v").dropna().values


def summarize(x):
    icc = icc_2_1(x)
    sem = x.std(ddof=1) * np.sqrt(max(0.0, 1 - icc))
    return icc, sem, 2.77 * sem


def band(icc):
    return "우수" if icc >= .9 else "양호" if icc >= .75 else "보통" if icc >= .5 else "낮음"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="blood_A")
    ap.add_argument("--rep", default="blood_rep")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--panel", default=None, help="비공개 최종 패널 파일(예: configs/panel.local.txt)")
    ap.add_argument("--effect", type=float, default=2.0, help="비교할 기대 변화량(년), 문헌 근거로 지정")
    a = ap.parse_args()

    markers = choose_markers(a.top, a.panel)
    btr, ptr = load_dataset(a.train, cpgs=markers)
    ptr = ptr.dropna(subset=["age"])
    Xtr, ytr = btr[ptr.index].loc[markers].T.values, ptr["age"].values
    brep, prep = load_dataset(a.rep, cpgs=markers)
    Xrep = brep.loc[markers].T.values
    print(f"replicate 쌍: {prep['subject_id'].nunique()}쌍")

    # ① ② clock-level
    ols = LinearRegression().fit(Xtr, ytr)
    x = wide(pd.Series(ols.predict(Xrep), index=brep.columns), prep)
    icc, sem, mdc = summarize(x)
    lo, hi = icc_ci(x)
    verdict = "개인 판정 가능" if a.effect >= mdc else "개인 단기 판정 지지 불가"
    summary = pd.DataFrame([dict(n_pairs=len(x), ICC=icc, ICC_CI_low=lo, ICC_CI_high=hi, 등급=band(icc),
                                 SEM_years=sem, MDC95_years=mdc, 기대변화_years=a.effect, 판정=verdict)])
    print("\n== ① ② 예측 나이 재현성 ==\n" + summary.round(3).to_string(index=False))
    print(f"해석: 같은 DNA를 두 번 재도 결과가 최대 약 {mdc:.1f}년 달라질 수 있음(95%). "
          f"기대 변화 {a.effect}년 → {verdict}.")

    # ③ marker-level
    rows = []
    for lab, m, w in zip(ranked_labels(len(markers)), markers, ols.coef_):
        mi = icc_2_1(wide(brep.loc[m], prep))
        rows.append(dict(marker=lab, ICC=mi, noise_contribution=abs(w) * (1 - mi)))
    per = pd.DataFrame(rows)
    per["noise_rank"] = per.noise_contribution.rank(ascending=False).astype(int)
    print("\n== ③ 마커별 재현성 (noise_rank 1 = 예측 노이즈 기여가 가장 큼) ==\n"
          + per.drop(columns="noise_contribution").round(3).to_string(index=False))

    # ④ fitting 방법별
    fitters = {"OLS": LinearRegression(),
               "Ridge": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 61))),
               "ElasticNet": make_pipeline(StandardScaler(),
                                           ElasticNetCV(l1_ratio=[.1, .5, .9, 1.0], cv=5, max_iter=20000))}
    rows = []
    for name, mdl in fitters.items():
        mdl.fit(Xtr, ytr)
        i2, s2, m2 = summarize(wide(pd.Series(mdl.predict(Xrep), index=brep.columns), prep))
        rows.append(dict(method=name, train_MAD=np.mean(np.abs(mdl.predict(Xtr) - ytr)), ICC=i2, MDC=m2))
    meth = pd.DataFrame(rows)
    print("\n== ④ 방법별 정확도 vs 재현성 ==\n" + meth.round(3).to_string(index=False))

    summary.round(4).to_csv(RESULTS / "reliability_summary.csv", index=False, encoding="utf-8-sig")
    per.drop(columns="noise_contribution").round(4).to_csv(RESULTS / "icc_per_marker.csv", index=False)
    meth.round(4).to_csv(RESULTS / "reliability_by_method.csv", index=False)


if __name__ == "__main__":
    main()
