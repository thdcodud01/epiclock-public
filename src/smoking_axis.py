"""4단계: 생활습관 축 — 흡연 이력이 소수 마커로 신뢰성 있게 포착되는가?

질문
  Q1. 나이·성별을 보정하고도 흡연과 연관된 CpG가 있는가? (discovery, 전체 CpG 스캔)
  Q2. 그 신호가 다른 코호트에서도 같은 방향으로 재현되는가? (replication)
      재현 코호트에 질병 상태가 섞여 있으면 질병도 보정하고, 대조군만으로도 다시 본다.
  Q3. 몇 개의 마커면 흡연 이력(ever vs never)을 구분할 수 있는가? (교차검증 AUC)
  Q4. 전 흡연자는 비흡연자와 현재 흡연자 사이 어디쯤 있는가? (위치 %)
      ※ 단면 자료이므로 "금연하면 회복된다"(인과)가 아니라
        "전 흡연자는 중간 지점에 위치한다"(관찰)로만 해석한다.

판단 기준(결과를 보기 전에 정함)
  - discovery: BH-FDR < 0.05
  - replication: 상위 마커가 같은 방향 + 재현 코호트에서 p < 0.05
  - 실용성: 5개 이하 마커로 교차검증 AUC >= 0.8

사용법
  합성 데이터:  python src/smoking_axis.py
  공개 데이터:  python src/smoking_axis.py --discovery GSE50660 --replication GSE42861
"""
import argparse
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict

from common import (RESULTS, figure_dir, bh_fdr, iter_beta, korean_font, load_dataset, load_pheno,
                    ranked_labels, residualizer)

warnings.filterwarnings("ignore", category=FutureWarning)
ORD = {"never": 0, "former": 1, "current": 2}
NS = [1, 2, 3, 5, 10, 20, 30]


def covars(p):
    cols = [p["age"].values, (p["sex"] == "M").astype(float).values]
    if "disease" in p and p["disease"].nunique() > 1:
        cols.append(p["disease"].values.astype(float))
    return np.column_stack(cols)


def clean_pheno(name):
    p = load_pheno(name).set_index("sample_id")
    return p[p["smoking"].isin(ORD)].dropna(subset=["age", "sex"])


def scan(name):
    """전체 CpG에 대해 나이·성별(·질병) 보정 편상관(흡연 0/1/2)을 조각 단위로 계산."""
    p = clean_pheno(name)
    R = residualizer(covars(p))
    y = R @ p["smoking"].map(ORD).values.astype(float)
    y = (y - y.mean()) / y.std()
    n, k = len(p), covars(p).shape[1]
    parts = []
    for chunk in iter_beta(name):
        X = chunk[p.index].dropna()
        B = R @ X.values.T.astype(np.float64)
        B -= B.mean(0)
        sd = B.std(0)
        sd[sd == 0] = np.nan
        parts.append(pd.Series((B / sd * y[:, None]).sum(0) / n, index=X.index))
    r = pd.concat(parts)
    dfree = n - 2 - k
    t = r * np.sqrt(dfree / (1 - r ** 2))
    pv = 2 * stats.t.sf(np.abs(t), dfree)
    res = pd.DataFrame({"partial_r": r, "p": pv, "fdr": bh_fdr(pv)}).sort_values("p")
    return res, p


def adjusted(beta, p, markers):
    """공변량 효과를 뺀 값(+전체 평균)으로 그룹 비교."""
    R = residualizer(covars(p))
    X = beta.loc[markers, p.index].values.T.astype(float)
    return pd.DataFrame(R @ X + X.mean(0), index=p.index, columns=markers)


def recovery_table(M, p):
    rows = []
    for lab, m in zip(ranked_labels(M.shape[1]), M.columns):
        g = {s: M.loc[p.smoking == s, m] for s in ORD}
        mn = {s: v.mean() for s, v in g.items()}
        rows.append(dict(marker=lab, never=mn["never"], former=mn["former"], current=mn["current"],
                         former_position_pct=(mn["former"] - mn["current"]) / (mn["never"] - mn["current"]) * 100,
                         p_never_vs_current=stats.ttest_ind(g["never"], g["current"], equal_var=False).pvalue,
                         p_former_vs_never=stats.ttest_ind(g["former"], g["never"], equal_var=False).pvalue))
    return pd.DataFrame(rows)


def auc_by_n(beta, p, ranked):
    y = (p["smoking"] != "never").astype(int).values
    out = []
    for k in [k for k in NS if k <= len(ranked)]:
        X = beta.loc[ranked[:k], p.index].T.values
        prob = cross_val_predict(LogisticRegressionCV(cv=5, max_iter=1000), X, y, cv=5,
                                 method="predict_proba")[:, 1]
        out.append(dict(n_markers=k, AUC=roc_auc_score(y, prob)))
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--discovery", default="smoke_A")
    ap.add_argument("--replication", default="smoke_B")
    ap.add_argument("--top", type=int, default=5)
    a = ap.parse_args()

    disc, pd_ = scan(a.discovery)
    disc.to_csv(RESULTS / f"smoking_scan_{a.discovery}.csv", encoding="utf-8-sig")
    sig = disc[disc.fdr < 0.05]
    top = list(sig.index[: a.top])
    print(f"[Q1 discovery {a.discovery}] {len(pd_)}명, FDR<0.05: {len(sig):,}개 CpG")

    # Q2 replication
    br, pr = load_dataset(a.replication, cpgs=top)
    pr = pr[pr["smoking"].isin(ORD)].dropna(subset=["age", "sex"])
    R = residualizer(covars(pr))
    yr = R @ pr["smoking"].map(ORD).values.astype(float)
    rep = []
    for lab, m in zip(ranked_labels(len(top)), top):
        r_, p_ = stats.pearsonr(R @ br.loc[m, pr.index].values.astype(float), yr)
        rep.append(dict(marker=lab, r_discovery=disc.loc[m, "partial_r"], r_replication=r_, p_replication=p_))
    rep = pd.DataFrame(rep)
    rep["replicated"] = (np.sign(rep.r_discovery) == np.sign(rep.r_replication)) & (rep.p_replication < .05)
    print(f"\n[Q2 replication {a.replication}] 공변량: 나이·성별"
          + ("·질병" if "disease" in pr and pr.disease.nunique() > 1 else ""))
    print(rep.round(4).to_string(index=False))

    # Q3 AUC
    bd, _ = load_dataset(a.discovery, cpgs=list(sig.index[: max(NS)]))
    auc = auc_by_n(bd, pd_, list(sig.index[: max(NS)]))
    print(f"\n[Q3 소수 마커 판별력 — {a.discovery}, 5겹 교차검증]\n" + auc.round(3).to_string(index=False))

    # Q4 위치
    rec = recovery_table(adjusted(bd, pd_, top), pd_)
    rec_r = recovery_table(adjusted(br, pr, top), pr)
    print(f"\n[Q4 전 흡연자 위치 — {a.discovery}] 평균 {rec.former_position_pct.mean():.0f}% "
          f"(비흡연자 쪽으로, 관찰 결과)")
    print(rec.round(4).to_string(index=False))
    print(f"[Q4 재현 — {a.replication}] 평균 {rec_r.former_position_pct.mean():.0f}%")
    if "disease" in pr and pr.disease.nunique() > 1:
        pc = pr[pr.disease == 0]
        rec_c = recovery_table(adjusted(br, pc, top), pc)
        print(f"  └ 대조군만({len(pc)}명): 평균 {rec_c.former_position_pct.mean():.0f}% (질병 영향 점검)")

    rep.to_csv(RESULTS / "smoking_replication.csv", index=False)
    auc.to_csv(RESULTS / "smoking_auc_by_n.csv", index=False)
    rec.assign(dataset=a.discovery).pipe(lambda d: pd.concat([d, rec_r.assign(dataset=a.replication)])) \
       .to_csv(RESULTS / "smoking_former_position.csv", index=False)

    korean_font()
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.plot(auc.n_markers, auc.AUC, "o-", color="#1f6f78")
    ax.axhline(0.8, color="grey", lw=0.8, ls=":")
    ax.set_xlabel("마커 수")
    ax.set_ylabel("교차검증 AUC (ever vs never)")
    ax.set_title("마커 수에 따른 흡연 이력 판별력")
    fig.tight_layout()
    fig.savefig(figure_dir(a.discovery, a.replication) / "smoking_auc_by_n.png", dpi=130)


if __name__ == "__main__":
    main()
