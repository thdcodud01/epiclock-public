"""5단계: 검체 확장 — 혈액에서 고른 마커로 타액 생체나이 모델을 만들 수 있는가?

질문
  Q1. 같은 사람의 혈액과 타액에서 마커 값은 얼마나 다른가? (혈액 모델을 그대로 쓸 수 있는가)
  Q2. 타액 데이터셋 안에서 교차검증하면 기준을 넘는가?
  Q3. 한 집단의 타액으로 학습한 모델이 다른 집단의 타액에서도 나이 순서를 맞추는가?
  Q4. 계수는 안정적인가? (부트스트랩에서 부호가 유지되는 비율)

판단 기준(결과를 보기 전에 정함, 원 분석과 동일)
  데이터셋 내 교차검증에서 ① r >= 0.80  ② MAD <= 6.0년
  ③ 평균 나이로 찍는 것보다 MAD가 50% 이상 작을 것 → 셋 다 충족 시 '설계·검증 가능'

사용법
  합성 데이터:  python src/saliva_transfer.py
  공개 데이터:  python src/saliva_transfer.py --a GSE78874 --b GSE92767 --paired GSE130153
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, LeaveOneOut, cross_val_predict

from common import RESULTS, choose_markers, load_dataset, ranked_labels

CRIT_R, CRIT_MAD, CRIT_GAIN = 0.80, 6.0, 0.50
N_REPEATS, N_BOOT = 20, 500


def metrics(y, p):
    e = p - y
    return dict(r=float(np.corrcoef(y, p)[0, 1]), MAD=float(np.abs(e).mean()), bias=float(e.mean()))


def cv_within(X, y):
    reps = pd.DataFrame([metrics(y, cross_val_predict(LinearRegression(), X, y,
                                                       cv=KFold(5, shuffle=True, random_state=i)))
                         for i in range(N_REPEATS)])
    out = dict(r=reps.r.mean(), MAD=reps.MAD.mean(), MAD_min=reps.MAD.min(), MAD_max=reps.MAD.max())
    if len(y) < 100:
        m = metrics(y, cross_val_predict(LinearRegression(), X, y, cv=LeaveOneOut()))
        out.update(LOO_r=m["r"], LOO_MAD=m["MAD"])
    return out


def sign_stability(X, y):
    full = LinearRegression().fit(X, y).coef_
    rng = np.random.default_rng(0)
    same = np.zeros(X.shape[1])
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        same += np.sign(LinearRegression().fit(X[i], y[i]).coef_) == np.sign(full)
    return same / N_BOOT


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="saliva_A")
    ap.add_argument("--b", default="saliva_B")
    ap.add_argument("--paired", default="paired_bs", help="같은 사람 혈액·타액 쌍 데이터")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--panel", default=None, help="비공개 최종 패널 파일(예: configs/panel.local.txt)")
    a = ap.parse_args()
    markers = choose_markers(a.top, a.panel)
    labels = ranked_labels(len(markers))

    # Q1 혈액 vs 타액
    if a.paired:
        bp, pp = load_dataset(a.paired, cpgs=markers)
        rows = []
        for lab, m in zip(labels, markers):
            t = pd.DataFrame({"v": bp.loc[m], "subj": pp.subject_id, "tissue": pp.tissue})
            w = t.pivot_table(index="subj", columns="tissue", values="v").dropna()
            rows.append(dict(marker=lab, blood_mean=w["blood"].mean(), saliva_mean=w["saliva"].mean(),
                             saliva_minus_blood_pctp=(w["saliva"] - w["blood"]).mean() * 100,
                             between_person_corr=w["blood"].corr(w["saliva"])))
        q1 = pd.DataFrame(rows).set_index("marker").astype(float).reset_index()
        print(f"== Q1. 같은 사람 혈액 vs 타액 ({len(w)}쌍) ==\n" + q1.round(3).to_string(index=False))
        q1.round(4).to_csv(RESULTS / "saliva_vs_blood.csv", index=False)

    data = {}
    for n in (a.a, a.b):
        b, p = load_dataset(n, cpgs=markers)
        p = p.dropna(subset=["age"])
        data[n] = (b.loc[markers, p.index].T.values, p["age"].values.astype(float))

    # Q2 데이터셋 내 교차검증 + Q4 계수 안정성
    rows = []
    print(f"\n[사전 기준] r >= {CRIT_R}, MAD <= {CRIT_MAD}년, 평균 추측 대비 {CRIT_GAIN:.0%} 이상 개선")
    for n, (X, y) in data.items():
        naive = np.abs(y - y.mean()).mean()
        cv = cv_within(X, y)
        gain = 1 - cv["MAD"] / naive
        ok = cv["r"] >= CRIT_R and cv["MAD"] <= CRIT_MAD and gain >= CRIT_GAIN
        stab = sign_stability(X, y)
        rows.append(dict(구분="데이터셋 내 CV", 학습=n, 평가=n, n=len(y), r=cv["r"], MAD=cv["MAD"],
                         평균추측MAD=naive, 개선율=gain, 기준충족=ok, LOO_MAD=cv.get("LOO_MAD"),
                         계수부호_최저안정성=stab.min()))
        print(f"  {n}: n={len(y)} 나이 {y.min():.0f}~{y.max():.0f}세 | CV r={cv['r']:.2f} "
              f"MAD={cv['MAD']:.2f} (반복 {cv['MAD_min']:.2f}~{cv['MAD_max']:.2f}) 개선율 {gain:.0%} "
              f"→ {'충족' if ok else '미충족'} | 계수 부호 안정성 최저 {stab.min():.0%}")

    # Q3 집단 간 적용
    for tr, te in ((a.a, a.b), (a.b, a.a)):
        m = LinearRegression().fit(*data[tr])
        X, y = data[te]
        mt = metrics(y, m.predict(X))
        e = m.predict(X) - y
        rows.append(dict(구분="집단 간 적용", 학습=tr, 평가=te, n=len(y), r=mt["r"], MAD=mt["MAD"],
                         평균오차=mt["bias"], 평균오차제거MAD=np.abs(e - e.mean()).mean()))
        print(f"  {tr} → {te}: r={mt['r']:.2f} MAD={mt['MAD']:.2f} 평균오차 {mt['bias']:+.1f}"
              f" (평균오차 제거 시 MAD {np.abs(e - e.mean()).mean():.2f})")
    pd.DataFrame(rows).round(3).to_csv(RESULTS / "saliva_transfer.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
