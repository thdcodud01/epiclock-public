"""2단계: 생체나이 예측 모델 — 코호트 간 교차검증, 마커 수·방법 비교, 보정 효과.

질문
  Q1. 소수 마커(상위 N개, 다중선형회귀)로 다른 코호트에서도 나이를 맞출 수 있는가?
  Q2. 후보 전체 + ElasticNet과 비교하면 소수 마커는 얼마나 손해인가?
  Q3. ElasticNet이 중요하다고 본 상위 N개만 떼어 쓰면 어떤가? (마커 중요도는 조합에 따라 달라지는가)
  Q4. 코호트별 측정 기준점 차이를 평행이동(shift)·선형 보정으로 줄일 수 있는가?

설계 (원 분석과 동일)
  - 학습 코호트 1개로 학습 → 나머지 코호트 각각에서 평가 (같은 코호트 안 분할보다 엄격)
  - 소수 마커는 평균 |r| 상위에서 유전자당 1개 (유전자 정보가 있을 때. 미지정 CpG 제외)
  - 보정값은 평가 코호트를 50:50으로 나눠 절반에서 추정, 나머지 절반에서만 성능 평가
  - 공개용: 마커 이름과 회귀 계수는 출력하지 않는다(로컬 확인은 --show-coef)

출력: results/crossval.csv, results/calibration.csv, figures/pred_vs_age.png

사용법
  합성 데이터:  python src/clock_model.py
  공개 데이터:  python src/clock_model.py --datasets GSE40279 GSE87571 GSE55763 --final-train GSE87571
"""
import argparse
import itertools

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import RESULTS, ROOT, choose_markers, figure_dir, korean_font, load_dataset, mad, r2

L1 = [.1, .5, .7, .9, .95, 1.0]


def enet():
    major, minor = (int(x) for x in sklearn.__version__.split(".")[:2])
    grid = {"alphas": 50} if (major, minor) >= (1, 7) else {"n_alphas": 50}
    return make_pipeline(StandardScaler(),
                         ElasticNetCV(l1_ratio=L1, cv=5, max_iter=5000, random_state=0, **grid))


def Xy(data, name, markers):
    b, p = data[name]
    return b.loc[markers].T.values, p["age"].values.astype(float)


def calibrate(y, pred, seed=42):
    idx = np.arange(len(y))
    cal, ev = train_test_split(idx, test_size=0.5, random_state=seed)
    shift = float((y[cal] - pred[cal]).mean())
    lin = LinearRegression().fit(pred[cal].reshape(-1, 1), y[cal])
    out = {"보정 없음": pred[ev], "평행이동": pred[ev] + shift,
           "선형 보정": lin.predict(pred[ev].reshape(-1, 1))}
    return y[ev], out, shift


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["blood_A", "blood_B", "blood_C"])
    ap.add_argument("--final-train", default=None, help="최종 모델 학습 코호트 (기본: 첫 번째)")
    ap.add_argument("--top", type=int, default=8)
    ap.add_argument("--panel", default=None,
                    help="비공개 최종 패널 파일(예: configs/panel.local.txt). 없으면 공개 선별 상위 --top 개")
    ap.add_argument("--show-coef", action="store_true")
    a = ap.parse_args()
    final_train = a.final_train or a.datasets[0]

    sel = pd.read_csv(RESULTS / "selected_markers.csv", index_col=0)
    cand = list(sel.index)
    top = choose_markers(a.top, a.panel)
    n_top = len(top)
    OLS = f"소수 마커 {n_top}개 OLS"
    print(f"후보 {len(cand)}개 / 소수 마커 {n_top}개" + (" (최종 패널)" if a.panel else " (공개 선별)"))
    need = list(dict.fromkeys(cand + top))
    data = {n: load_dataset(n, cpgs=need) for n in a.datasets}
    for n, (b, p) in data.items():
        data[n] = (b, p.dropna(subset=["age"]))
        data[n] = (b[data[n][1].index], data[n][1])
        print(f"  {n}: {len(data[n][1])}명")

    # ── Q1~Q3: 교차 코호트 검증 ──
    rows = []
    for tr in a.datasets:
        Xa, ya = Xy(data, tr, cand)
        en = enet().fit(Xa, ya)
        coef = en.named_steps["elasticnetcv"].coef_
        en_top = [cand[i] for i in np.argsort(-np.abs(coef))[:n_top]]
        models = {
            OLS: (LinearRegression().fit(*Xy(data, tr, top)), top),
            f"ENet 후보전체": (en, cand),
            f"ENet 중요도 상위{n_top} → OLS": (LinearRegression().fit(*Xy(data, tr, en_top)), en_top),
        }
        overlap = len(set(en_top) & set(top))
        for te in a.datasets:
            if te == tr:
                continue
            for label, (m, mk) in models.items():
                X, y = Xy(data, te, mk)
                p = m.predict(X)
                rows.append(dict(train=tr, test=te, model=label, MAD=mad(y, p), R2=r2(y, p),
                                 n_used=int((np.abs(coef) > 0).sum()) if "후보전체" in label else len(mk)))
        print(f"  [{tr}] ENet 중요도 상위{n_top}과 소수 마커의 겹침: {overlap}개")
        if a.show_coef:
            m = models[OLS][0]
            print("   ", dict(zip(top, np.round(m.coef_, 3))), round(m.intercept_, 3))
    cv = pd.DataFrame(rows).round(3)
    cv.to_csv(RESULTS / "crossval.csv", index=False, encoding="utf-8-sig")
    print("\n== 교차 코호트 검증 ==\n" + cv.to_string(index=False))
    print("\n== 방법별 평균 ==\n" + cv.groupby("model")[["MAD", "R2"]].mean().round(2).to_string())

    # ── Q4: 보정 (최종 학습 코호트 기준) ──
    rows, plots = [], []
    for label, fit in ((OLS, lambda X, y: LinearRegression().fit(X, y)),
                       ("ENet 후보전체", lambda X, y: enet().fit(X, y))):
        mk = top if label == OLS else cand
        m = fit(*Xy(data, final_train, mk))
        for te in a.datasets:
            if te == final_train:
                continue
            X, y = Xy(data, te, mk)
            y_ev, preds, shift = calibrate(y, m.predict(X))
            for how, p in preds.items():
                rows.append(dict(model=label, test=te, calibration=how, MAD=mad(y_ev, p),
                                 R2=r2(y_ev, p), shift=shift if how == "평행이동" else np.nan,
                                 n_eval=len(y_ev)))
            if label == OLS:
                plots.append((te, y_ev, preds["보정 없음"], preds["평행이동"], shift))
    cal = pd.DataFrame(rows).round(3)
    cal.to_csv(RESULTS / "calibration.csv", index=False, encoding="utf-8-sig")
    print(f"\n== 보정 효과 ({final_train} 학습, 평가 코호트 50:50 분할) ==\n" + cal.to_string(index=False))

    korean_font()
    fig, axes = plt.subplots(2, len(plots), figsize=(4.4 * len(plots), 8.4), squeeze=False)
    for j, (te, y, raw, adj, shift) in enumerate(plots):
        for i, (p, tag) in enumerate(((raw, "보정 전"), (adj, f"평행이동 보정 후 ({shift:+.1f}년)"))):
            ax = axes[i, j]
            ax.scatter(y, p, s=8, alpha=0.45, color="#1f6f78", edgecolor="none")
            lim = [min(y.min(), p.min()) - 2, max(y.max(), p.max()) + 2]
            ax.plot(lim, lim, "--", color="grey", lw=1)
            ax.set_title(f"{final_train} 학습 → {te}\n{tag} | MAD {mad(y, p):.2f}년, R² {r2(y, p):.3f}",
                         fontsize=9)
            ax.set_xlabel("실제 나이")
            ax.set_ylabel("예측 나이")
    fig.suptitle(f"소수 마커({n_top}개) 모델: 예측 나이 vs 실제 나이", fontsize=11)
    fig.tight_layout()
    out = figure_dir(*a.datasets) / "pred_vs_age.png"
    fig.savefig(out, dpi=130)
    print(f"→ results/crossval.csv, results/calibration.csv, {out.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
