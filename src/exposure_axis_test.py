"""4-2단계: 흡연 마커가 대기오염 노출도 반영하는가? (환경노출 축 가부 판정)

데이터: 대기오염 노출값(PM2.5, NO2 등)이 표현형에 포함된 공개 코호트
방법 (원 분석과 동일)
  - 흡연 교란을 없애기 위해 비흡연자만 사용(50명 이상일 때). 부족하면 흡연을 공변량으로.
  - 나이·성별 보정 후, 흡연 패널 마커 × 노출 변수 편상관
  - 음성 대조: 무작위 CpG 200개로 '우연히 유의하게 나오는 비율'을 함께 계산
  - 양성 대조(선택): 문헌에서 노출과 재현된 CpG를 --positive 로 주면 데이터가 정상인지 확인

판단 기준(결과를 보기 전에 정함)
  진행: 패널 마커 중 2개 이상이 PM2.5 또는 NO2와 (a) 흡연과 같은 방향(음의 상관)
        (b) 패널 내 BH-FDR < 0.05
  폐기: 위 조건 미충족 → "이 조건에서 재현되지 않음"으로 기록 (존재하지 않는다는 뜻은 아님)

전제: smoking_axis.py 실행 후 (흡연 패널 = discovery 상위 마커)
사용법
  합성 데이터:  python src/exposure_axis_test.py
  공개 데이터:  python src/exposure_axis_test.py --dataset GSE294810 --panel-from GSE50660

원 분석 경과: 노출값이 붙은 공개 데이터로 GSE294810(천식 코호트)을 찾았으나, 논문에서 쓴
  노출값이 GEO에 공개되지 않아 이 검증은 실행할 수 없었다. 문헌상으로도 대기오염 EWAS의
  외부 재현 신호에 흡연 마커가 없어서, 대기오염 축은 "공개 데이터로 검증 불가 + 문헌 근거
  약함"으로 제외했다. 이 스크립트는 노출 데이터가 확보될 때 바로 쓸 수 있도록 남겨 둔
  사전 설계이며, 합성 데이터(exposure_A)로만 동작을 확인했다.
"""
import argparse

import numpy as np
import pandas as pd
from scipy import stats

from common import RESULTS, bh_fdr, iter_beta, load_dataset, load_pheno, ranked_labels, residualizer

EXPOSURES = ["pm25", "no2", "pm10", "o3", "ec", "nox"]
KEY = ["pm25", "no2"]
N_NULL = 200


def assoc(beta, p, expo, label):
    rows = []
    for e in expo:
        s = p.dropna(subset=[e])
        if len(s) < 40:
            continue
        cov = [s["age"].values, (s["sex"] == "M").astype(float).values]
        if "smoke_code" in s:
            cov.append(s["smoke_code"].values)
        R = residualizer(np.column_stack(cov))
        y = R @ s[e].values.astype(float)
        for cpg in beta.index:
            v = beta.loc[cpg, s.index].values.astype(float)
            if np.isnan(v).any() or v.std() == 0:
                continue
            r, pv = stats.pearsonr(R @ v, y)
            rows.append(dict(cpg=cpg, group=label, exposure=e, n=len(s), r=r, p=pv))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="exposure_A")
    ap.add_argument("--panel-from", default="smoke_A", help="흡연 패널을 가져올 discovery 코호트")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--positive", nargs="*", default=[], help="양성 대조 CpG ID(문헌)")
    a = ap.parse_args()

    panel = list(pd.read_csv(RESULTS / f"smoking_scan_{a.panel_from}.csv", index_col=0).index[: a.top])
    p = load_pheno(a.dataset).set_index("sample_id").dropna(subset=["age", "sex"])
    expo = [e for e in EXPOSURES if e in p.columns]
    if not expo:
        raise SystemExit(
            f"[{a.dataset}] 표현형에 대기오염 노출 변수가 없습니다.\n"
            "→ 판정: 공개 데이터로는 검증 불가. ('연관이 없다'가 아니라 '확인할 수 없다')\n"
            "   GSE294810은 논문에서 쓴 노출값이 GEO에 공개되지 않아 원 분석에서도 같은 결론이었습니다.\n"
            "   노출값이 붙은 데이터를 확보하면 configs/datasets.json 의 exposures 에 컬럼을 지정해 다시 실행합니다.")
    if "smoking" in p:
        nv = p.index[p.smoking == "never"]
        if len(nv) >= 50:
            p = p.loc[nv]
            print(f"비흡연자 {len(p)}명만 사용 (흡연 교란 제거)")
        else:
            p = p.assign(smoke_code=p.smoking.astype("category").cat.codes)
            print("비흡연자가 부족해 흡연 상태를 공변량으로 추가")

    first = next(iter_beta(a.dataset, chunksize=5000))
    pool = [c for c in first.index if c not in panel + a.positive]
    nulls = list(np.random.default_rng(0).choice(pool, min(N_NULL, len(pool)), replace=False))
    b, _ = load_dataset(a.dataset, cpgs=panel + a.positive + nulls)
    b = b[[s for s in b.columns if s in p.index]]
    p = p.loc[b.columns]

    res_p = assoc(b.loc[[c for c in panel if c in b.index]], p, expo, "패널")
    res_p["fdr"] = bh_fdr(res_p.p.values)
    res_p["marker"] = res_p.cpg.map(dict(zip(panel, ranked_labels(len(panel)))))
    res_n = assoc(b.loc[nulls], p, expo, "음성대조")
    res_pos = assoc(b.loc[[c for c in a.positive if c in b.index]], p, expo, "양성대조") if a.positive else None

    print(res_p[["marker", "exposure", "n", "r", "p", "fdr"]].round(4).to_string(index=False))
    if res_pos is not None and len(res_pos):
        print("\n양성 대조:\n" + res_pos[["exposure", "r", "p"]].round(4).to_string(index=False))
    hits = res_p[res_p.exposure.isin(KEY) & (res_p.r < 0) & (res_p.fdr < .05)]
    print(f"\n조건 충족 마커 (PM2.5/NO2, 음의 방향, FDR<0.05): {hits.marker.nunique()}개")
    print(f"명목 유의(p<0.05) 비율 — 패널 {(res_p.p < .05).mean() * 100:.1f}% / "
          f"음성대조 {(res_n.p < .05).mean() * 100:.1f}% (우연 수준 ≈ 5%)")
    print("결론:", "진행 — 패널 마커가 대기오염 노출과 연관" if hits.marker.nunique() >= 2 else
          "폐기 — 이 조건에서 연관이 재현되지 않음")
    pd.concat([res_p.drop(columns="cpg"), res_n.drop(columns="cpg").assign(marker="null")]) \
      .round(4).to_csv(RESULTS / "exposure_axis_test.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
