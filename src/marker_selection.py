"""1단계: 여러 코호트에서 공통으로 나이와 연관된 CpG 선별 (마커 선별 퍼널).

방법 (원 분석과 동일)
  1) 코호트마다 모든 CpG의 나이 상관 r 계산 (조각 단위로 읽어 메모리 절약)
  2) 모든 코호트에 있는 CpG만 남김
  3) 모든 코호트에서 방향(부호)이 같고, |r|이 기준 이상인 CpG만 통과
  4) 코호트 간 평균 |r| 순으로 정렬

판단 기준(결과를 보기 전에 정함): 세 코호트 모두 방향 일치 + min|r| >= 기준
  기준별 통과 수를 모두 보여주고, 모델 후보는 0.5, 탐색용 후보는 0.4를 쓴다.

선택: --manifest 로 Illumina 450K 매니페스트(GPL13534)를 주면 유전자 이름을 붙이고,
      이후 소수 마커 모델에서 "유전자당 1개" 규칙을 쓸 수 있다.

출력 (results/ 는 저장소에 올리지 않음)
  results/corr_all_<코호트>.csv  : 코호트별 전체 CpG 나이 상관 (재실행 시 재사용)
  results/marker_funnel.csv      : 기준별 통과 CpG 수
  results/selected_markers.csv   : 통과 CpG와 코호트별 r

사용법
  합성 데이터:  python src/marker_selection.py
  공개 데이터:  python src/marker_selection.py --datasets GSE40279 GSE87571 GSE55763 \
                    --manifest data/raw/GPL13534-11288.txt
"""
import argparse

import numpy as np
import pandas as pd

from common import RESULTS, iter_beta, load_pheno, pearson_rows

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


def age_corr_all(name, refresh=False):
    out = RESULTS / f"corr_all_{name}.csv"
    if out.exists() and not refresh:
        return pd.read_csv(out, index_col=0).iloc[:, 0]
    pheno = load_pheno(name).set_index("sample_id")
    parts, n = [], 0
    for chunk in iter_beta(name):
        age = pheno.loc[chunk.columns, "age"].values.astype(float)
        ok = ~np.isnan(age)
        r = pearson_rows(chunk.values[:, ok], age[ok])
        parts.append(pd.Series(r, index=chunk.index, name=f"r_{name}"))
        n += len(chunk)
        print(f"  [{name}] {n:,} CpG", end="\r")
    s = pd.concat(parts)
    s.to_csv(out, encoding="utf-8-sig")
    print(f"  [{name}] 나이 상관 계산 완료: {len(s):,} CpG ({int(ok.sum())}명)")
    return s


def annotate(sel, manifest):
    use = ["Name", "CHR", "MAPINFO", "UCSC_RefGene_Name"]
    man = pd.read_csv(manifest, sep="\t", skiprows=37, usecols=use, dtype=str,
                      on_bad_lines="skip", low_memory=False).set_index("Name")

    def first_gene(v):
        if pd.isna(v) or v == "":
            return None                      # 유전자 미지정(위치만 있는 CpG)
        return list(dict.fromkeys(g for g in str(v).split(";") if g))[0]

    man["gene"] = man["UCSC_RefGene_Name"].map(first_gene)
    return sel.join(man[["gene", "CHR", "MAPINFO"]], how="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["blood_A", "blood_B", "blood_C"])
    ap.add_argument("--thr", type=float, default=0.5, help="저장할 후보 기준 min|r|")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--refresh", action="store_true", help="저장된 상관값을 무시하고 다시 계산")
    a = ap.parse_args()

    corr = pd.concat([age_corr_all(n, a.refresh) for n in a.datasets], axis=1, join="inner").dropna()
    corr.columns = [f"r_{n}" for n in a.datasets]
    agree = np.sign(corr).nunique(axis=1) == 1
    tbl = corr.assign(mean_abs_r=corr.abs().mean(axis=1), min_abs_r=corr.abs().min(axis=1),
                      direction=np.where(corr.iloc[:, 0] > 0, "+", "-"))

    rows = [("모든 코호트 공통 CpG", "-", len(corr)), ("방향 일치", "모든 코호트 같은 부호", int(agree.sum()))]
    rows += [("상관 기준", f"min|r| >= {t}", int((agree & (tbl.min_abs_r >= t)).sum())) for t in THRESHOLDS]
    funnel = pd.DataFrame(rows, columns=["단계", "기준", "CpG 수"])

    sel = tbl[agree & (tbl.min_abs_r >= a.thr)].sort_values("mean_abs_r", ascending=False)
    if a.manifest:
        sel = annotate(sel, a.manifest)
        print(f"유전자 미지정 CpG: {sel.gene.isna().sum()}개 (패널 설계에는 쓰기 어려움)")

    funnel.to_csv(RESULTS / "marker_funnel.csv", index=False, encoding="utf-8-sig")
    sel.to_csv(RESULTS / "selected_markers.csv", encoding="utf-8-sig")
    print("\n== 마커 선별 퍼널 ==")
    print(funnel.to_string(index=False))
    print(f"\n기준 min|r| >= {a.thr}: {len(sel)}개 → results/selected_markers.csv")


if __name__ == "__main__":
    main()
