"""공개 GEO 데이터를 처음 쓸 때 설정(configs/datasets.json)이 맞는지 확인.

  python src/check_dataset.py GSE87571

출력: 샘플 정보 컬럼, 표준 컬럼(나이·성별·흡연 등) 매핑 결과, beta 열 매핑 성공 여부,
     첫 조각의 크기. 매핑이 틀리면 datasets.json 의 해당 항목을 고친다.
"""
import sys

import geo_reader
from common import CONFIG

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in CONFIG:
        names = [k for k in CONFIG if not k.startswith("_")]
        sys.exit(f"사용법: python src/check_dataset.py <이름>\n가능한 이름: {names}")
    name = sys.argv[1]
    cfg = CONFIG[name]
    meta = geo_reader.read_meta(cfg["matrix"])
    print(f"[{name}] {cfg.get('role', '')}")
    print(f"샘플 정보 컬럼({len(meta.columns)}):")
    for i, c in enumerate(meta.columns):
        ex = meta.iloc[:, i].dropna().astype(str).unique()[:3]
        print(f"  - {c}: {list(ex)}")

    pheno = geo_reader.build_pheno(cfg, meta)
    print(f"\n표준 컬럼: {list(pheno.columns)}  /  샘플 {len(pheno)}명")
    for c in ("age", "sex", "smoking", "disease", "tissue", "rep"):
        if c in pheno:
            s = pheno[c]
            print(f"  {c}: " + (f"{s.min():.0f}~{s.max():.0f}, 결측 {s.isna().sum()}" if c == "age"
                                else str(s.value_counts(dropna=False).to_dict())))

    first = next(geo_reader.iter_beta(cfg, pheno, chunksize=2000), None)
    if first is None:
        sys.exit("\nbeta 값 없음: series_matrix에 beta 표가 없습니다(data_row_count 0). "
                 "GEO supplementary 파일을 받아 datasets.json 의 beta_files 로 지정해야 합니다.")
    print(f"\nbeta 첫 조각: CpG {first.shape[0]} × 샘플 {first.shape[1]}"
          f"  (샘플 정보 {len(pheno)}명 중 {first.shape[1]}명 매핑)")
    if first.shape[1] < len(pheno):
        print("  ⚠ 일부 샘플이 beta 열과 매핑되지 않음 → title_key_regex / col_key_regex 확인")
