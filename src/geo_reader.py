"""공개 GEO 450K 데이터 읽기 (configs/datasets.json 의 정의를 따름).

450K 전체(약 48만 CpG × 수백~수천 명)는 한 번에 메모리에 올리기 어려우므로,
원 분석과 같이 CpG 방향으로 '조각(chunk)' 단위로 읽는다.

지원하는 파일 구조
  (1) series_matrix 안에 beta 표가 있는 경우            → GSM ID가 열 이름
  (2) beta 표가 별도 파일(들)로 제공되는 경우            → 열 이름(X번호, 칩 바코드 등)을
      series_matrix의 !Sample_title 에서 뽑은 키와 맞춰 GSM ID로 바꾼다.
      p-value 등 보조 열은 col_key_regex 에 맞지 않으므로 자동으로 빠진다.
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _open(path):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")


def _clean(v):
    return v.strip().strip('"')


def _norm_key(k):
    """'age (y)' → 'age_y', 'PM2.5 (1-yr avg)' → 'pm2_5_1_yr_avg'. 괄호 안 정보도 유지한다
    (노출 기간처럼 괄호로만 구분되는 항목이 합쳐지지 않도록)."""
    return re.sub(r"\W+", "_", k.strip().lower()).strip("_")


# ─────────────────────────────────────────────────────────────
# 1) series_matrix 메타데이터 → 샘플 정보 표
# ─────────────────────────────────────────────────────────────
RESERVED = {"geo_accession": "sample_id"}


def read_meta(matrix_path):
    cols, owner = {}, {}                  # owner: 컬럼 이름 → 처음 만든 줄 번호
    with _open(ROOT / matrix_path) as f:
        for ln, line in enumerate(f):
            if line.startswith("!series_matrix_table_begin"):
                break
            if not line.startswith("!Sample_"):
                continue
            tag, *vals = [_clean(v) for v in line.rstrip("\n").split("\t")]
            tag = tag.replace("!Sample_", "")
            if tag == "characteristics_ch1":
                for i, v in enumerate(vals):
                    if ":" not in v:
                        continue
                    k, val = v.split(":", 1)
                    key = _norm_key(k)
                    if key in RESERVED.values() or key in RESERVED:
                        key = f"char_{key}"          # 'sample id' 같은 항목이 GSM ID와 겹치지 않게
                    base, n = key, 2
                    while key in owner and owner[key] != ln:   # 다른 줄의 같은 이름 → 번호 붙임
                        key, n = f"{base}_{n}", n + 1
                    owner.setdefault(key, ln)
                    cols.setdefault(key, [None] * len(vals))[i] = val.strip()
            elif tag not in cols:                              # 같은 태그가 여러 줄이면 첫 줄 사용
                cols[tag] = vals
                owner[tag] = ln
    meta = pd.DataFrame(cols)
    return meta.rename(columns=RESERVED)


def _pick_col(meta, spec_col=None, spec_regex=None, default=()):
    """정확히 같은 이름 → 정규식 → 기본 이름 순. 이름이 정확히 없으면 '<이름>_...' 로 시작하는
    컬럼도 받는다 (예: 'smoking (0, 1 and 2, ...)' → 'smoking_0_1_and_2_...')."""
    def find(name):
        if name in meta.columns:
            return name
        return next((c for c in meta.columns if c.startswith(name + "_")), None)

    if spec_col and find(spec_col):
        return find(spec_col)
    if spec_regex:
        for c in meta.columns:
            if re.search(spec_regex, c, re.I):
                return c
    for c in default:
        if find(c):
            return find(c)
    return None


def _map_smoking(v, rule):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip().lower()
    if "map" in rule:
        return rule["map"].get(s, np.nan)
    for key, lab in rule.get("contains", {}).items():
        short = len(key) <= 3
        if (short and (s == key or s.startswith(key + "-") or s.startswith(key + " "))) or \
           (not short and key in s):
            return lab
    return np.nan


def build_pheno(cfg, meta=None):
    """설정에 따라 표준 컬럼(sample_id, key, age, sex, smoking, ...)으로 정리."""
    meta = read_meta(cfg["matrix"]) if meta is None else meta
    p = pd.DataFrame({"sample_id": meta["sample_id"]})

    # 열 매핑 키 (별도 beta 파일일 때만 필요)
    if "title_key_regex" in cfg:
        p["key"] = meta["title"].str.extract(cfg["title_key_regex"])[0].values
    else:
        p["key"] = meta["sample_id"]

    age_c = _pick_col(meta, default=("age",)) or next((c for c in meta.columns if c.startswith("age")), None)
    if age_c:
        p["age"] = pd.to_numeric(meta[age_c].astype(str).str.extract(r"([-+]?\d+\.?\d*)")[0],
                                 errors="coerce")
    sex_c = _pick_col(meta, default=("gender", "sex"))
    if sex_c:
        first = meta[sex_c].astype(str).str.strip().str.upper().str[0]
        p["sex"] = first.where(first.isin(["M", "F"]))

    if "smoking" in cfg:
        r = cfg["smoking"]
        c = _pick_col(meta, r.get("col"), r.get("col_regex"))
        if c:
            p["smoking"] = meta[c].map(lambda v: _map_smoking(v, r))
    if "disease" in cfg:
        r = cfg["disease"]
        c = _pick_col(meta, r.get("col"), r.get("col_regex"))
        if c:
            low = meta[c].astype(str).str.lower()
            p["disease"] = (~low.str.contains("|".join(r["control_contains"]))).astype(int)
    for name, rx in cfg.get("exposures", {}).items():
        c = _pick_col(meta, spec_regex=rx)
        if c:
            p[name] = pd.to_numeric(meta[c], errors="coerce")
    if "subject" in cfg:
        r = cfg["subject"]
        p["subject_id"] = meta[r["from"]].str.extract(r["regex"])[0].values
    if "tissue_col" in cfg and cfg["tissue_col"] in meta.columns:
        p["tissue"] = meta[cfg["tissue_col"]].str.strip().str.lower()

    if "filter" in cfg:
        r = cfg["filter"]
        c = _pick_col(meta, r.get("col"), r.get("col_regex"))
        if c is None:
            print(f"  ⚠ 필터 컬럼을 찾지 못함: {r} → 필터 없이 진행")
        else:
            keep = meta[c].astype(str).str.contains(r["contains"], case=False, na=False)
            p = p[keep.values]

    if "replicates" in cfg:
        p = _pair_replicates(p, meta, cfg["replicates"])
    return p.reset_index(drop=True)


def _pair_replicates(p, meta, rule):
    """description의 'group G, sample S' 로 반복측정 쌍 구성.
    그룹이 2개면 두 그룹에서 같은 sample 번호끼리, 아니면 그룹 안의 앞 2개를 쌍으로."""
    g = meta.loc[p.index, rule["from"]].astype(str).str.extract(rule["regex"], flags=re.I)
    p = p.assign(grp=pd.to_numeric(g[0], errors="coerce").values,
                 smp=pd.to_numeric(g[1], errors="coerce").values).dropna(subset=["grp", "smp"])
    groups = sorted(p.grp.unique())
    if len(groups) == 2:
        p = p.drop_duplicates(["grp", "smp"])
        p["subject_id"] = p["smp"].astype(int).astype(str)
        p["rep"] = (p["grp"] == groups[1]).astype(int) + 1
    else:
        p = p.sort_values(["grp", "smp"])
        p["rep"] = p.groupby("grp").cumcount() + 1
        p = p[p.rep <= 2]
        p["subject_id"] = p["grp"].astype(int).astype(str)
    both = p.groupby("subject_id")["rep"].transform("nunique") == 2
    return p[both].drop(columns=["grp", "smp"])


# ─────────────────────────────────────────────────────────────
# 2) beta 표 조각 단위 읽기
# ─────────────────────────────────────────────────────────────
def _select_columns(path, col_regex, wanted_keys):
    with _open(ROOT / path) as f:
        header = [_clean(h) for h in f.readline().rstrip("\n").split("\t")]
    pos, keys, seen = [0], [], set()
    for i, h in enumerate(header[1:], start=1):
        m = re.search(col_regex, h)
        if m and m.group(1) in wanted_keys and m.group(1) not in seen:
            pos.append(i)
            keys.append(m.group(1))
            seen.add(m.group(1))
    return pos, keys


def iter_beta(cfg, pheno, chunksize=20000, cpgs=None):
    """(CpG × sample_id) DataFrame 조각을 차례로 돌려준다. cpgs가 있으면 그 행만."""
    key2id = dict(zip(pheno["key"], pheno["sample_id"]))
    keep = set(cpgs) if cpgs is not None else None
    files = cfg.get("beta_files")

    def _filter(chunk):
        chunk.index = chunk.index.astype(str).str.strip('"')
        return chunk[chunk.index.isin(keep)] if keep is not None else chunk

    if not files:                                         # (1) series_matrix 안의 표
        reader = pd.read_csv(ROOT / cfg["matrix"], sep="\t", comment="!", index_col=0,
                             chunksize=chunksize, low_memory=False)
        for chunk in reader:
            chunk.columns = [_clean(str(c)) for c in chunk.columns]
            cols = [s for s in pheno["sample_id"] if s in chunk.columns]
            out = _filter(chunk[cols]).astype(np.float32)
            if len(out):
                yield out
        return

    parts = []
    for path in files:                                    # (2) 별도 beta 파일
        pos, keys = _select_columns(path, cfg["col_key_regex"], set(key2id))
        reader = pd.read_csv(ROOT / path, sep="\t", usecols=pos, index_col=0,
                             chunksize=chunksize, low_memory=False)
        if len(files) == 1:
            for chunk in reader:
                chunk.columns = [key2id[k] for k in keys]
                out = _filter(chunk).astype(np.float32)
                if len(out):
                    yield out
            return
        # 파일이 여러 개면 샘플이 나뉘어 있으므로 합친 뒤 내보낸다(메모리 사용 큼).
        frames = []
        for chunk in reader:
            chunk.columns = [key2id[k] for k in keys]
            frames.append(_filter(chunk).astype(np.float32))
        parts.append(pd.concat(frames))
    full = parts[0].join(parts[1:], how="inner") if len(parts) > 1 else parts[0]
    full = full.loc[:, ~full.columns.duplicated()]
    for s in range(0, len(full), chunksize):
        yield full.iloc[s:s + chunksize]
