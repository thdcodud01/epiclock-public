"""공통 유틸: 데이터 로드(합성/공개 GEO), 평가 지표, 경로.

데이터셋 이름으로 불러온다.
  - 합성 데이터: data/synthetic/<이름>_beta.csv.gz + <이름>_pheno.csv   (예: blood_A)
  - 공개 GEO   : configs/datasets.json 에 정의된 이름                  (예: GSE87571)
모든 경로는 저장소 루트 기준 상대경로다.
"""
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

import geo_reader

ROOT = Path(__file__).resolve().parents[1]
SYN = ROOT / "data" / "synthetic"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
CONFIG = json.loads((ROOT / "configs" / "datasets.json").read_text(encoding="utf-8"))


def _is_synthetic(name):
    return (SYN / f"{name}_pheno.csv").exists()


@lru_cache(maxsize=None)
def load_pheno(name: str) -> pd.DataFrame:
    if _is_synthetic(name):
        return pd.read_csv(SYN / f"{name}_pheno.csv", dtype={"subject_id": str})
    if name in CONFIG:
        return geo_reader.build_pheno(CONFIG[name])
    raise FileNotFoundError(
        f"'{name}' 데이터가 없습니다. 합성 데이터는 `python src/make_synthetic.py`, "
        f"공개 데이터는 configs/datasets.json 과 docs/datasets.md 를 참고하세요.")


def iter_beta(name: str, chunksize=20000, cpgs=None):
    """(CpG × 샘플) 조각을 차례로 돌려준다. 열 순서는 load_pheno(name)의 sample_id 순서."""
    pheno = load_pheno(name)
    if _is_synthetic(name):
        keep = set(cpgs) if cpgs is not None else None
        for chunk in pd.read_csv(SYN / f"{name}_beta.csv.gz", index_col=0, chunksize=chunksize):
            if keep is not None:
                chunk = chunk[chunk.index.isin(keep)]
            cols = [s for s in pheno.sample_id if s in chunk.columns]
            if len(chunk):
                yield chunk[cols].astype(np.float32)
        return
    yield from geo_reader.iter_beta(CONFIG[name], pheno, chunksize, cpgs)


def figure_dir(*names) -> Path:
    """그림 저장 폴더. 합성 데이터를 하나라도 쓰면 figures/synthetic/ (저장소에 올리지 않음),
    공개 데이터만 쓰면 figures/ (README에 쓰는 그림)."""
    d = FIGURES / "synthetic" if any(_is_synthetic(n) for n in names) else FIGURES
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_dataset(name: str, cpgs=None):
    """(beta, pheno) 반환. cpgs를 주면 그 CpG만 읽는다(전체 450K는 메모리에 주의)."""
    pheno = load_pheno(name).set_index("sample_id")
    beta = pd.concat(list(iter_beta(name, cpgs=cpgs)))
    beta = beta[~beta.index.duplicated()]
    if cpgs is not None:
        beta = beta.reindex([c for c in cpgs if c in beta.index])
    samples = [s for s in beta.columns if s in pheno.index]
    return beta[samples], pheno.loc[samples]


# ─────────────────────────────────────────────────────────────
# 지표·통계 유틸
# ─────────────────────────────────────────────────────────────
def mad(y_true, y_pred) -> float:
    """평균 절대 오차(년)."""
    return float(np.mean(np.abs(np.asarray(y_pred) - np.asarray(y_true))))


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - y_true.mean()) ** 2))


def pearson_rows(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """X(행=CpG, 열=샘플)의 각 행과 y의 피어슨 상관 (벡터 연산)."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Xc = X - X.mean(axis=1, keepdims=True)
    yc = y - y.mean()
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc @ yc) / np.sqrt((Xc ** 2).sum(axis=1) * (yc ** 2).sum())


def residualizer(covars: np.ndarray) -> np.ndarray:
    """공변량 효과를 빼는 잔차화 행렬 R (n×n). R @ v = 공변량으로 설명되지 않는 부분."""
    C = np.column_stack([np.ones(covars.shape[0]), covars])
    return np.eye(C.shape[0]) - C @ np.linalg.pinv(C)


def bh_fdr(p) -> np.ndarray:
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def load_positions(manifest, cpgs=None) -> pd.DataFrame:
    """Illumina 450K 매니페스트(GPL13534)에서 CpG의 염색체·위치·유전자를 읽는다."""
    use = ["Name", "CHR", "MAPINFO", "UCSC_RefGene_Name"]
    man = pd.read_csv(manifest, sep="\t", skiprows=37, usecols=use, dtype=str,
                      on_bad_lines="skip", low_memory=False).set_index("Name")
    if cpgs is not None:
        man = man.reindex(cpgs)
    man["MAPINFO"] = pd.to_numeric(man["MAPINFO"], errors="coerce")
    return man


def pick_distinct_loci(ranked, pos, n, min_dist=1_000_000):
    """순위대로 보되, 이미 고른 CpG와 같은 염색체에서 min_dist 이내면 건너뛴다.
    (바로 옆 CpG끼리는 노출과 무관하게 함께 움직여 상관이 부풀려지기 때문)"""
    picked = []
    for c in ranked:
        if c not in pos.index or pd.isna(pos.at[c, "MAPINFO"]):
            continue
        ch, mp = pos.at[c, "CHR"], pos.at[c, "MAPINFO"]
        if all(pos.at[q, "CHR"] != ch or abs(pos.at[q, "MAPINFO"] - mp) >= min_dist for q in picked):
            picked.append(c)
        if len(picked) == n:
            break
    return picked


def ranked_labels(n):
    """공개용 마커 표기: 실제 CpG ID 대신 순위."""
    return [f"rank_{i + 1}" for i in range(n)]


def korean_font():
    """설치된 한글 폰트 중 첫 번째를 사용 (Windows: 맑은 고딕)."""
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR", "Noto Sans KR"):
        if name in have:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["axes.unicode_minus"] = False
