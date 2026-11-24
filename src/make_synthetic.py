"""합성(가짜) 예시 데이터 생성.

실제 GEO 데이터와 '구조'만 같고 값은 전부 난수로 만든 데이터다.
- CpG ID는 cgS000001 형식의 가짜 ID (실제 450K 프로브와 무관)
- 실제 코호트의 개인 값, 실제 나이-메틸화 관계와는 관계없음

만드는 데이터셋                         대응하는 공개 데이터(역할)
  blood_A / blood_B / blood_C            혈액 코호트 3개 (코호트마다 측정 기준점이 조금씩 다름)
  blood_rep                              같은 DNA를 2번 측정한 technical replicate 쌍
  smoke_A / smoke_B                      흡연 코호트 (B에는 질병 상태 변수 포함)
  exposure_A                             대기오염 노출값이 있는 코호트 (노출 효과 없음)
  saliva_A / saliva_B                    타액 코호트 2개 (나이 신호가 약하고 잡음이 큼)
  paired_bs                              같은 사람의 혈액·타액 쌍

사용법:  python src/make_synthetic.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

N_CPG, N_AGE, N_SMOKE = 800, 80, 25
RNG = np.random.default_rng(20261117)

cpg_ids = [f"cgS{i:06d}" for i in range(1, N_CPG + 1)]
base = RNG.uniform(0.15, 0.85, N_CPG)
age_slope = np.zeros(N_CPG)
age_idx = RNG.choice(N_CPG, N_AGE, replace=False)
age_slope[age_idx] = RNG.choice([-1, 1], N_AGE) * RNG.uniform(0.0006, 0.0035, N_AGE)
rest = np.setdiff1d(np.arange(N_CPG), age_idx)
smoke_idx = RNG.choice(rest, N_SMOKE, replace=False)
smoke_eff = np.zeros(N_CPG)
smoke_eff[smoke_idx] = -RNG.uniform(0.03, 0.12, N_SMOKE)
saliva_offset = RNG.normal(0, 0.05, N_CPG)                  # 조직 차이


def simulate(n, age_lo, age_hi, prefix, batch_shift=0.0, noise=0.05, age_signal=1.0,
             smoking=None, tissue_offset=0.0):
    age = RNG.integers(age_lo, age_hi + 1, n).astype(float)
    sex = RNG.choice(["F", "M"], n)
    mu = base[:, None] + tissue_offset + age_signal * age_slope[:, None] * (age[None, :] - 50)
    if smoking is not None:
        w = np.select([smoking == "current", smoking == "former"], [1.0, 0.45], 0.0)
        mu = mu + smoke_eff[:, None] * w[None, :]
    beta = np.clip(mu + batch_shift + RNG.normal(0, noise, (N_CPG, n)), 0.001, 0.999)
    ids = [f"{prefix}{i:04d}" for i in range(1, n + 1)]
    pheno = pd.DataFrame({"sample_id": ids, "age": age, "sex": sex})
    if smoking is not None:
        pheno["smoking"] = smoking
    return pd.DataFrame(beta, index=cpg_ids, columns=ids), pheno


def save(name, beta, pheno):
    beta = beta.round(4)
    beta.index.name = "cpg"
    beta.to_csv(OUT / f"{name}_beta.csv.gz", compression="gzip")
    pheno.to_csv(OUT / f"{name}_pheno.csv", index=False)
    print(f"{name:11s} CpG {beta.shape[0]} x 샘플 {beta.shape[1]}")


def smoking_status(n, p=(0.4, 0.35, 0.25)):
    return RNG.choice(["never", "former", "current"], n, p=list(p))


if __name__ == "__main__":
    save("blood_A", *simulate(250, 18, 90, "A"))
    save("blood_B", *simulate(300, 30, 80, "B", batch_shift=0.015))
    save("blood_C", *simulate(220, 33, 75, "C", batch_shift=-0.012))

    # technical replicate: 같은 DNA 60명을 2번 측정 (측정 잡음만 다름)
    b, p = simulate(60, 30, 80, "R", batch_shift=-0.012, noise=0.03)
    reps, rows = [], []
    for k in (1, 2):
        rb = np.clip(b + RNG.normal(0, 0.015, b.shape), 0.001, 0.999)
        rb.columns = [f"{c}_r{k}" for c in b.columns]
        reps.append(rb)
        rows.append(p.assign(subject_id=p.sample_id, rep=k, sample_id=[f"{s}_r{k}" for s in p.sample_id]))
    save("blood_rep", pd.concat(reps, axis=1), pd.concat(rows, ignore_index=True))

    save("smoke_A", *simulate(400, 25, 75, "SA", smoking=smoking_status(400)))
    b, p = simulate(300, 25, 75, "SB", smoking=smoking_status(300))
    p["disease"] = RNG.choice([0, 1], len(p), p=[0.45, 0.55])          # 환자/대조군 혼재
    save("smoke_B", b, p)

    b, p = simulate(300, 20, 70, "EX", smoking=smoking_status(300, (0.75, 0.15, 0.10)))
    p["pm25"] = RNG.gamma(4, 2.5, len(p)).round(2)                     # 노출 효과 없음(귀무)
    p["no2"] = RNG.gamma(5, 4, len(p)).round(2)
    p["pm10"] = (p["pm25"] * 1.6 + RNG.normal(0, 3, len(p))).round(2)
    save("exposure_A", b, p)

    save("saliva_A", *simulate(220, 35, 88, "SVA", age_signal=0.75, noise=0.06,
                               batch_shift=0.01, tissue_offset=saliva_offset[:, None]))
    save("saliva_B", *simulate(54, 18, 73, "SVB", age_signal=0.75, noise=0.06,
                               batch_shift=-0.01, tissue_offset=saliva_offset[:, None]))

    # 같은 사람 25명의 혈액·타액
    bb, pb = simulate(25, 30, 70, "PB")
    sal = np.clip(bb.values + saliva_offset[:, None] + RNG.normal(0, 0.04, bb.shape), 0.001, 0.999)
    bs = pd.DataFrame(sal, index=cpg_ids, columns=[c.replace("PB", "PS") for c in bb.columns])
    ph = pd.concat([pb.assign(subject_id=pb.sample_id, tissue="blood"),
                    pb.assign(subject_id=pb.sample_id, tissue="saliva",
                              sample_id=[s.replace("PB", "PS") for s in pb.sample_id])])
    save("paired_bs", pd.concat([bb, bs], axis=1), ph.drop(columns="age"))
