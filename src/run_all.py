"""합성 데이터로 전체 분석을 순서대로 실행:  python src/run_all.py"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = ["make_synthetic.py",
         "marker_selection.py --refresh", "clock_model.py", "reliability.py",
         "agegroup_change.py", "sex_difference.py",
         "smoking_axis.py", "never_smoker_spread.py", "exposure_axis_test.py",
         "saliva_transfer.py"]

for s in STEPS:
    script, *args = s.split()
    print(f"\n{'=' * 22} {s} {'=' * 22}", flush=True)
    subprocess.run([sys.executable, str(SRC / script), *args], cwd=SRC, check=True)
