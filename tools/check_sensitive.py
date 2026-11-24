"""git push 전에 민감 정보가 남아 있는지 검사.

1) 기본 패턴: 사용자 PC 경로, 비밀번호·키·토큰, IP 주소, 이메일, DB 접속 문자열
2) 개인 패턴: tools/sensitive_patterns.txt (git에 올리지 않음)
   한 줄에 하나씩 정규식을 적는다. 사번, 회사명, 상품명, 협력사명, 실명 등.
3) 데이터 파일 검사: 합성 데이터 폴더 밖의 csv/xlsx/parquet/pkl 이 커밋 대상이면 경고

검사 대상은 'git이 추적하거나 추적하려는 파일'만이다(.gitignore 제외분은 안 봄).
사용법:  python tools/check_sensitive.py      (문제가 없으면 종료 코드 0)
"""
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = [
    r"[A-Za-z]:\\\\?Users\\\\?",           # C:\Users\...
    r"/Users/[^/\s]+/|/home/[^/\s]+/",     # mac/linux 홈 경로
    r"(?i)pass(word|wd)?\s*[=:]",
    r"(?i)(api[_-]?key|secret|token)\s*[=:]",
    r"(?i)(oracle|cx_oracle|oracledb|jdbc|dsn\s*=)",
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",
    r"[\w.+-]+@[\w-]+\.[\w.]+",
]
DATA_EXT = {".csv", ".tsv", ".xlsx", ".xls", ".parquet", ".pkl", ".pickle", ".feather", ".h5"}
ALLOW_DATA_DIR = "data/synthetic/"
SKIP = {"tools/check_sensitive.py", "tools/sensitive_patterns.txt"}


def tracked_files():
    out = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def main():
    pats = list(BASE)
    local = ROOT / "tools" / "sensitive_patterns.txt"
    if local.exists():
        pats += [l.strip() for l in local.read_text(encoding="utf-8").splitlines()
                 if l.strip() and not l.startswith("#")]
    else:
        print("※ tools/sensitive_patterns.txt 가 없어 기본 패턴만 검사합니다.")
    rx = [re.compile(p) for p in pats]

    problems = []
    for rel in tracked_files():
        if rel in SKIP:
            continue
        p = ROOT / rel
        if p.suffix.lower() in DATA_EXT and not rel.startswith(ALLOW_DATA_DIR):
            problems.append(f"[데이터 파일] {rel}")
            continue
        if p.suffix.lower() in {".gz", ".zip", ".png", ".jpg", ".pdf", ".docx", ".pptx"}:
            if not rel.startswith(ALLOW_DATA_DIR) and p.suffix.lower() in {".pdf", ".docx", ".pptx", ".zip"}:
                problems.append(f"[문서·압축 파일] {rel}  ← 내용 직접 확인 필요")
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for i, line in enumerate(text.splitlines(), 1):
            for r in rx:
                if r.search(line):
                    problems.append(f"{rel}:{i}: {line.strip()[:100]}")
                    break

    if problems:
        print(f"확인 필요 {len(problems)}건:")
        print("\n".join(problems))
        sys.exit(1)
    print("민감 정보 패턴이 발견되지 않았습니다.")


if __name__ == "__main__":
    main()
