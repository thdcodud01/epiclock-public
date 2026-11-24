# 공개 데이터로 직접 돌려보기

합성 데이터로는 `python src/run_all.py` 한 줄이면 됩니다. 아래는 실제 공개 GEO 데이터를 쓰는 방법입니다.

## 1. 내려받을 파일

각 GEO 페이지(`https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=<GSE번호>`) 아래쪽에서 받아 `data/raw/<GSE번호>/`에 둡니다.

| 데이터셋 | 역할 | 필요한 파일 |
|---|---|---|
| GSE40279 | 혈액 (미국 656명) | series_matrix (beta 포함) |
| GSE87571 | 혈액 (스웨덴 732명, 학습) | series_matrix + `matrix1of2`, `matrix2of2` |
| GSE55763 | 혈액 (영국 2,664명) + 반복측정 쌍 | series_matrix + `normalized_betas` (약 10GB) |
| GSE50660 | 흡연 discovery | series_matrix (beta 포함) |
| GSE42861 | 흡연 replication | series_matrix (beta 포함) |
| GSE294810 | 대기오염 노출 (검증 불가) | 노출값이 GEO에 없어 `exposure_axis_test.py`는 실행되지 않음 |
| GSE78874, GSE92767 | 타액 | series_matrix (beta 포함) |
| GSE130153 | 같은 사람 혈액·타액 | series_matrix (beta 포함) |
| GPL13534 | 450K 매니페스트 (유전자 이름) | `GPL13534-11288.txt` → `data/raw/` |

## 2. 설정 확인

`configs/datasets.json`의 컬럼 이름은 원 분석 코드를 기준으로 적었습니다. 처음 쓸 때는 데이터셋마다 한 번씩 확인합니다.

```powershell
python src/check_dataset.py GSE87571
```

나이·성별·흡연 매핑과 beta 열 매핑 결과가 출력됩니다. 틀린 곳이 있으면 json의 해당 항목을 고칩니다.

## 3. 실행 순서

```powershell
python src/marker_selection.py --datasets GSE40279 GSE87571 GSE55763 --manifest data/raw/GPL13534-11288.txt
python src/clock_model.py      --datasets GSE40279 GSE87571 GSE55763 --final-train GSE87571
python src/reliability.py      --train GSE87571 --rep GSE55763_rep --effect 2
python src/agegroup_change.py  --datasets GSE40279 GSE87571 GSE55763
python src/sex_difference.py   --datasets GSE40279 GSE87571 GSE55763
python src/smoking_axis.py     --discovery GSE50660 --replication GSE42861
python src/never_smoker_spread.py --dataset GSE50660 --manifest data/raw/GPL13534-11288.txt
python src/saliva_transfer.py  --a GSE78874 --b GSE92767 --paired GSE130153
```

### 최종 패널로 실행 (로컬 전용)

`configs/panel.example.txt`를 `configs/panel.local.txt`로 복사해 최종 패널의 CpG ID를 한 줄에 하나씩 적은 뒤, 모델·신뢰도·타액 분석에 `--panel configs/panel.local.txt`를 붙입니다. 이 파일은 `.gitignore` 대상이라 저장소에 올라가지 않습니다. Windows PowerShell로 파일을 만들 때는 `-Encoding ascii`를 쓰세요.

## 4. 결과 저장 위치

- `results/`: 표 형태 결과(로컬 확인용, CpG ID 포함). 저장소에 올리지 않습니다.
- `figures/`: 공개 데이터로 만든 그림. README에 씁니다.
- `figures/synthetic/`: 합성 데이터로 만든 그림. 저장소에 올리지 않습니다.

## 5. 메모리와 시간

- 전체 CpG를 쓰는 분석은 데이터를 조각 단위로 읽습니다. `marker_selection`, `agegroup_change`, `smoking_axis`의 스캔 단계가 여기에 해당합니다.
- GSE87571은 beta가 두 파일에 샘플별로 나뉘어 있어서 합친 뒤 처리합니다. 메모리가 약 2~3GB 필요합니다.
- 코호트별 전체 상관값은 `results/corr_all_<GSE>.csv`에 저장됩니다. 다시 실행하면 이 파일을 재사용하고, 새로 계산하려면 `--refresh`를 붙입니다.
