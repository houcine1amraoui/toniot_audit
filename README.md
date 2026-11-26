# TON_IoT – attack detection under missing sensor data

Step 1 of the study: **inventory** the TON_IoT dataset stored on Google Drive and
**audit** whether it is suitable for evaluating smart-home / IoT attack detection
when sensor data is missing.

## Run in Colab

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/<your-user>/toniot-missing-data.git
%cd toniot-missing-data
!pip install -q -r requirements.txt

DATA_ROOT = "/content/drive/MyDrive/TON_IoT"        # <- folder you uploaded
OUT_DIR   = "/content/drive/MyDrive/TON_IoT_audit"  # results are saved to Drive
!python scripts/run_audit.py --data-root "$DATA_ROOT" --out-dir "$OUT_DIR"
```

Or open `notebooks/01_inventory_audit.ipynb`.

Options: `--inventory-only`, `--no-train-test`, `--no-plots`,
`--max-rows-probe 200000`, `--min-samples-per-attack 500`,
`--count-rows-max-mb 0` (row-count every CSV; set e.g. 500 to skip huge network files).

> If your Drive copy holds `.zip`/`.rar` files, the inventory flags them. Extract them first
> (e.g. `!unzip -q file.zip -d /content/drive/MyDrive/TON_IoT/`).

## What it does

**Inventory** (`inventory_files.csv`, `inventory_summary.csv`): every file with
category (IoT processed / IoT train-test / network / Linux / Windows / pcap / archive / docs),
device, size, row count, and header.

**Audit** (per IoT telemetry file, for both `Processed_IoT_dataset` and `Train_Test_IoT_dataset`):

| Block | What is measured | Why it matters for the study |
|---|---|---|
| Column quality | empty cells, padded whitespace, raw vs clean cardinality, unparseable numbers, ranges | natural missingness and encoding quirks (e.g. `true/false` mixed with `1/0`) |
| Labels | `type`/`label` counts, consistency | class balance, enough samples per attack |
| Time | parse rate, sorting, sampling interval, duplicate timestamps, natural gaps | missingness already present; resampling choices |
| Segments & temporal mixing | contiguous runs of each attack; how much normal data falls in each attack window | time-leakage risk, split design |
| Duplicates | exact duplicates, same features with different labels | label noise / ambiguity |
| Separability probes | decision-tree macro-F1 with values only, time only, values+time, chronological split | is there sensor signal to lose? is time leaking the label? |
| Sensor ablation | macro-F1 when each sensor is dropped | preview of per-sensor missingness sensitivity |
| Cross-device alignment | shared seconds between devices, label agreement | feasibility of a merged multi-sensor view |
| Processed vs Train_Test | row overlap, class distribution shift | which version to use |

**Suitability checks C1–C11** → `suitability_checks.csv` with PASS / WARN / FAIL and evidence.

Main output: `AUDIT_REPORT.md` (plus `audit_full.json`, per-device CSVs and PNG plots).

## Test without the real data

```bash
python tools/make_mock_toniot.py /tmp/mock_toniot
python scripts/run_audit.py --data-root /tmp/mock_toniot --out-dir /tmp/audit_out
```

## Layout

```
src/toniot_study/   config, inventory, loaders, audit, report, pipeline
scripts/run_audit.py
notebooks/01_inventory_audit.ipynb
tools/make_mock_toniot.py
```
