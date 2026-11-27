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

## Network-flow audit (`--part network`)

The missing-data study runs on TON_IoT **network flows** (`Processed_Network_dataset`, 23 CSVs, ~22M Zeek flows,
plus `train_test_network.csv`). The IoT telemetry audit showed that its labels follow attack time windows, while its
sensor values carry almost no attack signal.

```python
!python scripts/run_audit.py --data-root "$DATA_ROOT" --out-dir "$OUT_DIR" --part network --net-max-files 2   # quick test
!python scripts/run_audit.py --data-root "$DATA_ROOT" --out-dir "$OUT_DIR" --part network                     # full
```

One streaming pass (500k-row chunks, memory-safe) + probes on a uniform per-class sample (`--net-sample-per-class 20000`).
Outputs in `OUT_DIR/network/{processed,train_test}/`:

| File | Content |
|---|---|
| `NETWORK_REPORT.md`, `network_checks.csv` | checks N1–N11 (schema, labels, class sizes, signal, ID/time leakage, chronological generalisation, conflicts, group-loss sensitivity, multi-day classes, source-IP leakage) |
| `class_time_windows.csv`, `class_by_day.csv/.png` | when each class occurs |
| `column_missingness.csv`, `group_presence_pct_by_class.csv/.png` | Zeek `-` **structural** missingness by column, group and class |
| `group_ablation.csv` | macro-F1 when conn / dns / ssl / http / weird groups are removed = preview of missing-data impact |
| `feature_importance_behaviour.csv` | most used behavioural features |
| `src_ip_purity.csv`, `dst_ip_purity.csv` | IP-based leakage |
| `network_summary.json` | duplicates, probes, train_test overlap with processed |

Feature groups: `id` (IPs, ports), `time` (ts), `conn` (proto, service, bytes, packets, state…), `dns_*`, `ssl_*`, `http_*`, `weird_*`.

## Test without the real data

```bash
python tools/make_mock_toniot.py /tmp/mock_toniot --network
python scripts/run_audit.py --data-root /tmp/mock_toniot --out-dir /tmp/audit_out
```

## Layout

```
src/toniot_study/   config, inventory, loaders, audit, report, pipeline
scripts/run_audit.py
notebooks/01_inventory_audit.ipynb
tools/make_mock_toniot.py
```
