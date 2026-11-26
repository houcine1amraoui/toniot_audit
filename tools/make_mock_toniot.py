"""Create a small fake TON_IoT folder tree (same layout & quirks) to test the code
without the real data:  python tools/make_mock_toniot.py /tmp/mock_toniot"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
ATTACKS = ["backdoor", "ddos", "injection", "password", "ransomware", "scanning", "xss"]


def stamps(n):
    t0 = pd.Timestamp("2019-03-31 12:00:00")
    dt = rng.choice([1, 2, 5, 13], size=n).cumsum()
    dt[n // 2:] += 3 * 3600  # one natural gap
    t = t0 + pd.to_timedelta(dt, unit="s")
    return t.strftime("%d-%b-%y"), (" " + t.strftime("%H:%M:%S") + " ")


def types(n):
    ty = np.array(["normal"] * n, dtype=object)
    blocks = np.array_split(np.arange(int(n * 0.35), n), len(ATTACKS))
    for a, b in zip(ATTACKS, blocks):   # attacks in contiguous time windows
        ty[b] = np.where(rng.random(len(b)) < 0.7, a, "normal")
    return ty


def device(n, cols):
    d, t = stamps(n)
    ty = types(n)
    df = pd.DataFrame({"date": d, "time": t})
    att = ty != "normal"
    for c, kind in cols.items():
        if kind == "num":
            x = rng.normal(20, 5, n) + att * rng.normal(4, 2, n)
            df[c] = np.round(x, 3).astype(str)
        elif kind == "onoff":
            df[c] = np.where(rng.random(n) < 0.5 + 0.2 * att, " on", "off")
        elif kind == "bool_mixed":
            v = rng.random(n) < 0.5
            df[c] = np.where(rng.random(n) < 0.5, np.where(v, "true", "false"), np.where(v, "1", "0"))
    df.iloc[rng.choice(n, 20, replace=False), 2] = ""  # some empty sensor cells
    df["label"] = att.astype(int).astype(str)
    df["type"] = ty
    return df


def main(root):
    root = Path(root)
    specs = {
        "IoT_Fridge": {"fridge_temperature": "num", "temp_condition": "onoff"},
        "IoT_Garage_Door": {"door_state": "onoff", "sphone_signal": "bool_mixed"},
        "IoT_Thermostat": {"current_temperature": "num", "thermostat_status": "onoff"},
        "IoT_Weather": {"temperature": "num", "pressure": "num", "humidity": "num"},
    }
    p = root / "Processed_datasets" / "Processed_IoT_dataset"
    tt = root / "Train_Test_datasets" / "Train_Test_IoT_dataset"
    net = root / "Train_Test_datasets" / "Train_Test_Network_dataset"
    for d in (p, tt, net):
        d.mkdir(parents=True, exist_ok=True)
    for name, cols in specs.items():
        df = device(8000, cols)
        df.to_csv(p / f"{name}.csv", index=False)
        df.sample(3000, random_state=1).to_csv(tt / f"Train_Test_{name}.csv", index=False)
    pd.DataFrame({"src_ip": ["1.1.1.1"] * 100, "label": [0, 1] * 50,
                  "type": ["normal", "ddos"] * 50}).to_csv(net / "train_test_network.csv", index=False)
    (root / "Description_stats_datasets").mkdir(exist_ok=True)
    (root / "Description_stats_datasets" / "readme.txt").write_text("mock")
    (root / "Raw_datasets.zip").write_bytes(b"PK")
    print("mock written to", root)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mock_toniot")
