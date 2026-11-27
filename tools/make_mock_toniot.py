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


# ---------------------------------------------------------------- network ---
NET_COLS = ("ts,src_ip,src_port,dst_ip,dst_port,proto,service,duration,src_bytes,dst_bytes,conn_state,"
            "missed_bytes,src_pkts,src_ip_bytes,dst_pkts,dst_ip_bytes,dns_query,dns_qclass,dns_qtype,"
            "dns_rcode,dns_AA,dns_RD,dns_RA,dns_rejected,ssl_version,ssl_cipher,ssl_resumed,"
            "ssl_established,ssl_subject,ssl_issuer,http_trans_depth,http_method,http_uri,http_referrer,"
            "http_version,http_request_body_len,http_response_body_len,http_status_code,http_user_agent,"
            "http_orig_mime_types,http_resp_mime_types,weird_name,weird_addl,weird_notice,label,type").split(",")
NET_TYPES = ["normal", "backdoor", "ddos", "dos", "injection", "password", "ransomware", "scanning", "xss", "mitm"]


def net_rows(n, t0):
    ty = rng.choice(NET_TYPES, n, p=[.5, .06, .08, .08, .06, .06, .04, .08, .03, .01])
    att = ty != "normal"
    d = {c: np.array(["-"] * n, dtype=object) for c in NET_COLS}
    order = {t: i for i, t in enumerate(NET_TYPES)}
    d["ts"] = (t0 + np.array([order[t] for t in ty]) * 3600 * 6 * att + rng.integers(0, 20000, n)).astype(str)
    d["src_ip"] = np.where(att, "192.168.1." + pd.Series(ty).map(order).astype(str), "10.0.0." + rng.integers(1, 20, n).astype(str))
    d["src_port"] = rng.integers(1024, 65000, n).astype(str)
    d["dst_ip"] = "192.168.1." + rng.integers(1, 30, n).astype(str)
    svc = np.where(ty == "scanning", "-", rng.choice(["dns", "http", "ssl", "-"], n))
    d["dst_port"] = pd.Series(svc).map({"dns": "53", "http": "80", "ssl": "443", "-": "0"}).fillna("0").to_numpy()
    d["proto"] = np.where(svc == "dns", "udp", "tcp")
    d["service"] = svc
    base = pd.Series(ty).map(order).to_numpy()
    for c in ["duration", "src_bytes", "dst_bytes", "src_pkts", "dst_pkts", "src_ip_bytes", "dst_ip_bytes"]:
        d[c] = np.round(rng.exponential(1 + base * 3, n), 3).astype(str)
    d["missed_bytes"] = np.zeros(n, int).astype(str)
    d["conn_state"] = np.where(ty == "scanning", "REJ", rng.choice(["SF", "S0", "OTH"], n))
    dns = svc == "dns"
    d["dns_query"] = np.where(dns, rng.choice(["a.com", "b.org", "evil.x"], n), "-")
    for c in ["dns_qclass", "dns_qtype", "dns_rcode"]:
        d[c] = np.where(dns, rng.integers(0, 3, n).astype(str), "0")
    for c in ["dns_AA", "dns_RD", "dns_RA", "dns_rejected"]:
        d[c] = np.where(dns, rng.choice(["T", "F"], n), "-")
    ssl = svc == "ssl"
    d["ssl_version"] = np.where(ssl, "TLSv12", "-")
    d["ssl_cipher"] = np.where(ssl, rng.choice(["A", "B"], n), "-")
    http = svc == "http"
    d["http_method"] = np.where(http, np.where(ty == "xss", "POST", "GET"), "-")
    d["http_uri"] = np.where(http, np.where(ty == "injection", "/q?id=1'", "/"), "-")
    for c in ["http_trans_depth", "http_request_body_len", "http_response_body_len", "http_status_code"]:
        d[c] = np.where(http, rng.integers(0, 500, n).astype(str), "0")
    d["label"] = att.astype(int).astype(str)
    d["type"] = ty
    return pd.DataFrame(d)[NET_COLS]


def make_network(root, n_files=3, rows=40000):
    root = Path(root)
    p = root / "Processed_datasets" / "Processed_Network_dataset"
    p.mkdir(parents=True, exist_ok=True)
    t0 = 1554000000
    for i in range(1, n_files + 1):
        df = net_rows(rows, t0 + i * 86400)
        if i == 2:
            df["extra_col"] = "x"  # mimic Network_dataset_6 extra column
        df.to_csv(p / f"Network_dataset_{i}.csv", index=False, encoding="utf-8-sig")
    tt = root / "Train_Test_datasets" / "Train_Test_Network_dataset"
    tt.mkdir(parents=True, exist_ok=True)
    src = pd.read_csv(p / "Network_dataset_1.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    src.drop(columns=["ts", "http_referrer"]).sample(5000, random_state=0).to_csv(
        tt / "train_test_network.csv", index=False, encoding="utf-8-sig")
    print("mock network written to", root)


if __name__ == "__main__" and len(sys.argv) > 2 and sys.argv[2] == "--network":
    make_network(sys.argv[1])
