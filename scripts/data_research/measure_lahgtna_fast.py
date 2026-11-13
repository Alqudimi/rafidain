#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fast iq-subset estimation for Lahgtna v2 + v3-small:
exact test-split counts (few shards) + /rows sampling for durations."""
import json, os, statistics, subprocess, sys, urllib.request

import requests

HDR = {"User-Agent": "rafadain-fast/1.0"}
TMP = "/tmp/lahgtna_measure"
os.makedirs(TMP, exist_ok=True)
S = requests.Session()
S.headers.update(HDR)

import time as _t

def get_json(url, timeout=90, retries=4):
    last = None
    for a in range(retries):
        try:
            req = urllib.request.Request(url, headers=HDR)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (429, 500, 502, 503, 504):
                _t.sleep(3 + 3 * a)
                continue
            raise
        except Exception as e:
            last = e
            _t.sleep(2 + 2 * a)
    raise last

def sniff(data):
    if data[:4] == b"RIFF": return ".wav"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0): return ".mp3"
    if data[:4] == b"OggS": return ".ogg"
    if data[:4] == b"fLaC": return ".flac"
    if data[:4] == b"\x1aE\xdf\xa3": return ".webm"
    if data[4:8] == b"ftyp": return ".m4a"
    return ".bin"

def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                           "format=duration:stream=sample_rate,channels",
                           "-of", "json", path], capture_output=True, text=True, timeout=60)
    j = json.loads(out.stdout)
    st = (j.get("streams") or [{}])[0]
    return float(j["format"]["duration"]), st.get("sample_rate"), st.get("channels")

import pyarrow.parquet as pq

def count_test_exact(ds):
    """count languages on the TEST split only (small shard count)"""
    parq = get_json(f"https://datasets-server.huggingface.co/parquet?dataset={ds}")
    test_urls = [f["url"] for f in parq["parquet_files"] if f["split"] == "test"]
    agg = {}
    total = 0
    for u in test_urls:
        p = "/tmp/_t.parquet"
        with open(p, "wb") as f:
            f.write(S.get(u, timeout=900).content)
        pf = pq.ParquetFile(p)
        col = pf.read(columns=["language"]).column("language").to_pylist()
        for k, v in __import__("collections").Counter(col).items():
            agg[k] = agg.get(k, 0) + v
        total += len(col)
    return agg, total, len(test_urls)

def rows_total(ds, split):
    r = get_json(f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split={split}&offset=0&length=1")
    return r.get("num_rows_total", 0)

def sample_iq_durations(ds, split, n_pages=30, page_len=100):
    total = rows_total(ds, split)
    if not total:
        return None, 0
    offsets = [int(i * (total - page_len) / max(n_pages - 1, 1)) for i in range(n_pages)]
    picked, seen = [], 0
    langs_seen = {}
    page_fail = 0
    for off in offsets:
        try:
            rr = get_json(f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split={split}&offset={off}&length={page_len}")
        except Exception:
            page_fail += 1
            _t.sleep(2)
            continue
        for rw in rr.get("rows", []):
            row = rw.get("row", {})
            lg = row.get("language")
            seen += 1
            if lg:
                langs_seen[lg] = langs_seen.get(lg, 0) + 1
            if lg == "iq":
                aud = row.get("audio")
                if isinstance(aud, list) and aud:
                    aud = aud[0]
                if isinstance(aud, dict) and aud.get("src"):
                    picked.append((aud["src"], row.get("transcript_text", "")))
    durs, srs, fmts, texts = [], {}, {}, []
    for i, (src, tx) in enumerate(picked[:120]):
        try:
            data = S.get(src, timeout=240).content
            ext = sniff(data)
            if ext == ".bin":
                continue
            p = os.path.join(TMP, f"{ds.split('/')[-1][:8]}_{i:03d}{ext}")
            open(p, "wb").write(data)
            d, sr, ch = probe(p)
            durs.append(d)
            srs[sr] = srs.get(sr, 0) + 1
            fmts[ext] = fmts.get(ext, 0) + 1
            if len(texts) < 8 and tx:
                texts.append(tx)
        except Exception:
            pass
    info = {"rows_sampled": seen, "iq_picked": len(picked), "iq_probed": len(durs),
            "langs_in_sample": langs_seen, "failed_pages": page_fail}
    if durs:
        info.update({"mean_dur": round(statistics.mean(durs), 2),
                     "median_dur": round(statistics.median(durs), 2),
                     "min_dur": round(min(durs), 2), "max_dur": round(max(durs), 2),
                     "sr": srs, "fmt": fmts, "channels": ch,
                     "transcripts": texts[:8]})
    return info, total

out = {}
for ds in (sys.argv[1:] or ["oddadmix/lahgtna-v3-small"]):
    print(f"\n### {ds}")
    agg, tot_test, n_shards = count_test_exact(ds)
    iq_test = agg.get("iq", 0)
    print(f"  TEST exact: {tot_test:,} rows over {n_shards} shards | iq={iq_test} ({iq_test*100.0/max(tot_test,1):.2f}%)")
    for k, v in sorted(agg.items(), key=lambda x: -x[1]):
        print(f"     {k:4s}: {v:>7,}  ({v*100.0/max(tot_test,1):.1f}%)")
    res, n_train = sample_iq_durations(ds, "train")
    share = iq_test / max(tot_test, 1)
    est_rows = round(share * n_train)
    print(f"  TRAIN total rows: {n_train:,} | est iq rows (test share {share*100:.2f}%): {est_rows:,}")
    if res:
        print(f"  TRAIN iq duration sample: n={res.get('iq_probed')} | mean {res.get('mean_dur')}s | "
              f"median {res.get('median_dur')}s | sr={res.get('sr')} | fmt={res.get('fmt')}")
        for t in (res.get("transcripts") or [])[:4]:
            print("     *", t[:110])
        est_h = round(res["mean_dur"] * est_rows / 3600, 1)
        print(f"  >>> EST iq-train hours: {est_h} h  (rows {est_rows:,} x mean {res['mean_dur']}s)")
        out[ds] = {"test_exact": {"total": tot_test, "langs": agg},
                   "train_rows_total": n_train, "est_iq_train_rows": est_rows,
                   "duration_sample": res, "est_iq_train_hours": est_h}
    else:
        out[ds] = {"test_exact": {"total": tot_test, "langs": agg}, "train_rows_total": n_train}

json.dump(out, open("/tmp/lahgtna_v3small.json", "w"), indent=2, ensure_ascii=False)
print("\nsaved /tmp/lahgtna_v3small.json")
