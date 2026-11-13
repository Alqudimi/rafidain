#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact language counts (parquet column reads via ranged requests) for
oddadmix/dialectal-arabic-lahgtna-v2 and oddadmix/lahgtna-v3-small,
plus iq duration sampling via /rows audio assets."""
import io, json, os, subprocess, statistics, sys, threading, time
from collections import Counter
import urllib.request

import pyarrow.parquet as pq
import requests

HDR = {"User-Agent": "rafadain-measure/1.0"}
TMP = "/tmp/lahgtna_measure"
os.makedirs(TMP, exist_ok=True)

_tls = threading.local()

def sess():
    if not hasattr(_tls, "s"):
        s = requests.Session()
        s.headers.update(HDR)
        _tls.s = s
    return _tls.s

def get_json(url, timeout=90):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

class RangedFile(io.RawIOBase):
    """seekable read-only view of a remote file using HTTP Range on a Session."""
    def __init__(self, url):
        self.url = url
        r = sess().head(url, timeout=30, allow_redirects=True)
        r.raise_for_status()
        self.size = int(r.headers.get("Content-Length") or 0)
        self.pos = 0

    def seekable(self): return True
    def readable(self): return True
    def seek(self, off, whence=0):
        if whence == 0: self.pos = off
        elif whence == 2: self.pos = self.size + off
        else: self.pos += off
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.size - self.pos
        if n <= 0 or self.pos >= self.size: return b""
        end = min(self.pos + n, self.size) - 1
        for a in range(3):
            try:
                r = sess().get(self.url, headers={"Range": f"bytes={self.pos}-{end}"}, timeout=120)
                data = r.content
                self.pos += len(data)
                return data
            except Exception:
                time.sleep(1 + a)
        raise IOError(f"range read failed {self.url} @{self.pos}")

def count_shard(url):
    f = RangedFile(url)
    try:
        pf = pq.ParquetFile(f)
        return Counter(pf.read(columns=["language"]).column("language").to_pylist())
    finally:
        f.close()

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

def count_dataset(ds):
    parq = get_json(f"https://datasets-server.huggingface.co/parquet?dataset={ds}")
    shards = {}
    for f in parq["parquet_files"]:
        shards.setdefault(f["split"], []).append((f["url"], f.get("size", 0)))
    print(f"\n### {ds}")
    for sp, fl in shards.items():
        print(f"  {sp}: {len(fl)} shards, {sum(s for _, s in fl)/1e9:.2f} GB")
    out = {}
    for sp, fl in shards.items():
        agg = Counter()
        from concurrent.futures import ThreadPoolExecutor, as_completed
        nfail = 0
        with ThreadPoolExecutor(max_workers=12) as ex:
            futs = [ex.submit(count_shard, u) for u, _ in fl]
            done = 0
            for fut in as_completed(futs):
                done += 1
                try:
                    agg.update(fut.result())
                except Exception as e:
                    nfail += 1
                    if nfail <= 2:
                        print(f"    shard FAIL: {type(e).__name__}: {e}")
                if done % 80 == 0:
                    print(f"    {sp}: {done}/{len(fl)} shards")
        tot = sum(agg.values())
        print(f"  == {sp} language distribution ({tot:,} rows, {nfail} failed shards):")
        for k, v in sorted(agg.items(), key=lambda x: -x[1]):
            print(f"     {k:6s}: {v:>9,}  ({v*100.0/max(tot,1):.1f}%)")
        out[sp] = {"rows": tot, "langs": dict(agg)}
    return out

def sample_durations(ds, split, n_pages=25, page_len=100):
    """page /rows at spread offsets; download audio of iq rows; ffprobe"""
    r0 = get_json(f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split={split}&offset=0&length=1")
    total = r0.get("num_rows_total", 0)
    if not total:
        return None
    offsets = [int(i * (total - page_len) / max(n_pages - 1, 1)) for i in range(n_pages)]
    picked = []
    for off in offsets:
        rr = get_json(f"https://datasets-server.huggingface.co/rows?dataset={ds}&config=default&split={split}&offset={off}&length={page_len}")
        for rw in rr.get("rows", []):
            row = rw.get("row", {})
            if row.get("language") == "iq":
                aud = row.get("audio")
                if isinstance(aud, list) and aud:
                    aud = aud[0]
                if isinstance(aud, dict) and aud.get("src"):
                    picked.append((aud["src"], row.get("transcript_text", "")))
    durs, srs, fmts = [], {}, {}
    texts = []
    for i, (src, tx) in enumerate(picked[:130]):
        try:
            data = sess().get(src, timeout=180).content
            ext = sniff(data)
            if ext == ".bin":
                continue
            p = os.path.join(TMP, f"{ds.split('/')[-1]}_{i:03d}{ext}")
            open(p, "wb").write(data)
            d, sr, ch = probe(p)
            durs.append(d)
            srs[sr] = srs.get(sr, 0) + 1
            fmts[ext] = fmts.get(ext, 0) + 1
            if len(texts) < 8 and tx:
                texts.append(tx)
        except Exception as e:
            pass
    res = {"sample_rows_seen": n_pages * page_len, "iq_picked": len(picked),
           "iq_probed": len(durs)}
    if durs:
        res.update({
            "mean_dur": round(statistics.mean(durs), 2),
            "median_dur": round(statistics.median(durs), 2),
            "min_dur": round(min(durs), 2), "max_dur": round(max(durs), 2),
            "sr": srs, "fmt": fmts, "transcripts": texts[:8],
        })
        print(f"  duration sample ({split}): n={len(durs)} | mean {res['mean_dur']}s | median {res['median_dur']}s | "
              f"min {res['min_dur']}s max {res['max_dur']}s | sr={srs} | fmt={fmts}")
        for t in texts[:4]:
            print("    *", t[:110])
    return res

result = {}
for ds in ("oddadmix/dialectal-arabic-lahgtna-v2", "oddadmix/lahgtna-v3-small"):
    info = count_dataset(ds)
    iq_train = info.get("train", {}).get("langs", {}).get("iq", 0)
    iq_test = info.get("test", {}).get("langs", {}).get("iq", 0)
    dur = sample_durations(ds, "train")
    est = None
    if dur and dur.get("mean_dur") and iq_train:
        est = round(dur["mean_dur"] * iq_train / 3600, 1)
        print(f"  >>> ESTIMATED iq train hours for {ds}: {est} h  ({iq_train:,} rows x {dur['mean_dur']}s)")
    result[ds] = {"counts": info, "duration_sample": dur, "est_iq_train_hours": est}

json.dump(result, open("/tmp/lahgtna_measure.json", "w"), indent=2, ensure_ascii=False)
print("\nsaved /tmp/lahgtna_measure.json")
