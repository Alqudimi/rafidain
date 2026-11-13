#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure Lahgtna v2 'iq' subset: row counts, sampled durations -> hours estimate."""
import json, os, subprocess, statistics, sys, urllib.request
import pyarrow.parquet as pq

HDR = {"User-Agent": "rafadain-lahgtna/1.0"}
TMP = "/tmp/lahgtna_iq"
os.makedirs(TMP, exist_ok=True)

DS = "oddadmix/dialectal-arabic-lahgtna-v2"

def get(url, timeout=90):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def sniff(data):
    if data[:4] == b"RIFF": return ".wav"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0): return ".mp3"
    if data[:4] == b"OggS": return ".ogg"
    if data[:4] == b"fLaC": return ".flac"
    if data[:4] == b"\x1aE\xdf\xa3": return ".webm"
    if data[4:8] == b"ftyp": return ".m4a"
    return ".bin"

def open_parquet(url):
    """seekable HTTP file via fsspec https, or plain download fallback"""
    try:
        import fsspec
        fs = fsspec.filesystem("https")
        return fs.open(url, "rb"), "fsspec"
    except Exception:
        return None, "dl"

# file inventory
parq = get("https://datasets-server.huggingface.co/parquet?dataset=" + DS)
files = {}
for f in parq["parquet_files"]:
    files.setdefault(f["split"], []).append((f["url"], 0, f.get("size", 0)))
for sp, fl in files.items():
    print(f"{sp}: {len(fl)} shards, rows={sum(r for _,r,_ in fl):,}, bytes={sum(s for _,_,s in fl)/1e9:.1f}GB")

def url_for(split, i, n):
    return f"https://huggingface.co/datasets/{DS}/resolve/main/data/{split}-{i:05d}-of-{n:05d}.parquet"

train_n = len(files.get("train", []))
test_n = len(files.get("test", []))

# --- pass 1: language counts per shard (column-only reads) ---
def count_shard(url):
    fh, mode = open_parquet(url)
    if fh is None:
        path = "/tmp/_shard.parquet"
        req = urllib.request.Request(url, headers=HDR)
        with urllib.request.urlopen(req, timeout=600) as r, open(path, "wb") as f:
            f.write(r.read())
        fh = open(path, "rb")
    try:
        pf = pq.ParquetFile(fh, )
        tbl = pf.read(columns=["language"])
        col = tbl.column("language").to_pylist()
        from collections import Counter
        return Counter(col), len(col)
    finally:
        fh.close()

SHARDS_TRAIN = files["train"]
print("counting languages across train shards (8 threads, range reads) ...")
from concurrent.futures import ThreadPoolExecutor, as_completed
lang_tot = {}
failed = 0

def _one(u):
    cnt, n = count_shard(u)
    return cnt, n

with ThreadPoolExecutor(max_workers=8) as ex:
    futs = {ex.submit(_one, u): i for i, (u, _, _) in enumerate(SHARDS_TRAIN)}
    done = 0
    for fut in as_completed(futs):
        done += 1
        try:
            cnt, n = fut.result()
            for k, v in cnt.items():
                lang_tot[k] = lang_tot.get(k, 0) + v
        except Exception as e:
            failed += 1
            if failed <= 3:
                print(f"  shard FAIL: {type(e).__name__}: {e}")
        if done % 60 == 0:
            print(f"  ... {done}/{len(SHARDS_TRAIN)} shards")

tot_rows = sum(lang_tot.values())
tot_iq = lang_tot.get("iq", 0)
if failed:
    print(f"  {failed} shards failed (counts above cover the rest)")

print("TRAIN language distribution:")
for k, v in sorted(lang_tot.items(), key=lambda x: -x[1]):
    print(f"    {k:6s}: {v:>9,}  ({v*100.0/max(tot_rows,1):.1f}%)")
print(f"TRAIN iq rows total: {tot_iq:,} of {tot_rows:,}")

# test split
tot_iq_test = 0
for u, nrows, size in files.get("test", []):
    try:
        cnt, n = count_shard(u)
        tot_iq_test += cnt.get("iq", 0)
    except Exception as e:
        print(f"  test shard FAIL: {e}")
print(f"TEST iq rows: {tot_iq_test:,}")

# --- pass 2: sample iq rows from first shards -> durations ---
print("\nsampling iq audio for duration estimate ...")
durs, srs, fmts, texts = [], {}, {}, []
sampled = 0
TARGET = 150
for u, nrows, size in SHARDS_TRAIN[:6]:
    if sampled >= TARGET:
        break
    fh, mode = open_parquet(u)
    if fh is None:
        continue
    try:
        pf = pq.ParquetFile(fh)
        for batch in pf.iter_batches(batch_size=64, columns=["audio", "language", "transcript_text"]):
            langs = batch.column("language").to_pylist()
            auds = batch.column("audio").to_pylist()
            txts = batch.column("transcript_text").to_pylist()
            for lg, au, tx in zip(langs, auds, txts):
                if lg != "iq" or sampled >= TARGET:
                    continue
                if not isinstance(au, dict):
                    continue
                data = au.get("bytes")
                if not data:
                    continue
                ext = sniff(data)
                if ext == ".bin":
                    continue
                p = os.path.join(TMP, f"a{sampled:04d}{ext}")
                open(p, "wb").write(data)
                try:
                    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                                           "format=duration:stream=sample_rate",
                                           "-of", "json", p], capture_output=True, text=True, timeout=60)
                    j = json.loads(out.stdout)
                    d = float(j["format"]["duration"])
                    sr = j["streams"][0].get("sample_rate") if j.get("streams") else "?"
                    durs.append(d); srs[sr] = srs.get(sr, 0) + 1; fmts[ext] = fmts.get(ext, 0) + 1
                    if len(texts) < 6 and tx:
                        texts.append(tx)
                    sampled += 1
                except Exception:
                    pass
    finally:
        fh.close()

if durs:
    mean = statistics.mean(durs)
    med = statistics.median(durs)
    print(f"sampled {len(durs)} iq clips | mean {mean:.1f}s median {med:.1f}s min {min(durs):.1f} max {max(durs):.1f}")
    print(f"sr histogram: {srs} | format histogram: {fmts}")
    est = mean * tot_iq / 3600
    est_lo = med * tot_iq / 3600
    print(f"ESTIMATED iq-train hours: {est:.1f} h (median-based: {est_lo:.1f} h) over {tot_iq:,} rows")
    print("sample transcripts:")
    for t in texts:
        print("   *", t[:140])
json.dump({"train_iq_rows": tot_iq, "test_iq_rows": tot_iq_test,
           "train_langs": lang_tot, "sampled_mean_dur": round(statistics.mean(durs), 2) if durs else None,
           "est_hours": round(statistics.mean(durs) * tot_iq / 3600, 1) if durs else None},
          open("/tmp/lahgtna_iq_summary.json", "w"), indent=2)
print("\nsaved /tmp/lahgtna_iq_summary.json")
