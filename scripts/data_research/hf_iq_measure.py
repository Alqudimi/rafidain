#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure the Iraqi subset of Lahgtna v2 + MADIS5 via /filter + audio sampling."""
import json, os, subprocess, urllib.request, urllib.parse, statistics

HDR = {"User-Agent": "rafidain-iq/1.0"}
TMP = "/tmp/iq_probe"
os.makedirs(TMP, exist_ok=True)

def get(url, timeout=90):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())

def frows(ds, cfg, split, where, limit=100, offset=0):
    q = urllib.parse.urlencode({
        "dataset": ds, "config": cfg, "split": split,
        "where": where, "limit": limit, "offset": offset})
    return get(f"https://datasets-server.huggingface.co/filter?{q}")

def count(ds, cfg, split, where):
    r = frows(ds, cfg, split, where, limit=1)
    return r.get("num_rows_total", 0), r

def sniff(data):
    if data[:4] == b"RIFF": return ".wav"
    if data[:3] == b"ID3" or (len(data) > 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0): return ".mp3"
    if data[:4] == b"OggS": return ".ogg"
    if data[:4] == b"fLaC": return ".flac"
    if data[:4] == b"\x1aE\xdf\xa3": return ".webm"
    if data[4:8] == b"ftyp": return ".m4a"
    return ".bin"

def dl(url, path):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
        f.write(r.read())
    return path

def probe(path):
    try:
        out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                               "format=duration:stream=sample_rate,channels",
                               "-of", "json", path], capture_output=True, text=True, timeout=60)
        j = json.loads(out.stdout)
        dur = float(j["format"]["duration"])
        sr = j["streams"][0].get("sample_rate") if j.get("streams") else None
        return dur, sr
    except Exception as e:
        return None, None

# ---------- 1) Lahgtna v2 : Iraqi subset ----------
DS = "oddadmix/dialectal-arabic-lahgtna-v2"
print("=" * 78)
print("LAHGtna v2 — Iraqi subset")
tot = {}
for split in ("train", "test"):
    n, sample = count(DS, "default", split, "\"language\"='iq'")
    tot[split] = n
    print(f"  {split}: iq rows = {n:,}")
    if n and split == "train" and not sample.get("rows"):
        pass
# language distribution across all 13 codes (counts via filter num_rows_total)
print("  full language distribution (train):")
for code in ("iq","eg","sa","ma","tn","dz","ly","sd","ye","ps","lb","bh","kw","sy","jo","ae","qa","om","hr","msa","ar"):
    try:
        n, _ = count(DS, "default", "train", f"\"language\"='{code}'")
        if n:
            print(f"    {code:4s}: {n:,}")
    except Exception as e:
        pass
# sample durations for iq train
r = frows(DS, "default", "train", "\"language\"='iq'", limit=100)
rows = r.get("rows", [])
print(f"  sampling {len(rows)} iq-train rows for durations ...")
durs, srs, exts = [], {}, {}
for i, rw in enumerate(rows):
    row = rw.get("row", {})
    aud = row.get("audio")
    if isinstance(aud, list) and aud: aud = aud[0]
    if isinstance(aud, dict) and aud.get("src"):
        try:
            p = os.path.join(TMP, f"s{i:03d}")
            dl(aud["src"], p + ".tmp")
            with open(p + ".tmp", "rb") as f: head = f.read(16)
            p2 = p + sniff(head)
            os.rename(p + ".tmp", p2)
            d, sr = probe(p2)
            if d:
                durs.append(d)
                srs[sr] = srs.get(sr, 0) + 1
                exts[os.path.splitext(p2)[1]] = exts.get(os.path.splitext(p2)[1], 0) + 1
        except Exception as e:
            print(f"    row {i}: {type(e).__name__}: {e}")
if durs:
    n_train = tot.get("train", 0)
    mean = statistics.mean(durs)
    med = statistics.median(durs)
    tot_secs = mean * n_train
    print(f"  sampled {len(durs)} clips | mean dur {mean:.1f}s | median {med:.1f}s | min {min(durs):.1f} max {max(durs):.1f}")
    print(f"  sample-rate histogram: {srs}")
    print(f"  format histogram: {exts}")
    print(f"  ESTIMATED iq-train hours: {tot_secs/3600:.1f} h over {n_train:,} rows")
    # transcript sanity: print one transcript
    t0 = rows[0]["row"].get("transcript_text", "")
    print(f"  sample transcript: {t0[:120]}")

# ---------- 2) MADIS5 : Iraqi subset ----------
print("=" * 78)
print("MADIS5 — Iraqi subset")
DS2 = "badrex/MADIS5-spoken-arabic-dialects"
for dv in ("Iraqi Arabic", "Iraqi", "iq"):
    try:
        n, _ = count(DS2, "default", "test", f"\"dialect\"='{dv}'")
        print(f"  dialect='{dv}': {n:,}")
        if n:
            break
    except Exception as e:
        print(f"  dialect='{dv}': err {e}")
# page through Iraqi rows to sum audio_duration exactly
n_iq = 0
for dv in ("Iraqi Arabic", "Iraqi"):
    try:
        n_iq, _ = count(DS2, "default", "test", f"\"dialect\"='{dv}'")
        if n_iq: break
    except Exception: pass
if n_iq:
    total_dur, seen, domains, dur_list = 0.0, 0, {}, []
    off = 0
    while off < n_iq:
        rr = frows(DS2, "default", "test", f"\"dialect\"='{dv}'", limit=100, offset=off)
        for rw in rr.get("rows", []):
            row = rw.get("row", {})
            d = row.get("audio_duration")
            if d:
                total_dur += d
                dur_list.append(d)
                dom = row.get("domain", "?")
                domains[dom] = domains.get(dom, 0) + 1
            seen += 1
        off += 100
        if off > 2000: break
    if dur_list:
        print(f"  Iraqi rows paged: {seen} | sum duration {total_dur/3600:.2f} h | mean {statistics.mean(dur_list):.1f}s")
        print(f"  domains: {domains}")
        print(f"  EXACT Iraqi hours in MADIS5: {total_dur/3600:.2f} h")
    # sample one audio for sr/format
    rr = frows(DS2, "default", "test", f"\"dialect\"='{dv}'", limit=5)
    for rw in rr.get("rows", []):
        aud = rw["row"].get("audio")
        if isinstance(aud, list) and aud: aud = aud[0]
        if isinstance(aud, dict) and aud.get("src"):
            try:
                p = os.path.join(TMP, "madis_iq")
                dl(aud["src"], p + ".tmp")
                with open(p + ".tmp", "rb") as f: head = f.read(16)
                os.rename(p + ".tmp", p + sniff(head))
                d, sr = probe(p + sniff(head))
                print(f"  sample clip: dur={d}s sr={sr} fmt={sniff(head)}")
            except Exception as e:
                print("  sample failed:", e)
            break

# ---------- 3) MADIS5 full dialect distribution ----------
print("=" * 78)
print("MADIS5 — full dialect distribution (test)")
for dv in ("Iraqi Arabic","Egyptian Arabic","Gulf Arabic","Moroccan Arabic","Algerian Arabic",
           "Tunisian Arabic","Libyan Arabic","Levantine Arabic","Saudi Arabic","Yemeni Arabic",
           "Sudanese Arabic","Jordanian Arabic","Palestinian Arabic","MSA","Modern Standard Arabic"):
    try:
        n, _ = count(DS2, "default", "test", f"\"dialect\"='{dv}'")
        if n: print(f"    {dv:24s}: {n:,}")
    except Exception: pass
