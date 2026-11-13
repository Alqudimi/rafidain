#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe candidate HF datasets for Iraqi speech: files, sizes, rows, columns."""
import json, sys, urllib.request, urllib.parse

HDR = {"User-Agent": "rafidain-probe/1.0"}

def get(url, timeout=40):
    req = urllib.request.Request(url, headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode()), r.status
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode()), e.code
        except Exception:
            return None, e.code
    except Exception as e:
        return {"_err": str(e)}, 0

CANDIDATES = [
    "otozz/iraqi_train_set",
    "otozz/iraqi_test_set",
    "hnaalbhadly6/iraqi_arabic_speech_youssef",
    "karimBD/All_WSP_Iraqi_shiekh_hara_sawtarabi_MediaSpeech",
    "karimBD/Nemo_Iraqi_shiekh_hara_sawtarabi_MediaSpeech",
    "ArSyra/arsyra-iraqi",
    "oddadmix/dialectal-arabic-lahgtna-v2",
    "Falah/classification_arabic_dialects",
    "badrex/MADIS5-spoken-arabic-dialects",
    "ashabrawy/did-Baghdad",
    "moaead/dialectal-arabic-voices",
    "echonova-ai/arabic-dialect-asr-benchmark",
    "safaa94/Myvoice-iraqi-accent-samples",
    "xtz999/iraqi_arabic_speech_youssef",
]

def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or u == "TB":
            return f"{n:.0f}{u}"
        n /= 1000

for ds in CANDIDATES:
    print("=" * 78)
    print("DS:", ds)
    info, st = get(f"https://huggingface.co/api/datasets/{ds}")
    if not isinstance(info, dict) or "_err" in info:
        print("  ! unavailable:", st, info if not isinstance(info, dict) else info.get("_err"))
        continue
    print(f"  downloads={info.get('downloads')} likes={info.get('likes')} "
          f"private={info.get('private')} gated={info.get('gated')} "
          f"lastModified={str(info.get('lastModified'))[:10]}")
    card = info.get("cardData") or {}
    if card:
        keep = {k: card[k] for k in ("size_categories", "license", "language",
                                      "configs", "pretty_name") if k in card}
        print("  card:", json.dumps(keep, ensure_ascii=False)[:300])
    sib = info.get("siblings") or []
    audio_ext = (".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm", ".tar", ".zip", ".parquet")
    files = [s["rfilename"] for s in sib]
    audio_files = [f for f in files if f.lower().endswith(audio_ext)]
    others = [f for f in files if not f.lower().endswith(audio_ext)]
    print(f"  files total={len(files)} audio/archive/parquet={len(audio_files)}")
    for f in audio_files[:12]:
        print(f"    A {f}")
    if len(audio_files) > 12:
        print(f"    ... +{len(audio_files)-12} more")
    for f in others[:8]:
        print(f"    - {f}")
    # datasets-server: size + first rows (audio/text columns)
    sz, _ = get(f"https://datasets-server.huggingface.co/size?dataset={ds}")
    if isinstance(sz, dict) and sz.get("size"):
        for cfgname, cfg in list(sz["size"].get("dataset", {}).items())[:3]:
            if not isinstance(cfg, dict):
                continue
            print(f"  config '{cfgname}': rows={cfg.get('num_rows')} "
                  f"orig={human(cfg.get('num_bytes_original_files', 0))} "
                  f"parquet={human(cfg.get('num_bytes_parquet_files', 0))}")
    else:
        err = (sz or {}).get("error", "")[:80] if isinstance(sz, dict) else ""
        print("  size endpoint:", err or "no data")
    parq, _ = get(f"https://datasets-server.huggingface.co/parquet?dataset={ds}")
    if isinstance(parq, dict) and parq.get("parquet_files"):
        conf = parq["parquet_files"][0].get("config", "?")
        split = parq["parquet_files"][0].get("split", "?")
        print(f"  parquet config={conf} split={split}")
        rows, _ = get(f"https://datasets-server.huggingface.co/first-rows?dataset={ds}&config={urllib.parse.quote(conf)}&split={urllib.parse.quote(split)}")
        if isinstance(rows, dict) and rows.get("rows"):
            r0 = rows["rows"][0].get("row", {})
            for k, v in r0.items():
                if isinstance(v, dict) and "bytes" in v:
                    print(f"    col '{k}': AUDIO ({len(v.get('bytes') or b'')} b, name={v.get('path')})")
                elif isinstance(v, list):
                    print(f"    col '{k}': {type(v[0]).__name__} list len={len(v)} -> {str(v)[:90]}")
                else:
                    print(f"    col '{k}': {str(v)[:90]}")
        elif isinstance(rows, dict):
            print("    first-rows:", str(rows.get("error"))[:100])
