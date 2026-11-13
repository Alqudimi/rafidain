#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Deep stats: per-language row counts + parquet sizes for candidate datasets."""
import json, urllib.request, urllib.parse

HDR = {"User-Agent": "rafidain-stats/1.0"}

def get(url, timeout=60):
    req = urllib.request.Request(url, headers=HDR)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return {"_http_err": e.code, "body": e.read().decode()[:200]}
        except Exception:
            return {"_http_err": e.code}
    except Exception as e:
        return {"_err": str(e)}

def human(n):
    n = float(n or 0)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if n < 1000 or u == "TB":
            return f"{n:.1f}{u}"
        n /= 1000

TARGETS = {
    "oddadmix/dialectal-arabic-lahgtna-v2": ["language"],
    "moaead/dialectal-arabic-voices": ["language", "source"],
    "badrex/MADIS5-spoken-arabic-dialects": ["dialect", "domain"],
    "hnaalbhadly6/iraqi_arabic_speech_youssef": [],
    "xtz999/iraqi_arabic_speech_youssef": [],
    "Falah/classification_arabic_dialects": ["label", "audio"],
}

for ds, cols in TARGETS.items():
    print("=" * 78)
    print("DS:", ds)
    # 1) parquet inventory (rows+bytes per split)
    parq = get(f"https://datasets-server.huggingface.co/parquet?dataset={ds}")
    if parq.get("parquet_files"):
        agg = {}
        for f in parq["parquet_files"]:
            k = (f.get("config"), f.get("split"))
            a = agg.setdefault(k, [0, 0])
            a[0] += f.get("num_rows", 0) or 0
            a[1] += f.get("size", 0) or 0
        for (cfg, split), (rows, size) in sorted(agg.items()):
            print(f"  parquet {cfg}/{split}: rows={rows:,} bytes={human(size)}")
    else:
        print("  parquet:", str(parq.get("error") or parq)[:120])
    # 2) categorical statistics per split
    stats = get(f"https://datasets-server.huggingface.co/statistics?dataset={ds}")
    splits = set()
    if parq.get("parquet_files"):
        splits = {(f.get("config"), f.get("split")) for f in parq["parquet_files"]}
    for cfg, split in sorted(splits):
        st = get(f"https://datasets-server.huggingface.co/statistics?dataset={ds}&config={cfg}&split={split}")
        if not st.get("statistics"):
            print(f"  stats {cfg}/{split}: {str(st.get('error') or st)[:100]}")
            continue
        print(f"  --- statistics {cfg}/{split} ---")
        for c in st["statistics"]:
            name = c.get("column_name", "?")
            ctype = c.get("column_type", "?")
            if ctype == "class_label":
                freqs = (c.get("frequencies") or {})
                names = c.get("class_label_values") or list(freqs.keys())
                # class_label: frequencies list aligns with class_label_values
                pairs = []
                if c.get("class_label_values"):
                    for i, lab in enumerate(c["class_label_values"]):
                        pairs.append((lab, freqs.get(str(i), freqs.get(i, 0))))
                else:
                    pairs = list(freqs.items())
                total = sum(v for _, v in pairs) or 1
                print(f"    {name} (class_label, {total:,} rows):")
                for lab, v in sorted(pairs, key=lambda x: -x[1])[:14]:
                    print(f"      {str(lab):26s} {v:>9,}  ({v*100.0/total:.1f}%)")
            elif ctype in ("string", "bool"):
                freqs = (c.get("frequencies") or {})
                total = sum(freqs.values()) or 1
                print(f"    {name} ({ctype}, {total:,} non-null):")
                for k, v in sorted(freqs.items(), key=lambda x: -x[1])[:14]:
                    print(f"      {str(k)[:40]:42s} {v:>9,}  ({v*100.0/total:.1f}%)")
            elif ctype == "float" and c.get("column_statistics"):
                cs = c["column_statistics"]
                if "sum" in cs:
                    print(f"    {name}: sum={cs['sum']:,.0f} min={cs.get('min',0):.1f} max={cs.get('max',0):.1f}")
