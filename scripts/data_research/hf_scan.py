#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scan the Hugging Face datasets API for Iraqi-Arabic speech datasets."""
import json, sys, urllib.request

API = "https://huggingface.co/api"
HDR = {"User-Agent": "rafidain-scan/1.0"}

def get(url):
    req = urllib.request.Request(url, headers=HDR)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())

TERMS = [
    "iraqi", "iraq arabic", "iraqi arabic", "iraqi dialect",
    "iraqi speech", "iraqi audio", "arabic dialect", "arabiya iraq",
    "baghdad", "basra", "mosul", "iraqi tts", "iraqi asr",
]

seen = {}
for t in TERMS:
    try:
        rows = get(f"{API}/datasets?search={urllib.parse.quote(t)}&limit=100&full=false")
    except Exception as e:
        print(f"  ! search '{t}' failed: {e}")
        continue
    for d in rows:
        did = d["id"]
        if seen.get(did) is None or d.get("downloads", 0) > seen[did].get("downloads", 0):
            seen[did] = d
    print(f"[{t}] {len(rows)} hits (cumulative unique: {len(seen)})")

print("=" * 70)
ranked = sorted(seen.values(),
                key=lambda d: (-(d.get("likes") or 0), -(d.get("downloads") or 0)))
for d in ranked:
    if d.get("downloads", 0) == 0 and d.get("likes", 0) == 0:
        continue
    print(f"{d['id']:65s}  dl={d.get('downloads',0):>6}  likes={d.get('likes',0):>4}  "
          f"mod={str(d.get('lastModified',''))[:10]}")

print("=" * 70)
print("ALL unique ids (incl. zero-download):")
for d in sorted(seen.values(), key=lambda x: x["id"]):
    print(" ", d["id"])
