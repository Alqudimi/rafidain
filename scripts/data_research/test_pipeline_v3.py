#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Functional tests for the v3 pipeline additions (lahgtna + kharrufa + prep).

Uses a mini Kharrufa fixture (60 ar-IQ wavs) instead of the 2.06 GB archive,
and capped Lahgtna /rows paging (15 -> 30 pages) to stay fast."""
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path

BASE = Path("/home/z/my-project")
PIPE = BASE / "scripts" / "pipeline_v3.py"
ROOT = Path("/tmp/iq_v3_test")
MINI = Path("/tmp/kharrufa_mini.zip")

if ROOT.exists():
    shutil.rmtree(ROOT)
os.makedirs(ROOT / "raw")

# ---------- build the mini fixture once (60 ar-IQ wavs + metadata) ----------
if not MINI.exists():
    src = zipfile.ZipFile("/tmp/kharrufa_corpus.bin")
    names = src.namelist()
    iq_wavs = [n for n in names if n.startswith("ar-IQ_hayder/audio_files/") and n.endswith(".wav")][:60]
    with zipfile.ZipFile(MINI, "w", zipfile.ZIP_STORED) as z:
        for n in iq_wavs:
            z.writestr("ar-IQ_hayder/audio_files/" + n.split("/")[-1], src.read(n))
        z.writestr("ar-IQ_hayder/metadata.txt", src.read("ar-IQ_hayder/metadata.txt"))
    src.close()
    print(f"fixture: {MINI} ({MINI.stat().st_size/1e6:.0f} MB, 60 wavs)")

sys.path.insert(0, str(BASE / "scripts"))
import importlib.util
spec = importlib.util.spec_from_file_location("iraqi_pipeline_v3", PIPE)
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

results = []

def check(name, cond, detail=""):
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)

def place_zip(zdir):
    zdir.mkdir(parents=True, exist_ok=True)
    zp = zdir / "iraqi_dialect_tts_corpus.zip"
    if zp.exists():
        zp.unlink()
    shutil.copyfile(MINI, zp)

# ------------------------------------------------------------------ 1) kharrufa
print("\n=== TEST 1: kharrufa (mini fixture) ===", flush=True)
t0 = time.time()
place_zip(ROOT / "raw" / "_kharrufa")
try:
    P._fetch_kharrufa(str(ROOT), str(ROOT / "raw" / "kharrufa"))
    kwavs = sorted((ROOT / "raw" / "kharrufa").glob("*.wav"))
    check("kharrufa: wav count == 60", len(kwavs) == 60, f"{len(kwavs)} wavs in {time.time()-t0:.0f}s")
    check("kharrufa: metadata extracted", (ROOT / "raw" / "kharrufa" / "_kharrufa_metadata.txt").exists())
    sizes = [w.stat().st_size for w in kwavs]
    check("kharrufa: all wavs non-trivial", min(sizes) > 1024, f"min {min(sizes):,} B")
    check("kharrufa: size plausible", 40e6 < sum(sizes) < 200e6, f"{sum(sizes)/1e6:.0f} MB")
    check("kharrufa: zip kept (<400 wavs branch)",
          (ROOT / "raw" / "_kharrufa" / "iraqi_dialect_tts_corpus.zip").exists())
    pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
                         "-of", "json", str(kwavs[0])], capture_output=True, text=True)
    j = json.loads(pr.stdout)
    check("kharrufa: first wav is 44.1kHz", j["streams"][0]["sample_rate"] == "44100",
          f"dur={j['format']['duration']}s")
except Exception as e:
    check("kharrufa: fetch raised", False, repr(e))

# ------------------------------------------------------------------ 2) lahgtna (capped)
print("\n=== TEST 2: lahgtna (/rows paging, capped at 15 pages) ===", flush=True)
os.environ["IRAQI_LAHGTNA_MAX_PAGES"] = "15"
t0 = time.time()
lahdir = ROOT / "raw" / "lahgtna"
try:
    P._fetch_lahgtna(str(ROOT), str(lahdir), workers=8)
    st = json.loads((ROOT / "raw" / "_lahgtna_state.json").read_text())
    check("lahgtna: 15 pages done", len(st["pages_done"]) == 15,
          f"{len(st['pages_done'])} pages in {time.time()-t0:.0f}s")
    check("lahgtna: no failed pages", not st["failed_pages"], str(st["failed_pages"][:5]))
    check("lahgtna: no failed audio", not st["failed_audio"])
    lwavs = sorted(lahdir.glob("*.wav"))
    check("lahgtna: wavs downloaded", len(lwavs) >= 10, f"{len(lwavs)} wavs")
    tx = [json.loads(l) for l in (lahdir / "_transcripts.jsonl").read_text().splitlines()]
    check("lahgtna: transcripts saved", len(tx) == len(lwavs), f"{len(tx)} transcripts")
    if lwavs:
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
                             "-of", "json", str(lwavs[0])], capture_output=True, text=True)
        j = json.loads(pr.stdout)
        check("lahgtna: 16kHz wav", j["streams"][0]["sample_rate"] == "16000",
              f"dur={j['format']['duration']}s size={lwavs[0].stat().st_size:,}B")
        check("lahgtna: transcripts non-empty", tx[0]["text"].strip() != "", tx[0]["text"][:60])
    n_before = len(lwavs)
except Exception as e:
    check("lahgtna: fetch raised", False, repr(e))
    n_before = 0

# ------------------------------------------------------------------ 3) lahgtna resume
print("\n=== TEST 3: lahgtna resume (15 -> 30 pages) ===", flush=True)
os.environ["IRAQI_LAHGTNA_MAX_PAGES"] = "30"
try:
    t0 = time.time()
    P._fetch_lahgtna(str(ROOT), str(lahdir), workers=8)
    st = json.loads((ROOT / "raw" / "_lahgtna_state.json").read_text())
    check("lahgtna: 30 pages attempted",
          len(st["pages_done"]) + len(st["failed_pages"]) == 30,
          f"{len(st['pages_done'])} done + {len(st['failed_pages'])} failed (transients retried next run)")
    lwavs2 = sorted(lahdir.glob("*.wav"))
    check("lahgtna: failures recorded for retry (<=2 pages)", len(st["failed_pages"]) <= 2,
          str(st["failed_pages"][:5]))
    check("lahgtna: no lost wavs after resume", len(lwavs2) >= 28,
          f"{len(lwavs2)} wavs on disk")
except Exception as e:
    check("lahgtna: resume raised", False, repr(e))

# ------------------------------------------------------------------ 4) prep end-to-end
print("\n=== TEST 4: prep on the fetched data ===", flush=True)
t0 = time.time()
outd = ROOT / "data" / "ft"
rc = subprocess.run([sys.executable, str(PIPE), "prep", "--root", str(ROOT),
                     "--out", str(outd), "--workers", "8"],
                    capture_output=True, text=True, timeout=600)
check("prep: exit code 0", rc.returncode == 0, (rc.stderr or "")[-200:] if rc.returncode else "")
if rc.returncode == 0:
    print(f"  [t] prep done in {time.time()-t0:.0f}s", flush=True)
    og_lah = list((outd / "lahgtna").glob("*.ogg"))
    og_kh = list((outd / "kharrufa").glob("*.ogg"))
    check("prep: lahgtna oggs produced", len(og_lah) >= 8, f"{len(og_lah)} oggs")
    check("prep: kharrufa oggs produced", len(og_kh) >= 40, f"{len(og_kh)} oggs")
    idx = [json.loads(l) for l in (outd / "_prep_index.jsonl").read_text().splitlines()]
    tot_h = sum(r["dur"] for r in idx) / 3600
    check("prep: index has durations", tot_h > 0.05, f"total {tot_h:.2f} h over {len(idx)} clips")
    st = json.loads((outd / "_stats.json").read_text())
    check("prep: stats per-source present",
          {"lahgtna", "kharrufa"} <= set(st["per_source"]),
          json.dumps(st["per_source"], ensure_ascii=False))
    if og_lah:
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=sample_rate",
                             "-of", "json", str(og_lah[0])], capture_output=True, text=True)
        j = json.loads(pr.stdout)
        check("prep: decodable opus (24k or 48k container rate)",
              j["streams"][0]["sample_rate"] in ("24000", "48000")
              and float(j["format"]["duration"]) > 0 and og_lah[0].suffix == ".ogg",
              f"dur={j['format']['duration']}s rate={j['streams'][0]['sample_rate']}")

# ------------------------------------------------------------------ 5) cmd_small end-to-end
print("\n=== TEST 5: cmd_small full path (marks .done, skips on re-run) ===", flush=True)
shutil.rmtree(ROOT / "data" / "ft", ignore_errors=True)
for d in ("lahgtna", "kharrufa"):
    shutil.rmtree(ROOT / "raw" / d, ignore_errors=True)
(ROOT / "raw" / "_lahgtna_state.json").unlink(missing_ok=True)
place_zip(ROOT / "raw" / "_kharrufa")
os.environ["IRAQI_LAHGTNA_MAX_PAGES"] = "8"
print("  [t] cmd_small #1 (fresh fetch, 8 pages) ...", flush=True)
t0 = time.time()
rc1 = subprocess.run([sys.executable, str(PIPE), "small", "--root", str(ROOT),
                      "--only", "lahgtna", "kharrufa", "--workers", "8"],
                     capture_output=True, text=True, timeout=600)
check("small: exit code 0", rc1.returncode == 0, (rc1.stdout or "")[-300:] if rc1.returncode else "")
print(f"  [t] cmd_small #1 done in {time.time()-t0:.0f}s", flush=True)
done1 = (ROOT / ".done" / "small_lahgtna.done").exists() and (ROOT / ".done" / "small_kharrufa.done").exists()
check("small: .done markers written", done1)
rc2 = subprocess.run([sys.executable, str(PIPE), "small", "--root", str(ROOT),
                      "--only", "lahgtna", "kharrufa", "--workers", "8"],
                     capture_output=True, text=True, timeout=300)
check("small: re-run skips cached sets", "already done, skipping" in rc2.stdout,
      "second run output contains skip markers")

os.environ["IRAQI_LAHGTNA_MAX_PAGES"] = "0"
print("\n" + "=" * 60)
fails = [r for r in results if not r[1]]
print(f"RESULTS: {len(results) - len(fails)}/{len(results)} passed")
for n, ok, d in fails:
    print("  FAIL:", n, d)
sys.exit(1 if fails else 0)
