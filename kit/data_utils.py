# -*- coding: utf-8 -*-
"""
data_utils.py — أدوات البيانات ونقاط التحقق لـ Seed-VC العراقي
================================================================
  audit()          : تدقيق شامل لمجلد بيانات (عدد/مدة/صيغ/ساعات/متحدثون تقريبياً)
  make_holdout()   : عزل مقاطع اختبار معزولة عن التدريب (مع حماية من إعادة التسريب)
  verify_ckpt()    : فحص سلامة أي نقطة (مفاتيح + أوزان finite + حالة محسّن + خطوة)
  fingerprint()    : بصمة سريعة لمجلد البيانات لمنع تسرّب الاختبار

استخدام CLI:
    python data_utils.py audit /path/to/data
    python data_utils.py holdout /path/to/data --count 24 --test-dir /path/to/holdout
    python data_utils.py verify runs/iraqi_v1/DiT_epoch_00000_step_01000.pth
"""
import argparse
import hashlib
import json
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a"}
MIN_DUR, MAX_DUR = 1.0, 30.0


def iter_audio(data_dir):
    for p in Path(data_dir).rglob("*"):
        if p.suffix.lower() in AUDIO_EXT:
            yield p


def audio_duration(path):
    """مدة المقطع بالثواني (بلا تحميل كامل للبيانات عند الإمكان)."""
    try:
        import soundfile as sf
        info = sf.info(str(path))
        return float(info.frames / info.samplerate)
    except Exception:
        try:
            import librosa
            return float(librosa.get_duration(path=str(path)))
        except Exception:
            return None


def fingerprint(data_dir):
    """بصمة SHA1 لأسماء وأحجام كل المقاطع — لاكتشاف أي تغيّر في البيانات."""
    h = hashlib.sha1()
    for p in sorted(iter_audio(data_dir)):
        h.update(p.relative_to(data_dir).as_posix().encode())
        h.update(str(p.stat().st_size).encode())
    return h.hexdigest()[:16]


def audit(data_dir, sample_size=400, fast=False):
    data_dir = Path(data_dir)
    files = list(iter_audio(data_dir))
    assert files, f"لا ملفات صوتية في {data_dir}"

    fmts = Counter(p.suffix.lower() for p in files)
    top_dirs = Counter(p.parent.relative_to(data_dir).as_posix() if p.parent != data_dir else "."
                       for p in files)

    total_hours, durs, bad = 0.0, [], []
    sample = files if fast else random.sample(files, min(sample_size, len(files)))
    for p in sample:
        d = audio_duration(p)
        if d is None:
            bad.append(p.name)
        else:
            durs.append(d)
    if durs:
        mean_dur = sum(durs) / len(durs)
        est_hours = mean_dur * len(files) / 3600
    else:
        mean_dur, est_hours = 0.0, 0.0

    inside = sum(1 for d in durs if MIN_DUR <= d <= MAX_DUR)
    report = {
        "n_files": len(files),
        "formats": dict(fmts),
        "top_dirs": dict(top_dirs.most_common(10)),
        "sampled": len(sample),
        "mean_duration": round(mean_dur, 2),
        "est_hours": round(est_hours, 1),
        "pct_in_1_30s": round(100 * inside / max(len(durs), 1), 1),
        "unreadable": bad[:20],
        "fingerprint": fingerprint(data_dir),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nأحصينا {len(files)} ملفاً بمتوسط مدة {mean_dur:.1f}s "
          f"(تقدير {est_hours:.1f} ساعة)")
    print(f"{report['pct_in_1_30s']}% داخل نطاق 1–30 ثانية (خارج النطاق يتجاهله المدرب)")
    if bad:
        print(f"⚠ {len(bad)} ملفاً غير قابل للقراءة (عينة): {bad[:5]}")
    return report


def make_holdout(data_dir, count=24, test_dir=None, seed=42):
    """عزل count مقطعاً عشوائياً إلى test_dir (نقل، لا نسخ — منع التسريب الهيكلي)."""
    data_dir, test_dir = Path(data_dir), Path(test_dir or (Path(data_dir).parent / "holdout_test"))
    test_dir.mkdir(parents=True, exist_ok=True)
    existing = list(iter_audio(test_dir))
    if existing:
        print(f"[holdout] يوجد {len(existing)} مقطعاً معزولاً مسبقاً في {test_dir} — لا نلمسها")
        print("[holdout] لعزل مجموعة جديدة: احذف المجلد أو مرر مساراً جديداً")
        return existing

    # اختيار من مقاطع متوسطة الطول (2–15 ثانية): مثالية للاختبار السمعي
    cands = []
    files = list(iter_audio(data_dir))
    random.seed(seed)
    random.shuffle(files)
    unreadable = 0
    for p in files:
        if len(cands) >= count:
            break
        d = audio_duration(p)
        if d is None:
            unreadable += 1
            continue
        if 2.0 <= d <= 15.0:
            cands.append(p)
    # إن لم تكتفِ المقاطع المتوسطة، أكمل من أي طول صالح
    if len(cands) < count:
        for p in files:
            if len(cands) >= count:
                break
            if p in cands:
                continue
            d = audio_duration(p)
            if d is None:
                continue
            if MIN_DUR <= d <= MAX_DUR:
                cands.append(p)

    if not cands:
        raise RuntimeError(
            f"لم يُعثر على أي مقطع صالح للعزل (فُحص {len(files)}، غير قابل للقراءة {unreadable}) — "
            "افحص البيانات بـ: python data_utils.py audit " + str(data_dir))
    if unreadable:
        print(f"[holdout] ⚠ {unreadable} ملفاً غير قابل للقراءة للمدة — تجاهَلها العزل")

    moved = []
    for p in cands:
        dst = test_dir / p.name
        shutil.move(str(p), str(dst))
        moved.append(dst)
    print(f"[holdout] عُزل {len(moved)} مقطعاً إلى {test_dir}")
    print(f"[holdout] بصمة بيانات التدريب بعد العزل: {fingerprint(data_dir)}")
    with open(test_dir / "holdout_manifest.json", "w", encoding="utf-8") as f:
        json.dump({"moved_from": str(data_dir), "files": [m.name for m in moved],
                   "seed": seed}, f, ensure_ascii=False, indent=2)
    return moved


def verify_ckpt(path):
    """فحص سلامة نقطة تحقق: المفاتيح، finite، حالة المحسّن، الخطوة."""
    import torch
    path = Path(path)
    assert path.exists(), f"الملف غير موجود: {path}"
    size_gb = path.stat().st_size / 1e9
    state = torch.load(path, map_location="cpu")
    keys = list(state.keys())
    net_keys = list(state.get("net", {}).keys()) if "net" in state else []
    report = {"file": path.name, "size_gb": round(size_gb, 2), "top_keys": keys,
              "net_modules": net_keys}
    has_opt = "optimizer" in state and bool(state.get("optimizer"))
    report["has_optimizer_state"] = has_opt
    if "iters" in state:
        report["iters"] = state["iters"]
    if "epoch" in state:
        report["epoch"] = state["epoch"]

    # فحص finite على عينة من الأوزان
    non_finite = 0
    checked = 0
    for mk in net_keys:
        sd = state["net"][mk]
        for i, (k, v) in enumerate(sd.items()):
            if not torch.is_tensor(v) or not v.is_floating_point():
                continue
            checked += 1
            if not torch.isfinite(v).all():
                non_finite += 1
            if checked >= 300:
                break
        if checked >= 300:
            break
    report["weights_checked"] = checked
    report["non_finite_tensors"] = non_finite
    report["verdict"] = ("سليمة ✓" if non_finite == 0 else "⚠ توجد أوزان غير رقمية!")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    if has_opt:
        print("→ قابلة للاستئناف الكامل (أوزان + مُحسِّن + مجدول)")
    else:
        print("→ أوزان فقط (نقطة رسمية أو ft_model) — تصلح لبداية جديدة لا لاستئناف كامل")
    return report


def main():
    ap = argparse.ArgumentParser(description="أدوات البيانات والنقاط")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a1 = sub.add_parser("audit", help="تدقيق مجلد بيانات")
    a1.add_argument("data_dir")
    a1.add_argument("--sample", type=int, default=400)
    a1.add_argument("--fast", action="store_true")
    a2 = sub.add_parser("holdout", help="عزل مقاطع اختبار")
    a2.add_argument("data_dir")
    a2.add_argument("--count", type=int, default=24)
    a2.add_argument("--test-dir", default=None)
    a2.add_argument("--seed", type=int, default=42)
    a3 = sub.add_parser("verify", help="فحص نقطة تحقق")
    a3.add_argument("ckpt")
    a4 = sub.add_parser("fingerprint", help="بصمة مجلد بيانات")
    a4.add_argument("data_dir")
    args = ap.parse_args()

    if args.cmd == "audit":
        audit(args.data_dir, args.sample, args.fast)
    elif args.cmd == "holdout":
        make_holdout(args.data_dir, args.count, args.test_dir, args.seed)
    elif args.cmd == "verify":
        verify_ckpt(args.ckpt)
    elif args.cmd == "fingerprint":
        print(fingerprint(args.data_dir))


if __name__ == "__main__":
    main()
