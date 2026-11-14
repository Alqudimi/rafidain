# -*- coding: utf-8 -*-
"""
ckpt_utils.py — إدارة نقاط التحقق: حفظ ذرّي + مرآة Drive + تقليم آمن
====================================================================
مصمم ليعمل مع مستودع seed-vc الرسمي دون أي تعديل فيه.
كل الدوال مستقلة (لا تعتمد على متغيرات نوتبوك خارجية).

الضمانات:
  • حفظ ذرّي (tmp → rename): لا نقطة نصف مكتوبة حتى لو انقطعت الجلسة أثناء الحفظ
  • تقليم لا يمسّ نقاطاً محمية (gold_*) ولا يخالف أصل المستودع (يُبقي آخر N)
  • مرآة Drive تحمي أحدث نقطة دائماً حتى لو كانت العملية الأم ستُقتل بعدها بلحظة
"""
import os
import glob
import json
import re
import shutil
import time
from pathlib import Path

CKPT_PATTERN = "DiT_epoch_*_step_*.pth"


def _step_of(path):
    """رقم الخطوة من اسم الملف الرسمي DiT_epoch_00000_step_01000.pth"""
    m = re.search(r"step_(\d+)", Path(path).stem)
    return int(m.group(1)) if m else -1


def list_ckpts(run_dir):
    """كل نقاط التحقق الرسمية في مجلد التشغيل مرتبة تصاعدياً بالخطوة."""
    run_dir = Path(run_dir)
    return sorted(run_dir.glob(CKPT_PATTERN), key=_step_of)


def save_atomic(state, path):
    """حفظ torch.save بشكل ذرّي: يكتب إلى .tmp ثم rename — لا أنصاف ملفات أبداً."""
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    # fsync غير متاح عبر API العامة لـ torch.save؛ rename بعد اكتمال الكتابة كافٍ عملياً
    os.replace(tmp, path)
    return path


def prune_keep_n(run_dir, keep=2, protect=("gold_",), also_delete_glob=None):
    """تقليم آمن: يُبقي آخر keep نقطة، ويحمي أي ملف يبدأ اسمه بأحد protect prefixes.

    أيضاً: يحذف ملفات .tmp اليتيمة (من انقطاع أثناء الحفظ).
    يعيد قائمة (المحذوفات, المتبقون).
    """
    run_dir = Path(run_dir)
    removed, kept = [], []
    # تنظيف أنصاف الملفات أولاً
    for tmp in run_dir.glob("*.pth.tmp"):
        tmp.unlink()
        print(f"[prune] حُذف ملف ناقص: {tmp.name}")
    ckpts = list_ckpts(run_dir)
    for c in ckpts[:-keep] if keep > 0 else []:
        if c.name.startswith(protect):
            kept.append(c)
            continue
        c.unlink()
        removed.append(c.name)
        print(f"[prune] حُذف {c.name} (إبقاء آخر {keep})")
    kept += ckpts[-keep:] if keep > 0 else []
    return removed, kept


def mirror_to_drive(run_dir, drive_dir, keep_on_drive=2, protect=("gold_",)):
    """نسخ مرآة آمنة إلى Drive: يُنسخ الجديد، ثم يُقلَّم القديم (لا يُحذف قبل وصول البديل).

    drive_dir: مسار مجلد النقاط على Drive (مثال: /content/drive/MyDrive/iraqi_seedvc/runs/iraqi_v1)
    """
    run_dir, drive_dir = Path(run_dir), Path(drive_dir)
    if not drive_dir.exists():
        print(f"[mirror] مجلد Drive غير موجود: {drive_dir}")
        return []
    local_ckpts = list_ckpts(run_dir)
    if not local_ckpts:
        return []
    copied = []
    for c in local_ckpts[-keep_on_drive:]:
        dst = drive_dir / c.name
        if not dst.exists() or dst.stat().st_size != c.stat().st_size:
            tmp_dst = dst.with_suffix(".pth.part")
            shutil.copy2(c, tmp_dst)
            os.replace(tmp_dst, dst)
            copied.append(c.name)
            print(f"[mirror] {c.name} → Drive ({c.stat().st_size/1e9:.2f}GB)")
    # تقليم Drive بعد النسخ (حماية gold_ دائماً)
    drive_ckpts = list_ckpts(drive_dir)
    for c in drive_ckpts[:-keep_on_drive] if keep_on_drive > 0 else []:
        if c.name.startswith(protect):
            continue
        try:
            c.unlink()
            print(f"[mirror] قُلِّم من Drive: {c.name}")
        except OSError as e:
            print(f"[mirror] تعذر حذف {c.name}: {e}")
    return copied


def write_heartbeat(path, payload):
    """نبضة قلب JSON ذرّية — تقرأها أداة المراقبة والساعة لاحقاً."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def save_full_state(run_dir, state, step, keep=2, drive_dir=None,
                    keep_on_drive=2, heartbeat=None, protect=("gold_",)):
    """الواجهة الجامعة: حفظ ذرّي + تقليم + مرآة + نبضة — استدعاء واحد من المدرب.

    state: القاموس الرسمي الكامل {net, optimizer, scheduler, iters, epoch}
    step:  رقم الخطوة الحالية (يُستخدم للاسم وللنبضة)
    """
    epoch = state.get("epoch", 0)
    name = f"DiT_epoch_{epoch:05d}_step_{step:05d}.pth"
    path = save_atomic(state, Path(run_dir) / name)
    print(f"[ckpt] حُفظت {name}")
    removed, kept = prune_keep_n(run_dir, keep=keep, protect=protect)
    if drive_dir:
        mirror_to_drive(run_dir, drive_dir, keep_on_drive=keep_on_drive, protect=protect)
    if heartbeat:
        write_heartbeat(heartbeat, {
            "step": int(step), "epoch": int(epoch),
            "ts": time.time(), "ckpt": name,
            "kept": [k.name for k in kept],
        })
    return path
