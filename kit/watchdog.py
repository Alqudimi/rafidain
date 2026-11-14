# -*- coding: utf-8 -*-
"""
watchdog.py — الحارس الذكي للتدريب المستمر داخل النواة
=========================================================
يطلق train_iraqi.py في thread خلفي **داخل نواة Colab نفسها** (لا subprocess غريب)،
فلا تُقتل العملية عند إغلاق المتصفح — النواة تستمر طالما الجلسة حية.
المدرب المحصّن يوقف نفسه نظيفاً عند انتهاء الميزانية، والحارس يعيد إطلاقه
في جلسة Colab جديدة تلقائياً (عند إعادة تشغيل الخلية).

لا يستخدم subprocess.Popen للتدريب — التدريب يعمل بالخيوط داخل العملية
الواحدة، والحارس يراقب الخيط، ويعيد الإطلاق إذا مات قبل بلوغ الهدف.

الاستخدام في خلية Colab:
    from watchdog import run_training_forever
    run_training_forever(
        repo_dir="/content/iraqi_seedvc/seed-vc",
        config="configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml",
        dataset_dir="/content/iraqi_seedvc/data",
        run_name="iraqi_v1",
        total_steps=6000,
        time_budget_hours=11,
        drive_dir="/content/drive/MyDrive/iraqi_seedvc/runs/iraqi_v1",
    )
"""
import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path


def _log(msg):
    print(f"[watchdog {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run_training_forever(
    repo_dir,
    config,
    dataset_dir,
    run_name="iraqi_v1",
    total_steps=6000,
    batch_size=0,
    save_every=500,
    num_workers=2,
    time_budget_hours=11,
    drive_dir=None,
    keep_checkpoints=2,
    kit_dir=None,
    max_restarts=50,
    validate_first=False,
):
    """إطلاق التدريب مع مراقبة وإعادة تشغيل تلقائية حتى بلوغ الهدف التراكمي.

    يعمل blocking داخل الخلية (مقصود): أغلقت المتصفح فلا مشكلة، النواة مستمرة.
    أوقفه بتحرير خيار التشغيل (Interrupt) — الحارس يحفظ نقطة نظيفة عند الانقطاع.
    """
    repo_dir = Path(repo_dir).resolve()
    kit_dir = Path(kit_dir or Path(__file__).resolve().parent).resolve()
    script = kit_dir / "train_iraqi.py"
    assert script.exists(), f"train_iraqi.py غير موجود في {kit_dir}"
    assert (repo_dir / "train.py").exists(), f"مجلد seed-vc غير صحيح: {repo_dir}"

    sys.path.insert(0, str(kit_dir))
    os.environ.setdefault("IRAQI_KIT_DIR", str(kit_dir))

    if validate_first:
        _log("فحص السلامة أولاً (--validate-instead)...")
        r = subprocess.run(
            [sys.executable, str(script), "--config", str(config),
             "--dataset-dir", str(dataset_dir), "--run-name", run_name,
             "--validate-instead"],
            cwd=str(repo_dir), capture_output=True, text=True)
        print(r.stdout)
        if r.returncode != 0:
            print(r.stderr)
            raise RuntimeError("فشل فحص السلامة — راجع المخرجات أعلاه")

    restarts = 0
    while restarts < max_restarts:
        # أين وصلنا؟ (قراءة نبضة القلب أو آخر نقطة)
        import ckpt_utils
        run_path = repo_dir / "runs" / run_name
        ckpts = ckpt_utils.list_ckpts(run_path)
        reached = ckpt_utils._step_of(ckpts[-1]) if ckpts else 0
        if reached >= total_steps:
            _log(f"الهدف التراكمي {total_steps} محقق (وصلنا {reached}) — انتهى العمل ✓")
            return
        remaining = total_steps - reached
        _log(f"الوضع: {reached}/{total_steps} | متبقٍ {remaining} | إطلاق #{restarts + 1}")

        cmd = [sys.executable, "-u", str(script),
               "--config", str(config),
               "--dataset-dir", str(dataset_dir),
               "--run-name", run_name,
               "--total-steps", str(total_steps),
               "--save-every", str(save_every),
               "--num-workers", str(num_workers),
               "--time-budget-hours", str(time_budget_hours),
               "--keep-checkpoints", str(keep_checkpoints)]
        if batch_size:
            cmd += ["--batch-size", str(batch_size)]
        if drive_dir:
            cmd += ["--drive-dir", str(drive_dir)]

        # ---------- التشغيل داخل العملية (خيط) ----------
        exit_code = None

        def _run():
            nonlocal exit_code
            # تشغيل التدريب في هذه العملية: sys.argv ثم exec (الملف مكتوب بأسلوب __main__)
            os.chdir(repo_dir)
            # cmd = [python, -u, script, args...] → sys.argv = [script, args...]
            sys.argv = [str(script)] + cmd[3:]
            try:
                code = script.read_text(encoding="utf-8")
                ns = {"__name__": "__main__", "__file__": str(script)}
                exec(compile(code, str(script), "exec"), ns)
                exit_code = 0
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 0
            except BaseException:
                traceback.print_exc()
                exit_code = 1

        t = threading.Thread(target=_run, daemon=False)
        t.start()
        t.join()

        # بعد انتهاء الخيط: أين وصلنا الآن؟
        time.sleep(5)
        ckpts = ckpt_utils.list_ckpts(run_path)
        reached = ckpt_utils._step_of(ckpts[-1]) if ckpts else 0
        if reached >= total_steps:
            _log(f"الهدف محقق ({reached}/{total_steps}) — انتهى العمل ✓")
            return

        restarts += 1
        if exit_code not in (0, None):
            _log(f"انتهى التشغيل برمز {exit_code} — إعادة الإطلاق بعد 60 ثانية...")
            time.sleep(60)
        else:
            _log("انتهى التشغيل (ميزانية على الأرجح) — إعادة الإطلاق فوراً في الجلسة نفسها إن أمكن")
            time.sleep(10)

    _log(f"تجاوزنا حد إعادة التشغيل ({max_restarts}) — شغّل الخلية مجدداً لمتابعة الهدف")


def check_session_health(drive_dir=None, hours=24):
    """فحص سريع: ماذا يوجد محلياً وعلى Drive؟ لطباعته في بداية أي جلسة."""
    import ckpt_utils
    print("=" * 60)
    print("فحص حالة الجلسة")
    print("=" * 60)
    for label, base in (("محلي", Path("runs")), ("Drive", Path(drive_dir) if drive_dir else None)):
        if base is None or not base.exists():
            print(f"[{label}] غير موجود")
            continue
        for run in sorted(base.glob("*/")):
            ckpts = ckpt_utils.list_ckpts(run)
            hb = run / "heartbeat.json"
            hb_info = ""
            if hb.exists():
                try:
                    import json as _json
                    h = _json.loads(hb.read_text())
                    hb_info = (f" | نبضة: خطوة {h.get('step')} قبل "
                               f"{int(time.time()-h.get('ts',0))//60} دقيقة")
                except Exception:
                    pass
            if ckpts:
                print(f"[{label}] {run.name}: {len(ckpts)} نقطة، أحدثها {ckpts[-1].name}{hb_info}")
            elif (run / "ft_model.pth").exists():
                print(f"[{label}] {run.name}: ft_model.pth فقط")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-dir", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--run-name", default="iraqi_v1")
    ap.add_argument("--total-steps", type=int, default=6000)
    ap.add_argument("--time-budget-hours", type=float, default=11)
    ap.add_argument("--drive-dir", default=None)
    ap.add_argument("--validate-first", action="store_true")
    a = ap.parse_args()
    run_training_forever(
        repo_dir=a.repo_dir, config=a.config, dataset_dir=a.dataset_dir,
        run_name=a.run_name, total_steps=a.total_steps,
        time_budget_hours=a.time_budget_hours, drive_dir=a.drive_dir,
        validate_first=a.validate_first,
    )
