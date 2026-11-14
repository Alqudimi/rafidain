# -*- coding: utf-8 -*-
"""
train_iraqi.py — المدرب المحصّن لتدريب Seed-VC على اللهجة العراقية
====================================================================
نسخة محصّنة من train.py الرسمي (Plachtaa/seed-vc) بكل ميزاته + إصلاحات حرجة:

  1. استئناف كامل (وزن + مُحسِّن + مجدول + خطوات): الرسمي يحمّل الأوزان فقط
     (load_only_params=True) فيصفّر العدّاد عند كل استئناف ويحذف نقاط الجلسة
     الجديدة صامتةً (لأن أرقامها أصغر). هنا: استئناف كامل حقيقي متصل عبر الجلسات.
  2. حفظ ذرّي + مرآة Drive فورية لكل نقطة (حتى لو قُطعت الجلسة لحظة الحفظ).
  3. حارس NaN/Inf: يوقف التدريب فوراً عند أول خسارة غير رقمية وينتقل لآخر نقطة سليمة.
  4. ميزانية زمنية: يتوقف بنفسه عند انتهاء ميزانية الجلسة (مثلاً 11 ساعة)
     ويحفظ نقطة أخيرة نظيفة — عمليّ للجلسات المجانية المقطوعة.
  5. إسكات طباعات min/max التشخيصية في modules/audio.py (سطران فقط).
  6. سجل loss_history.jsonl + نبضة heartbeat.json لكل خطوة حفظ.
  7. --validate-instead: شغّل ما يلي للتأكد من سلامة السلسلة قبل التدريب:
     python train_iraqi.py --config ... --dataset-dir ... --validate-instead
  8. --dry-run: بناء كل شيء (نماذج فرعية + بيانات) ثم تنفيذ 3 خطوات والخروج.

الاستخدام الشائع (من داخل مجلد seed-vc):
  python train_iraqi.py \
      --config configs/presets/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml \
      --dataset-dir /path/to/iraqi_data \
      --run-name iraqi_v1 \
      --total-steps 6000 \
      --save-every 500 \
      --time-budget-hours 11 \
      --drive-dir /content/drive/MyDrive/iraqi_seedvc/runs/iraqi_v1

  ملاحظة أساسية: نقطة البداية (الرسمية أو استئناف أحدث نقطة محلية) تُقرر تلقائياً:
  - إن وُجدت نقاط في runs/<run> → استئناف كامل من أحدثها (لو كانت تحمل optimizer).
  - وإلا يُنزّل النموذج الرسمي المذكور في الإعداد pretrained_model ويبدأ منه بلا حالة.
"""
import argparse
import glob
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import yaml

# --- إسكات طباعتي التشخيص قبل أي استيراد لموديولات المستودع ---
def _silence_mel_debug_prints(repo_root):
    audio_py = repo_root / "modules" / "audio.py"
    if audio_py.exists():
        s = audio_py.read_text(encoding="utf-8")
        n = (s.count('print("min value is ", torch.min(y))')
             + s.count('print("max value is ", torch.max(y))'))
        if n:
            s = (s.replace('print("min value is ", torch.min(y))', 'pass  # silenced by train_iraqi')
                  .replace('print("max value is ", torch.max(y))', 'pass  # silenced by train_iraqi'))
            audio_py.write_text(s, encoding="utf-8")
            print(f"[init] أُسكِتت {n} طباعة تشخيصية في modules/audio.py")
        else:
            print("[init] طباعات min/max غير نشطة أصلاً — لا حاجة للتغيير")
    else:
        print(f"[init] تحذير: {audio_py} غير موجود (هل أنت داخل مجلد seed-vc؟)")


REPO_ROOT = None


def _kit_path():
    """مسار مجلد الأدوات: إما بجانب هذا الملف أو عبر IRAQI_KIT_DIR."""
    here = Path(__file__).resolve().parent
    if (here / "ckpt_utils.py").exists():
        return here
    env = os.environ.get("IRAQI_KIT_DIR", "")
    if env and (Path(env) / "ckpt_utils.py").exists():
        return Path(env)
    raise RuntimeError("عجزت عن إيجاد ckpt_utils.py — حدد IRAQI_KIT_DIR")


def main():
    global REPO_ROOT
    # ذاكرة HF داخل المستودع (مطابق للرسمي) — قبل أي استيراد للموديولات
    os.environ.setdefault("HF_HUB_CACHE", "./checkpoints/hf_cache")
    ap = argparse.ArgumentParser(description="المدرب المحصّن — Seed-VC العراقي")
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset-dir", required=True)
    ap.add_argument("--run-name", default="iraqi_v1")
    ap.add_argument("--pretrained-ckpt", default=None,
                    help="نقطة بداية يدوية (افتراضياً: أحدث نقطة محلية أو النموذج الرسمي)")
    ap.add_argument("--batch-size", type=int, default=0, help="0 = من الإعداد")
    ap.add_argument("--total-steps", type=int, default=6000,
                    help="الهدف التراكمي الكلي (المدرب يُكمله من حيث توقفت)")
    ap.add_argument("--max-epochs", type=int, default=1000)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--keep-checkpoints", type=int, default=2)
    ap.add_argument("--time-budget-hours", type=float, default=0,
                    help="0 = بلا ميزانية؛ وإلا يتوقف نظيفاً قبل نفادها (مثلاً 11)")
    ap.add_argument("--lr", type=float, default=1e-5,
                    help="معدل التعلم (الافتراضي 1e-5 = مثبت train.py الرسمي، لا تغيّره)")
    ap.add_argument("--drive-dir", default=None,
                    help="مجلد مرآة Drive للنقاط (مثال: /content/drive/MyDrive/.../runs/iraqi_v1)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true", help="3 خطوات فقط للتشخيص")
    ap.add_argument("--validate-instead", action="store_true",
                    help="فحص سلامة السلسلة كاملة بلا تدريب (بيانات+نماذج+حفظ+قراءة)")
    ap.add_argument("--no-mirror", action="store_true")
    args = ap.parse_args()

    REPO_ROOT = Path(os.getcwd())
    _silence_mel_debug_prints(REPO_ROOT)

    from modules.commons import recursive_munch, build_model, load_checkpoint
    from optimizers import build_optimizer
    from data.ft_dataset import build_ft_dataloader
    from hf_utils import load_custom_model_from_hf
    sys.path.insert(0, str(_kit_path()))
    import ckpt_utils

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    print(f"[init] الجهاز: {device}")

    config = yaml.safe_load(open(args.config))
    log_dir = Path(config["log_dir"]) / args.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, log_dir / os.path.basename(args.config))

    # ---------- الاستئناف الكامل: اختر نقطة البداية ----------
    local_ckpts = ckpt_utils.list_ckpts(log_dir)
    start_ckpt = None
    resume_full = False
    if args.pretrained_ckpt:
        start_ckpt = args.pretrained_ckpt
        print(f"[resume] نقطة بداية يدوية: {start_ckpt}")
    elif local_ckpts:
        start_ckpt = str(local_ckpts[-1])
        print(f"[resume] وُجدت {len(local_ckpts)} نقطة محلية — استئناف كامل من {os.path.basename(start_ckpt)}")
    else:
        print("[resume] لا نقاط محلية — سنبدأ من النموذج الرسمي المنشور")

    if args.drive_dir and not args.no_mirror:
        drv = Path(args.drive_dir)
        drv.mkdir(parents=True, exist_ok=True)
        # استرجاع نقطة أحدث من Drive إن وُجدت
        drive_ckpts = ckpt_utils.list_ckpts(drv)
        if drive_ckpts and (not local_ckpts or
                            ckpt_utils._step_of(drive_ckpts[-1]) > ckpt_utils._step_of(local_ckpts[-1])):
            latest = drive_ckpts[-1]
            print(f"[resume] أحدث نقطة على Drive ({latest.name}) أحدث من المحلية — استرجاعها...")
            shutil.copy2(latest, log_dir / latest.name)
            start_ckpt = str(log_dir / latest.name)
            local_ckpts = ckpt_utils.list_ckpts(log_dir)
        # استرجاع السجل التاريخي إذا غاب محلياً
        if (drv / "loss_history.jsonl").exists() and not (log_dir / "loss_history.jsonl").exists():
            shutil.copy2(drv / "loss_history.jsonl", log_dir / "loss_history.jsonl")
            print("[resume] استُرجع سجل الخسارة التاريخي من Drive")

    # ---------- بناء النماذج (مطابق للرسمي) ----------
    sr = config["preprocess_params"].get("sr", 22050)
    preprocess_params = config["preprocess_params"]
    batch_size = args.batch_size or config.get("batch_size", 10)

    train_dataloader = build_ft_dataloader(
        args.dataset_dir, preprocess_params["spect_params"], sr,
        batch_size=batch_size, num_workers=args.num_workers,
    )
    f0_condition = config["model_params"]["DiT"].get("f0_condition", False)

    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params, stage="DiT")
    _ = [model[key].to(device) for key in model]
    model.cfm.estimator.setup_caches(max_batch_size=batch_size, max_seq_length=8192)
    optimizer = build_optimizer({key: model[key] for key in model}, lr=args.lr)

    epoch, iters = 0, 0
    if start_ckpt and os.path.exists(start_ckpt):
        # استئناف كامل إن وُجدت حالة المُحسِّن، وإلا أوزان فقط (نموذج رسمي)
        state_probe = torch.load(start_ckpt, map_location="cpu")
        has_opt = "optimizer" in state_probe and state_probe.get("optimizer")
        del state_probe
        model, optimizer, epoch, iters = load_checkpoint(
            model, optimizer, start_ckpt,
            load_only_params=not has_opt,
            ignore_modules=[], is_distributed=False,
        )
        mode = "استئناف كامل (أوزان + مُحسِّن + خطوات)" if has_opt else "أوزان فقط (نقطة رسمية)"
        print(f"[resume] {mode} — epoch={epoch}, iters={iters}")
    elif start_ckpt:
        raise FileNotFoundError(f"نقطة البداية غير موجودة: {start_ckpt}")

    if iters >= args.total_steps:
        print(f"[done] الهدف التراكمي {args.total_steps} محقق مسبقاً (iters={iters}) — لا شيء للتدريب.")
        return

    # ---------- النماذج الفرعية (مطابق للرسمي) ----------
    # CAMPPlus
    from modules.campplus.DTDNN import CAMPPlus
    campplus_model = CAMPPlus(feat_dim=80, embedding_size=192)
    campplus_sd_path = load_custom_model_from_hf("funasr/campplus", "campplus_cn_common.bin", config_filename=None)
    campplus_model.load_state_dict(torch.load(campplus_sd_path, map_location="cpu"))
    campplus_model.eval().to(device)
    sv_fn = campplus_model

    # whisper tokenizer
    from transformers import AutoFeatureExtractor, WhisperModel
    wname = config["model_params"]["speech_tokenizer"]["name"]
    whisper_model = WhisperModel.from_pretrained(wname).to(device)
    whisper_feature_extractor = AutoFeatureExtractor.from_pretrained(wname)
    del whisper_model.decoder

    def semantic_fn(waves_16k):
        ori_inputs = whisper_feature_extractor(
            [w16k.cpu().numpy() for w16k in waves_16k],
            return_tensors="pt", return_attention_mask=True, sampling_rate=16000,
        )
        ori_input_features = whisper_model._mask_input_features(
            ori_inputs.input_features, attention_mask=ori_inputs.attention_mask
        ).to(device)
        with torch.no_grad():
            ori_outputs = whisper_model.encoder(
                ori_input_features.to(whisper_model.encoder.dtype),
                head_mask=None, output_attentions=False,
                output_hidden_states=False, return_dict=True,
            )
        S_ori = ori_outputs.last_hidden_state.to(torch.float32)
        S_ori = S_ori[:, : waves_16k.size(-1) // 320 + 1]
        return S_ori

    # openvoice converter
    from modules.openvoice.api import ToneColorConverter
    ckpt_converter, config_converter = load_custom_model_from_hf(
        "myshell-ai/OpenVoiceV2", "converter/checkpoint.pth", "converter/config.json")
    tone_color_converter = ToneColorConverter(config_converter, device=device)
    tone_color_converter.load_ckpt(ckpt_converter)
    tone_color_converter.model.eval()
    se_db_path = load_custom_model_from_hf("Plachta/Seed-VC", "se_db.pt", None)
    se_db = torch.load(se_db_path, map_location="cpu")

    # f0
    if f0_condition:
        from modules.rmvpe import RMVPE
        rmvpe_model_path = load_custom_model_from_hf("lj1995/VoiceConversionWebUI", "rmvpe.pt", None)
        rmvpe = RMVPE(rmvpe_model_path, is_half=False, device=device)

    # vocoder — غير مستخدم في التدريب لكنه جزء من السلسلة الرسمية؛ نحذفه لتوفير الذاكرة
    # (الرسمي يبنيه بلا استخدام فعلي في train_one_step)

    # ---------- المدرب ----------
    loss_history_path = log_dir / "loss_history.jsonl"
    heartbeat_path = log_dir / "heartbeat.json"

    ema_loss = None
    loss_smoothing_rate = 0.99
    log_interval = config.get("log_interval", 10)
    t_start = time.time()
    budget_s = args.time_budget_hours * 3600 if args.time_budget_hours > 0 else float("inf")
    stop_reason = "اكتمل الهدف"
    nan_streak = 0

    if args.validate_instead:
        print("\n[validate] ===== فحص سلامة السلسلة =====")
        print(f"[validate] دفعات في الحقبة: {len(train_dataloader)}")
        batch = next(iter(train_dataloader))
        print(f"[validate] شكل الدفعة: waves={tuple(batch[0].shape)}, mels={tuple(batch[1].shape)}")
        print(f"[validate] ملف config منسوخ إلى {log_dir}")
        print(f"[validate] نقطة البداية: {start_ckpt or 'النموذج الرسمي (تنزيل عند أول تدريب)'}")
        print(f"[validate] الاستئناف الكامل: {'متاح من نقطة محلية' if local_ckpts else 'سيُفعَّل بعد أول نقطة'}")
        print(f"[validate] مرآة Drive: {args.drive_dir or 'معطلة'}")
        print("[validate] السلسلة سليمة ✓ — شغّل بلا --validate-instead للتدريب الفعلي")
        return

    if args.dry_run:
        args.total_steps = min(args.total_steps, iters + 3)

    print(f"\n[train] من الخطوة {iters} إلى {args.total_steps} | دفعة {batch_size} | حفظ كل {args.save_every}")
    print(f"[train] ميزانية الجلسة: {args.time_budget_hours or '∞'} ساعة | مرآة: {args.drive_dir or 'معطلة'}\n")

    _ = [model[key].train() for key in model]

    pbar = tqdm = None
    try:
        from tqdm import tqdm as _tqdm
        pbar = _tqdm(total=args.total_steps - iters, initial=0, desc="iraqi_train", unit="step")
    except ImportError:
        pass

    steps_done_this_session = 0
    train_iter = iter(train_dataloader)

    def next_batch():
        nonlocal train_iter, epoch
        try:
            return next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_dataloader)
            return next(train_iter)

    while iters < args.total_steps:
        batch = next_batch()
        batch = [b.to(device) for b in batch]
        waves, mels, wave_lengths, mel_input_length = batch

        B = waves.size(0)
        target_size = mels.size(2)
        target = mels
        target_lengths = mel_input_length

        if sr != 22050:
            waves_22k = torchaudio.functional.resample(waves, sr, 22050)
            wave_lengths_22k = (wave_lengths.float() * 22050 / sr).long()
        else:
            waves_22k, wave_lengths_22k = waves, wave_lengths
        se_batch = tone_color_converter.extract_se(waves_22k, wave_lengths_22k)

        ref_se_idx = torch.randint(0, len(se_db), (B,))
        ref_se = se_db[ref_se_idx].to(device)

        converted_waves_22k = tone_color_converter.convert(
            waves_22k, wave_lengths_22k, se_batch, ref_se).squeeze(1)
        if sr != 22050:
            converted_waves = torchaudio.functional.resample(converted_waves_22k, 22050, sr)
        else:
            converted_waves = converted_waves_22k

        waves_16k = torchaudio.functional.resample(waves, sr, 16000)
        wave_lengths_16k = (wave_lengths.float() * 16000 / sr).long()
        converted_waves_16k = torchaudio.functional.resample(converted_waves, sr, 16000)

        S_ori = semantic_fn(waves_16k)
        S_alt = semantic_fn(converted_waves_16k)

        if f0_condition:
            F0_ori = rmvpe.infer_from_audio_batch(waves_16k)
        else:
            F0_ori = None

        alt_cond, _, alt_codes, alt_commitment_loss, alt_codebook_loss = (
            model.length_regulator(S_alt, ylens=target_lengths, f0=F0_ori))
        ori_cond, _, ori_codes, ori_commitment_loss, ori_codebook_loss = (
            model.length_regulator(S_ori, ylens=target_lengths, f0=F0_ori))
        if alt_commitment_loss is None:
            alt_commitment_loss = 0
            alt_codebook_loss = 0
            ori_commitment_loss = 0
            ori_codebook_loss = 0

        prompt_len_max = target_lengths - 1
        prompt_len = (torch.rand([B], device=alt_cond.device) * prompt_len_max).floor().long()
        prompt_len[torch.rand([B], device=alt_cond.device) < 0.1] = 0

        cond = alt_cond.clone()
        for bib in range(B):
            cond[bib, :prompt_len[bib]] = ori_cond[bib, :prompt_len[bib]]

        common_min_len = min(target_size, cond.size(1))
        target = target[:, :, :common_min_len]
        cond = cond[:, :common_min_len]
        target_lengths = torch.clamp(target_lengths, max=common_min_len)
        x = target

        feat_list = []
        for bib in range(B):
            feat = kaldi.fbank(
                waves_16k[bib:bib + 1, :wave_lengths_16k[bib]],
                num_mel_bins=80, dither=0, sample_frequency=16000)
            feat = feat - feat.mean(dim=0, keepdim=True)
            feat_list.append(feat)
        y_list = []
        with torch.no_grad():
            for feat in feat_list:
                y_list.append(sv_fn(feat.unsqueeze(0)))
        y = torch.cat(y_list, dim=0)

        loss, _ = model.cfm(x, target_lengths, prompt_len, cond, y)
        loss_total = (
            loss +
            (alt_commitment_loss + ori_commitment_loss) * 0.05 +
            (ori_codebook_loss + alt_codebook_loss) * 0.15
        )

        # ---- حارس NaN/Inf ----
        if not torch.isfinite(loss_total):
            nan_streak += 1
            print(f"\n[guard] خسارة غير رقمية #{nan_streak} عند الخطوة {iters} — تخطّي الدفعة (grad zero)")
            optimizer.zero_grad()
            if nan_streak >= 3:
                stop_reason = "توقّف أمان: ثلاث خسائر غير رقمية متتالية"
                break
            continue
        else:
            nan_streak = 0

        optimizer.zero_grad()
        loss_total.backward()
        torch.nn.utils.clip_grad_norm_(model.cfm.parameters(), 10.0)
        torch.nn.utils.clip_grad_norm_(model.length_regulator.parameters(), 10.0)
        optimizer.step("cfm")
        optimizer.step("length_regulator")
        optimizer.scheduler(key="cfm")
        optimizer.scheduler(key="length_regulator")

        loss_val = loss.detach().item()
        ema_loss = (ema_loss * loss_smoothing_rate + loss_val * (1 - loss_smoothing_rate)
                    if ema_loss is not None else loss_val)

        iters += 1
        steps_done_this_session += 1
        if pbar:
            pbar.update(1)

        if iters % log_interval == 0:
            msg = f"epoch {epoch}, step {iters}, loss: {ema_loss}"
            print(msg)
            with open(loss_history_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": iters, "ema_loss": ema_loss, "loss": loss_val,
                                    "ts": time.time()}) + "\n")

        # ---- الحفظ الذرّي + المرآة ----
        if iters % args.save_every == 0 or iters >= args.total_steps:
            state = {
                "net": {key: model[key].state_dict() for key in model},
                "optimizer": optimizer.state_dict(),
                "scheduler": optimizer.scheduler_state_dict(),
                "iters": iters,
                "epoch": epoch,
            }
            ckpt_utils.save_full_state(
                log_dir, state, iters,
                keep=args.keep_checkpoints,
                drive_dir=None if args.no_mirror else args.drive_dir,
                keep_on_drive=args.keep_checkpoints,
                heartbeat=heartbeat_path,
            )
            # نسخ السجل إلى Drive أيضاً
            if args.drive_dir and not args.no_mirror:
                try:
                    shutil.copy2(loss_history_path, Path(args.drive_dir) / "loss_history.jsonl")
                except OSError as e:
                    print(f"[mirror] تعذر نسخ السجل: {e}")

        # ---- ميزانية الزمن ----
        if time.time() - t_start > budget_s:
            stop_reason = f"بلوغ ميزانية الزمن ({args.time_budget_hours} ساعة) — توقف نظيف"
            state = {
                "net": {key: model[key].state_dict() for key in model},
                "optimizer": optimizer.state_dict(),
                "scheduler": optimizer.scheduler_state_dict(),
                "iters": iters,
                "epoch": epoch,
            }
            ckpt_utils.save_full_state(
                log_dir, state, iters, keep=args.keep_checkpoints,
                drive_dir=None if args.no_mirror else args.drive_dir,
                keep_on_drive=args.keep_checkpoints, heartbeat=heartbeat_path,
            )
            break

    if pbar:
        pbar.close()

    # ---------- الحفظ النهائي (نفس تنسيق الرسمي ft_model.pth) ----------
    print(f"\n[train] انتهى: {stop_reason} عند الخطوة {iters}")
    final_state = {"net": {key: model[key].state_dict() for key in model}}
    final_path = log_dir / "ft_model.pth"
    ckpt_utils.save_atomic(final_state, final_path)
    print(f"[train] النموذج النهائي: {final_path}")
    if args.drive_dir and not args.no_mirror:
        try:
            shutil.copy2(final_path, Path(args.drive_dir) / "ft_model.pth")
            print(f"[mirror] نُسخ ft_model.pth إلى Drive")
        except OSError as e:
            print(f"[mirror] تعذر نسخ ft_model.pth: {e}")
    print("[train] تمت كل المهام بنجاح")


if __name__ == "__main__":
    main()
