"""The Arabic-RTL Gradio control panel: full control over the model.

Tabs
----
1. ``التشغيل اللحظي``  — microphone streaming conversion (audio-to-audio).
2. ``تحويل ملف``      — upload a file, convert it with the same engine path.
3. ``الأصوات المستهدفة`` — register / preview / delete target voices.
4. ``النموذج``        — engine info, latency profile, checkpoint hot-swap.
5. ``التدريب``        — launch / watch / resume fine-tuning jobs.
6. ``النظام``         — health, GPU, sessions, versions.

The panel talks to the *application layer* directly (same components object
the REST/WS API uses), so everything you can do here is also scriptable.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import gradio as gr

from rafidain import __version__
from rafidain.core.logging import get_logger
from rafidain.domain.value_objects import SAMPLE_RATE_IN
from rafidain.webui.convert import (
    close_mic_stream,
    convert_file,
    stream_mic_chunk,
)

logger = get_logger(__name__)

PROFILE_LABELS = {
    "ultra_low": "ultra_low — أدنى تأخير (~180ms خوارزمي)",
    "balanced": "balanced — التوازن الرسمي (~430ms شامل الاستدلال)",
    "quality": "quality — أعلى جودة (غير تفاعلي)",
}
_ORIGINAL_WEIGHTS = "الأوزان الأصلية من HuggingFace (seed-uvit-tat-xlsr-tiny)"
_REPO_ROOT = Path(__file__).resolve().parents[2]

_RTL_CSS = """
.gradio-container { direction: rtl; text-align: right; }
.gradio-container .prose { text-align: right; }
.tabitem { direction: rtl; }
footer {visibility: hidden}
#rafidain-header {text-align: center; padding: 6px 0 0 0;}
#rafidain-header h1 {margin: 0; font-size: 1.6em;}
#rafidain-header p {margin: 2px 0 8px 0; opacity: .8;}
.mic-out audio {width: 100%;}
"""


def _components(app):
    """Resolve the wired components — only valid once the app has started."""
    components = getattr(app.state, "components", None)
    if components is None:
        raise gr.Error("الخدمة لم تكمل الإقلاع بعد — انتظر ثوانٍ ثم أعد المحاولة")
    return components


def _components_or_none(app):
    """Layout-time variant: None before startup (data loads via demo.load)."""
    return getattr(app.state, "components", None)


def _run(coro):
    """Await an async application call from any gradio callback context."""
    try:
        return asyncio.run(coro)
    except RuntimeError:  # already inside a running event loop
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, coro).result()


def _voices_choices(app) -> list[tuple[str, str]]:
    if _components_or_none(app) is None:  # layout-time: filled by demo.load
        return []

    async def _list():
        comp = _components(app)
        voices = await comp.voice_service.list()
        return [
            (f"{v.name or v.voice_id} ({v.duration_s:.0f}s)", v.voice_id)
            for v in voices
        ]

    try:
        return _run(_list())
    except gr.Error:
        raise
    except Exception as exc:  # noqa: BLE001
        raise gr.Error(f"تعذّر جلب الأصوات: {exc}") from exc


def _import_checkpoint_sync():
    """Import finetune.checkpoint_sync regardless of the install layout."""
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    from finetune.checkpoint_sync import (  # noqa: PLC0415
        default_remote_root,
        detect_google_drive,
        latest_checkpoint,
        run_dir_for,
    )

    return default_remote_root, detect_google_drive, latest_checkpoint, run_dir_for


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------


def build_webui(app):  # noqa: PLR0915 — a UI builder is naturally long
    build_start = time.time()

    # ----------------------------- realtime mic handlers -------------------
    async def _mic_stream(audio, state, voice_id, profile_key):
        """gradio streaming generator: mic chunk → converted chunk."""
        comp = _components(app)
        if audio is None:
            yield None, state, "بانتظار الصوت من الميكروفون…"
            return
        sample_rate, data = audio
        state, out = await stream_mic_chunk(
            comp, state, data, int(sample_rate), voice_id, profile_key
        )
        sr_out = comp.engine.sr_out
        if state is not None and state.errors:
            note = f"⚠ {state.errors[-1]}"
        elif out.size:
            note = (
                f"🔄 يبث الآن — كتل: {state.blocks_done} · "
                f"داخل: {state.samples_in / SAMPLE_RATE_IN:.1f}s · "
                f"خارج: {state.samples_out / sr_out:.1f}s"
            )
        else:
            note = "🎙 يستمع… (المحرك يجمع السياق الأولي)"
        yield (sr_out, out) if out.size else None, state, note

    async def _mic_stop(state):
        tail = await close_mic_stream(state)
        extra = f" — آخر ذيل: {tail.size} عيّنة" if tail.size else ""
        yield None, f"⏹ توقّف البث{extra}"

    async def _mic_restart(state):
        await close_mic_stream(state)
        return None, None, "🔄 جاهز لبث جديد — تحدث الآن."

    # ----------------------------- file conversion -------------------------
    async def _convert_file(file_path, voice_id, profile_key):
        if not file_path:
            raise gr.Error("ارفع ملفاً أولاً (wav/mp3/flac/m4a/ogg…)")
        comp = _components(app)
        data = Path(file_path).read_bytes()
        out_path, stats = await convert_file(comp, data, voice_id, profile_key)
        note = (
            f"✅ تم التحويل: {stats['in_seconds']}s ← {stats['out_seconds']}s "
            f"({stats['profile']}، خرج {stats['sample_rate_out']}Hz)"
        )
        return str(out_path), note

    # ----------------------------- voice management ------------------------
    async def _register_voice(file_path, name, notes):
        if not file_path:
            raise gr.Error("ارفع أو سجّل صوتاً مرجعياً (1–30 ثانية صوت نظيف)")
        comp = _components(app)
        data = Path(file_path).read_bytes()
        voice = await comp.voice_service.register(
            name=name or Path(file_path).stem, audio=data, notes=notes
        )
        return (
            f"✅ سُجّل الصوت «{voice.name}» ({voice.duration_s:.1f}s) — {voice.voice_id}",
            _voices_table(),
        )

    def _voices_table() -> list[list]:
        if _components_or_none(app) is None:
            return []

        async def _table():
            comp = _components(app)
            voices = await comp.voice_service.list()
            return [
                [v.voice_id, v.name, f"{v.duration_s:.1f}s", v.created_at_iso[:19]]
                for v in voices
            ]

        return _run(_table()) or []

    async def _voice_preview(voice_id):
        if not voice_id:
            raise gr.Error("اختر صوتاً أولاً")
        comp = _components(app)
        await comp.voice_service.get(voice_id)
        path = Path(comp.settings.voices_dir) / voice_id / "ref_22050.wav"
        if not path.exists():
            raise gr.Error("لا توجد معاينة محفوظة لهذا الصوت")
        return str(path)

    async def _voice_delete(voice_id):
        if not voice_id:
            raise gr.Error("اختر صوتاً أولاً")
        comp = _components(app)
        await comp.voice_service.delete(voice_id)
        return f"🗑 حُذف الصوت {voice_id}", _voices_table()

    def _refresh_voice_dropdowns():
        choices = _voices_choices(app)
        update = gr.update(choices=choices)
        return update, gr.update(choices=choices), gr.update(choices=choices)

    # ----------------------------- engine control --------------------------
    def _engine_info() -> str:
        comp = _components_or_none(app)
        if comp is None:
            return "_يُحمّل عند فتح الصفحة…_"
        engine = comp.engine
        device = "?"
        if hasattr(engine, "describe"):
            device = engine.describe().get("device", "?")
        lines = [
            f"**المحرك:** `{engine.impl}`",
            f"**جاهز:** {'✅' if engine.is_ready() else '⏳ (غير محمّل)'}",
            f"**الجهاز:** `{device}`",
            f"**معدل الخرج:** {engine.sr_out} Hz",
        ]
        if comp.warmup_stats:
            stats = json.dumps(comp.warmup_stats, ensure_ascii=False)
            lines.append(f"**الإحماء:** `{stats}`")
        es = comp.settings.engine
        ckpt = es.local_checkpoint or f"{es.hf_repo}/{es.checkpoint}"
        lines.append(f"**الشيك بوينت:** `{ckpt}`")
        lines.append(f"**بروفايل البث الافتراضي:** `{comp.settings.default_profile}`")
        return "\n\n".join(lines)

    def _scan_checkpoints() -> list[str]:
        comp = _components_or_none(app)
        items = [_ORIGINAL_WEIGHTS]
        seen: set[Path] = set()
        cache_dir = (
            Path(comp.settings.engine.cache_dir) if comp else Path("./checkpoints")
        )
        roots = [Path("runs"), cache_dir, Path("third_party/seed-vc/runs")]
        for root in roots:
            if not root.is_dir():
                continue
            for ft in root.rglob("ft_model.pth"):
                if ft not in seen:
                    seen.add(ft)
                    items.append(str(ft))
        return items

    def _reload_engine(impl, device, fp16, ckpt_choice, ckpt_custom, profile_key):
        comp = _components(app)
        from rafidain.infrastructure.engine_registry import build_engine

        es = comp.settings.engine.model_copy(
            update={
                "impl": impl,
                "device": device,
                "fp16": bool(fp16) and impl == "seedvc",
                "local_checkpoint": None,
                "local_config": None,
            }
        )
        chosen = (ckpt_custom or "").strip() or ckpt_choice
        if chosen and chosen != _ORIGINAL_WEIGHTS:
            ckpt = Path(chosen)
            if not ckpt.is_file():
                raise gr.Error(f"الشيك بوينت غير موجود: {ckpt}")
            es = es.model_copy(update={"local_checkpoint": str(ckpt)})
            cfg = ckpt.parent / "config_patched.yml"
            if cfg.is_file():
                es = es.model_copy(update={"local_config": str(cfg)})
        settings = comp.settings.model_copy(
            update={"engine": es, "default_profile": profile_key}
        )

        def _heavy():
            engine = build_engine(settings)
            engine.load()
            if settings.engine.warmup:
                engine.warmup(settings.profile(profile_key))
            return engine

        try:
            engine = _run(asyncio.to_thread(_heavy))
        except Exception as exc:  # noqa: BLE001
            raise gr.Error(f"فشل تحميل المحرك: {exc}") from exc
        comp.engine = engine
        comp.settings = settings
        comp.warmup_stats = {"impl": engine.impl, "reloaded_via": "/web"}
        return (
            _engine_info(),
            gr.update(choices=_scan_checkpoints(), value=chosen or _ORIGINAL_WEIGHTS),
            "✅ أعيد بناء المحرك بالإعدادات الجديدة (الجلسات القادمة تستخدمه فوراً)",
        )

    # ----------------------------- training --------------------------------
    async def _train_submit(
        dataset_dir, run_name, max_steps, save_every, batch, prepare, gdrive_mode
    ):
        if not dataset_dir or not Path(dataset_dir).is_dir():
            raise gr.Error(f"مجلد البيانات غير موجود: {dataset_dir}")
        comp = _components(app)
        from rafidain.application.finetune_service import FinetuneParams

        params = FinetuneParams(
            dataset_dir=str(dataset_dir),
            run_name=run_name or "iraqi_run",
            max_steps=int(max_steps),
            save_every=int(save_every),
            batch_size=int(batch),
            prepare=bool(prepare),
            extra={"gdrive": gdrive_mode},
        )
        job_id = await comp.finetune_service.submit(params)
        return job_id, f"🚀 أُطلقت مهمة التدريب `{job_id}` — راقب التقدم أدناه"

    def _job_tick(job_id):
        if not job_id:
            return 0.0, "لا توجد مهمة", "", gr.update(interactive=False)

        async def _status():
            comp = _components(app)
            return await comp.finetune_service.status(job_id)

        record = _run(_status())
        if record is None:
            return 0.0, "مهمة غير معروفة", "", gr.update(interactive=False)
        progress = round(float(record.progress or 0.0) * 100, 1)
        logs = "\n".join(list(record.logs)[-30:])
        status = f"{record.state} · {record.message or ''}"
        return (
            progress,
            status,
            logs or "(لا سجلات بعد)",
            gr.update(interactive=record.state in ("queued", "running")),
        )

    async def _train_cancel(job_id):
        if not job_id:
            return "لا توجد مهمة لإلغائها"
        comp = _components(app)
        ok = await comp.finetune_service.cancel(job_id)
        return "🗑 أُلغيت المهمة" if ok else "تعذّر الإلغاء (ربما انتهت)"

    def _resume_info(run_name):
        _, detect_google_drive, latest_checkpoint, run_dir_for = _import_checkpoint_sync()
        from rafidain.core.config import seed_vc_root

        name = run_name or "iraqi_run"
        root = seed_vc_root()
        local_dir = run_dir_for(root, name)
        gdrive = detect_google_drive(auto_mount=False)
        remote_dir = None
        if gdrive:
            remote_root = gdrive / "RafidainRuns"
            remote_dir = remote_root / name
        local = latest_checkpoint(local_dir)
        remote = latest_checkpoint(remote_dir) if remote_dir else None
        lines = [
            f"**مجلد التشغيل:** `{local_dir}`",
            f"**أحدث شيك بوينت محلي:** {local.name if local else '— لا شيء'}",
            f"**Google Drive:** {'مركّب ✅' if gdrive else 'غير مركّب (استئناف محلي فقط)'}",
        ]
        if remote_dir is not None:
            lines.append(f"**أحدث نسخة على Drive:** {remote.name if remote else '—'}")
        lines.append(
            "\n> التدريب يستأنف تلقائياً من أحدث شيك بوينت (محلي أو من Drive) "
            "عند إعادة تشغيل نفس `run_name`."
        )
        return "\n".join(lines)

    # ----------------------------- system ----------------------------------
    def _system_info() -> str:
        comp = _components_or_none(app)
        if comp is None:
            return "_يُحمّل عند فتح الصفحة…_"
        lines = [
            f"**النسخة:** rafidain `{__version__}` · gradio `{gr.__version__}`",
            f"**الجلسات النشطة:** {comp.session_manager.active_count}",
            f"**المحرك:** `{comp.engine.impl}` — جاهز: "
            f"{'✅' if comp.engine.is_ready() else '⏳'}",
        ]
        try:
            import torch

            cuda = torch.cuda.is_available()
            lines.append(f"**torch:** `{torch.__version__}` · CUDA: {'✅' if cuda else '❌'}`")
            if cuda:
                lines.append(f"**الكرت:** `{torch.cuda.get_device_name(0)}`")
        except ImportError:
            lines.append("**torch:** غير مثبّت (بيئة خفيفة / محرك وهمي)")
        lines.append(
            f"**المسارات:** بيانات `{comp.settings.data_dir}` · "
            f"نقاط `{comp.settings.engine.cache_dir}`"
        )
        lines.append(f"**زمن بناء الواجهة:** {time.time() - build_start:.1f}s")
        return "\n\n".join(lines)

    # =====================================================================
    # Layout
    # =====================================================================
    theme = gr.themes.Soft(primary_hue="teal", neutral_hue="slate")
    with gr.Blocks(theme=theme, title="رافدَين — لوحة التحكم", css=_RTL_CSS) as demo:
        gr.HTML(
            """
            <div id="rafidain-header" dir="rtl">
              <h1>🏛️ رافدَين — لوحة التحكم بتحويل الصوت اللحظي</h1>
              <p>Seed-VC · seed-uvit-tat-xlsr-tiny · للهجة العراقية —
                 <a href="/live" target="_blank">صفحة البث الحي الحقيقية ↗</a> ·
                 <a href="/docs" target="_blank">REST/WS API ↗</a></p>
            </div>
            """
        )

        with gr.Tabs():
            # ---------------------------------------------------- realtime mic
            with gr.Tab("🎙️ التشغيل اللحظي"):
                with gr.Row():
                    with gr.Column(scale=1):
                        mic_voice = gr.Dropdown(
                            choices=_voices_choices(app),
                            label="الصوت الهدف (سجّل أصواتاً من تبويب «الأصوات»)",
                            info="يجب اختيار صوت قبل البث",
                        )
                        mic_profile = gr.Radio(
                            choices=list(PROFILE_LABELS),
                            value="balanced",
                            label="ملف زمن الوصول (Latency Profile)",
                            info="balanced ≈ 430ms شاملاً الاستدلال على T4",
                        )
                        mic_note = gr.Markdown("بانتظار البث…")
                    with gr.Column(scale=2):
                        mic_in = gr.Audio(
                            sources=["microphone"],
                            type="numpy",
                            streaming=True,
                            label="تحدّث هنا — التحويل يجري لحظياً",
                            show_download_button=False,
                        )
                        mic_out = gr.Audio(
                            streaming=True,
                            autoplay=True,
                            label="الصوت المحوّل (يخرج لحظياً)",
                            elem_classes=["mic-out"],
                        )
                mic_state = gr.State(None)
                mic_in.stream(
                    _mic_stream,
                    inputs=[mic_in, mic_state, mic_voice, mic_profile],
                    outputs=[mic_out, mic_state, mic_note],
                )
                mic_in.stop(
                    _mic_stop, inputs=[mic_state], outputs=[mic_out, mic_note]
                )
                mic_in.clear(
                    _mic_restart,
                    inputs=[mic_state],
                    outputs=[mic_out, mic_state, mic_note],
                )

            # ------------------------------------------------------- file tab
            with gr.Tab("📁 تحويل ملف"):
                with gr.Row():
                    with gr.Column(scale=1):
                        file_voice = gr.Dropdown(
                            choices=_voices_choices(app),
                            label="الصوت الهدف",
                        )
                        file_profile = gr.Radio(
                            choices=list(PROFILE_LABELS),
                            value="balanced",
                            label="ملف زمن الوصول",
                            info="quality أبطأ لكنه أدق للملفات غير التفاعلية",
                        )
                        file_btn = gr.Button("حوّل الملف", variant="primary")
                        file_note = gr.Markdown("الصيغ المدعومة: wav mp3 flac m4a opus ogg")
                    with gr.Column(scale=1):
                        file_in = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="ارفع أو سجّل الملف المصدر",
                        )
                        file_out = gr.Audio(
                            type="filepath",
                            label="الملف المحوّل (wav)",
                            show_download_button=True,
                        )
                file_btn.click(
                    _convert_file,
                    inputs=[file_in, file_voice, file_profile],
                    outputs=[file_out, file_note],
                )

            # ---------------------------------------------------- voices tab
            with gr.Tab("🎭 الأصوات المستهدفة"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(
                            "سجّل صوتاً هدفاً من 1–30 ثانية صوت **نظيف** للمتحدث "
                            "الذي تريد التحويل إليه. يتطلب موافقته الصريحة."
                        )
                        voice_file = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="الصوت المرجعي",
                        )
                        voice_name = gr.Textbox(
                            label="اسم الصوت", placeholder="مثال: أبو علي"
                        )
                        voice_notes = gr.Textbox(label="ملاحظات (اختياري)")
                        voice_btn = gr.Button("سجّل الصوت", variant="primary")
                        voice_msg = gr.Markdown("")
                    with gr.Column():
                        voices_table = gr.Dataframe(
                            headers=["المعرف", "الاسم", "المدة", "أُنشئ"],
                            label="الأصوات المسجّلة",
                            value=_voices_table(),
                            interactive=False,
                        )
                        with gr.Row():
                            voice_pick = gr.Dropdown(
                                choices=_voices_choices(app), label="اختر صوتاً"
                            )
                            voice_preview_btn = gr.Button("▶ استمع")
                            voice_delete_btn = gr.Button("🗑 احذف", variant="stop")
                        voice_preview = gr.Audio(
                            type="filepath", label="معاينة الصوت المرجعي"
                        )
                voice_btn.click(
                    _register_voice,
                    inputs=[voice_file, voice_name, voice_notes],
                    outputs=[voice_msg, voices_table],
                ).then(
                    _refresh_voice_dropdowns,
                    outputs=[mic_voice, file_voice, voice_pick],
                )
                voice_preview_btn.click(
                    _voice_preview, inputs=[voice_pick], outputs=[voice_preview]
                )
                voice_delete_btn.click(
                    _voice_delete,
                    inputs=[voice_pick],
                    outputs=[voice_msg, voices_table],
                ).then(
                    _refresh_voice_dropdowns,
                    outputs=[mic_voice, file_voice, voice_pick],
                )

            # ----------------------------------------------------- model tab
            with gr.Tab("🧠 النموذج"):
                engine_md = gr.Markdown(_engine_info())
                with gr.Row():
                    with gr.Column():
                        eng_impl = gr.Radio(
                            choices=["mock", "seedvc"],
                            value=app.state.settings.engine.impl,
                            label="تنفيذ المحرك",
                            info="mock = محاكاة بدون GPU · seedvc = النموذج الحقيقي",
                        )
                        eng_device = gr.Radio(
                            choices=["auto", "cuda", "cpu", "mps"],
                            value=app.state.settings.engine.device,
                            label="الجهاز",
                        )
                        eng_fp16 = gr.Checkbox(
                            value=app.state.settings.engine.fp16,
                            label="FP16 (أسرع على CUDA)",
                        )
                        eng_profile = gr.Radio(
                            choices=list(PROFILE_LABELS),
                            value=app.state.settings.default_profile,
                            label="بروفايل البث الافتراضي",
                        )
                    with gr.Column():
                        eng_ckpt = gr.Dropdown(
                            choices=_scan_checkpoints(),
                            value=(
                                app.state.settings.engine.local_checkpoint
                                or _ORIGINAL_WEIGHTS
                            ),
                            label="الشيك بوينت",
                            info="تظهر تلقائياً نماذج runs/*/ft_model.pth بعد التدريب",
                        )
                        eng_ckpt_custom = gr.Textbox(
                            label="أو مسار شيك بوينت يدوي (.pth)",
                            placeholder="./runs/iraqi_v1/ft_model.pth",
                        )
                        eng_reload_btn = gr.Button(
                            "♻️ أعد بناء المحرك بالإعدادات", variant="primary"
                        )
                        eng_msg = gr.Markdown("")
                eng_reload_btn.click(
                    _reload_engine,
                    inputs=[
                        eng_impl, eng_device, eng_fp16, eng_ckpt, eng_ckpt_custom, eng_profile
                    ],
                    outputs=[engine_md, eng_ckpt, eng_msg],
                )

            # -------------------------------------------------- training tab
            with gr.Tab("🎓 التدريب والاستئناف"):
                gr.Markdown(
                    "ضبط دقيق للهجة العراقية — نفس مسار `run_finetune.py`. "
                    "الشيك بوينتات تُحفظ تلقائياً إلى Google Drive على Colab "
                    "ويستأنف التدريب من آخرها عند أي انقطاع."
                )
                with gr.Row():
                    with gr.Column():
                        train_dataset = gr.Textbox(
                            label="مجلد البيانات (مسار كامل)",
                            placeholder="/content/iraqi_seedvc/data/iraqi_finetune",
                        )
                        train_run = gr.Textbox(
                            value="iraqi_v1", label="اسم التشغيل (run_name)"
                        )
                        train_steps = gr.Number(value=2000, label="أقصى عدد خطوات", precision=0)
                        train_save = gr.Number(value=250, label="احفظ كل (خطوة)", precision=0)
                        train_batch = gr.Number(value=2, label="حجم الدفعة", precision=0)
                        train_prepare = gr.Checkbox(value=True, label="جهّز البيانات أولاً")
                        train_gdrive = gr.Radio(
                            choices=["auto", "on", "off"],
                            value="auto",
                            label="مزامنة Google Drive",
                        )
                        train_btn = gr.Button("🚀 أطلق التدريب", variant="primary")
                        train_msg = gr.Markdown("")
                    with gr.Column():
                        train_job = gr.Textbox(
                            label="معرف المهمة (job_id)", interactive=False
                        )
                        train_progress = gr.Slider(
                            0, 100, value=0, label="التقدم (%)", interactive=False
                        )
                        train_status = gr.Markdown("—")
                        train_logs = gr.Textbox(
                            label="آخر السجلات", lines=14, max_lines=14, interactive=False
                        )
                        train_cancel = gr.Button(
                            "⛔ ألغِ المهمة", variant="stop", interactive=False
                        )
                        timer = gr.Timer(value=2, active=False)
                train_resume_md = gr.Markdown(_resume_info("iraqi_v1"))
                train_run.change(_resume_info, inputs=[train_run], outputs=[train_resume_md])

                train_btn.click(
                    _train_submit,
                    inputs=[
                        train_dataset, train_run, train_steps, train_save,
                        train_batch, train_prepare, train_gdrive,
                    ],
                    outputs=[train_job, train_msg],
                ).then(lambda: gr.update(active=True), outputs=[timer])
                timer.tick(
                    _job_tick,
                    inputs=[train_job],
                    outputs=[train_progress, train_status, train_logs, train_cancel],
                )
                train_cancel.click(_train_cancel, inputs=[train_job], outputs=[train_status])

            # ---------------------------------------------------- system tab
            with gr.Tab("🖥️ النظام"):
                sys_md = gr.Markdown(_system_info())
                sys_refresh = gr.Button("🔄 حدّث")
                sys_refresh.click(_system_info, outputs=[sys_md])

        # Populate everything once the page opens (components exist by then).
        def _initial_load():
            if _components_or_none(app) is None:
                return (
                    gr.update(), gr.update(), gr.update(),
                    gr.update(), gr.update(), gr.update(), gr.update(),
                )
            choices = _voices_choices(app)
            update = gr.update(choices=choices)
            return (
                update,
                update,
                update,
                _voices_table(),
                _engine_info(),
                _resume_info("iraqi_v1"),
                _system_info(),
            )

        demo.load(
            _initial_load,
            outputs=[
                mic_voice, file_voice, voice_pick, voices_table,
                engine_md, train_resume_md, sys_md,
            ],
        )

    return demo
