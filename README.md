<div dir="rtl">

# رافِدَين (Rafidain) — نظام تحويل الصوت اللحظي للهجة العراقية

**باك-إند إنتاجي، بمعمارية Clean Architecture ملتزمة بمبادئ SOLID، لتحويل الصوت (Voice Conversion) في المكالمات المباشرة — مبني على نموذج [Seed-VC](https://github.com/Plachtta/seed-vc)‏ `seed-uvit-tat-xlsr-tiny` ومُهيّأ للهجة العراقية مع خط ضبط دقيق (Fine-tuning) كامل.**

| | |
|---|---|
| النموذج | seed-uvit-tat-xlsr-tiny (25M باراميتر، 22050Hz، مشفّر محتوى XLSR-large، فوكودر HiFT) |
| زمن الوصول الحرجي | **~430ms** شاملاً الاستدلال (قياس منشور رسمياً للنموذج نفسه بنفس إعدادات البروفايل `balanced`) |
| بروتوكول النقل | REST لدورة الحياة + WebSocket للصوت اللحظي (PCM s16le أحادي 16kHz) |
| لغة التنفيذ | Python 3.10+ / FastAPI / asyncio |
| الاختبارات | 42 اختباراً (وحدات + تكامل كامل عبر WebSocket) — كلها تمر بدون GPU |

---

## 1) لماذا هذا المشروع؟

تحويل الصوت اللحظي في المكالمات يتطلب أكثر من نموذج ذكي: يتطلب **نظاماً** يضبط الذاكرة والتزامن وزمن الوصول والضغط الخلفي (Backpressure) والمراقبة. «رافدَين» يقدّم هذا النظام:

- **بث لحظي حقيقي**: كتل صوتية 180ms مع crossfade متساوي القدرة، انظر [ميزانية زمن الوصول](ARCHITECTURE.md#7-ميزانية-زمن-الوصول-latency-budget).
- **معمارية نظيفة قابلة للصيانة**: طبقات مفصولة (نطاق / تطبيق / بنية تحتية / واجهات) بمنافذ Ports واضحة — التفاصيل في [ARCHITECTURE.md](ARCHITECTURE.md) مع جدول مطابقة SOLID وسجلات قرارات ADR.
- **محرك قابل للاستبدال**: محرك Seed-VC الحقيقي ومحرك وهمي Mock بنفس الواجهة — التطوير والاختبار والـ CI تعمل بلا GPU إطلاقاً.
- **ضبط دقيق للهجة العراقية**: خط كامل لتجهيز البيانات + التدريب + التقييم + النشر — دليل تفصيلي في [FINETUNING.md](FINETUNING.md).
- **إنتاجية جاهزة**: مصادقة API-Key، مقاييس Prometheus، سجلات JSON منظمة، فحوصات Health/Readiness، Docker + docker-compose (GPU)، CI عبر GitHub Actions.
- **ضغط خلفي محسوب**: طابور إرسال محدود يُسقط أقدم الكتل الصوتية (لا رسائل التحكم) ويقيس الأحداث — لا انفجار ذاكرة مع العملاء البطيئين.

## 2) البنية في سطور

```
REST (voices/sessions/finetune)          WebSocket (البث اللحظي)
        │                                        │
        ▼                                        ▼
┌─────────────────── طبقة الواجهات api ───────────────────┐
│  http_* (REST)      ws/stream.py (قارئ + كاتب متوازيان)  │
├────────────────── طبقة التطبيق application ──────────────┤
│  SessionManager  StreamPipeline  VoiceService  Finetune  │
├────────────────── النطاق domain (نقي) ───────────────────┤
│  Session/SessionState/LatencyProfile + Ports (بروتوكولات) │
├────────────────── البنية التحتية infrastructure ─────────┤
│  seedvc/loader+engine  mock_engine  audio/*  stores  jobs │
└───────────────────────────────────────────────────────────┘
```

الاعتماد يتجه للداخل فقط (Dependency Rule). التفاصيل والأسباب: [ARCHITECTURE.md](ARCHITECTURE.md).

## 3) البدء السريع

```bash
git clone https://github.com/Alqudimi/rafidain && cd rafidain
```

> **أسهل طريقة — المثبّتات الجاهزة** (`scripts/install/`): تثبيت + تهيئة + تشغيل كامل بأمر واحد
> لثلاث بيئات. التفاصيل الكاملة وخيارات الاستضافة في `INSTALL-SCRIPTS.md`.
>
> ```bash
> # Google Colab (خليّة واحدة، بعد اختيار GPU T4):
> !bash scripts/install/setup_colab.sh            # أو عبر URL — انظر INSTALL-SCRIPTS.md
>
> # Ubuntu / Debian (يستخدم venv، ويكتشف GPU تلقائياً، ودعم systemd عبر --service):
> bash scripts/install/setup_ubuntu.sh            # أو: curl -fsSL <url> | bash -s -- --service
>
> # Windows 10/11 (PowerShell عادي بلا صلاحيات مدير):
> powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install\setup_windows.ps1
> ```
>
> السكربتات تكتشف كل شيء تلقائياً (GPU/CPU، إصدار CUDA، Python، ffmpeg)، تثبّت الـ Pin
> الموثّق `torch==2.6.0`، تنزّل النقاط، تولّد `.env` مع مفتاح API عشوائي، تُشغّل الاختبارات
> الـ 42، ترفع الخادم، وتطبع روابط التجربة — وعلى Colab تفتح نفقاً عاماً `trycloudflare.com`.

### أ. تجربة فورية بدون GPU (المحرك الوهمي — دقيقتان)

```bash
cd rafidain
make install-dev          # المتطلبات الخفيفة فقط (بدون torch)
make serve-mock           # يشغّل الخادم على :8000 بالمحرك الوهمي
# في طرفية ثانية:
uv run python scripts/smoke_ws_client.py --url http://127.0.0.1:8000 --seconds 3
```

الوثائق التفاعلية: `http://127.0.0.1:8000/docs`

### ب. التشغيل الإنتاجي (GPU)

```bash
make seedvc                                   # استنساخ مستودع Seed-VC
make install ENGINE=seedvc      # uv يكتشف GPU/CUDA ويثبت العجلة المقفلة المناسبة
make download-models                          # تنزيل النقاط من HuggingFace مسبقاً
cp .env.example .env                          # ثم عدّل القيم
make serve                                    # uvicorn بالمحرك الحقيقي
```

> متغيرات التشغيل الحرجة: `RAFIDAIN__ENGINE__DEVICE=cuda` و `RAFIDAIN__API_KEYS=["<مفتاحك>"]` (صيغة JSON؛ لا تترك المصادقة معطلة خارج التطوير).

### ج. Docker

```bash
RAFIDAIN__API_KEYS='["change-me"]' docker compose -f docker/docker-compose.yml up -d
# مع عامل الضبط الدقيق (GPU منفصلة):
RAFIDAIN_DATASETS_DIR=./datasets RAFIDAIN_RUNS_DIR=./runs \
  docker compose -f docker/docker-compose.yml --profile finetune up -d
```

التفاصيل الكاملة (الأحجام، الشبكة، التوسع): [DEPLOYMENT.md](DEPLOYMENT.md).

## 4) بروتوكول البث اللحظي

**1)** أنشئ صوتاً هدفاً (5–30 ثانية صوت نظيف للمتحدث المطلوب):

```bash
curl -X POST http://host/api/v1/voices \
  -H "X-API-Key: ..." \
  -F "file=@reference.wav" -F "name=الصوت-الهدف"
```

**2)** أنشئ جلسة:

```bash
curl -X POST http://host/api/v1/sessions \
  -H "X-API-Key: ..." -H "Content-Type: application/json" \
  -d '{"voice_id": "<id>", "latency_profile": "balanced"}'
# → {"session_id": "...", "ws_url": "ws://host/api/v1/stream/<id>"}
```

**3)** اربط WebSocket وأرسل/استقبل الصوت (PCM s16le أحادي 16kHz):

```python
import asyncio, json, websockets

async def main(ws_url, pcm_16k: bytes):
    async with websockets.connect(ws_url) as ws:
        print(json.loads(await ws.recv()))                  # session.ready
        await ws.send(json.dumps({"type": "stream.start"})) # stream.started
        frame = 16000 * 20 // 1000 * 2                      # إطار 20ms
        for i in range(0, len(pcm_16k) - frame, frame):
            await ws.send(pcm_16k[i:i + frame])             # صوت الدخول
            out = await ws.recv()                           # الصوت المحوّل
            await asyncio.sleep(0.02)                       # إيقاع زمن-حقيقي
        await ws.send(json.dumps({"type": "metrics.poll"})) # مقاييس لحظية
        await ws.send(json.dumps({"type": "stream.stop"}))  # stream.stopped
```

رسائل التحكم: `stream.start/pause/resume/stop` و `metrics.poll`. الخادم يرسل: `session.ready`, `stream.*`, `metrics.snapshot`, `warning`, `error`, `session.closed`. رموز إغلاق WS: `4401` مصادقة، `4404` جلسة مجهولة، `4410` مشغولة، `4429` منتهية، `4403` خطأ بروتوكول.

## 5) ملفات زمن الوصول (Latency Profiles)

| البروفايل | block | crossfade | سياق أيمن | خطوات التدفق | الزمن الخوارزمي | الاستخدام |
|---|---|---|---|---|---|---|
| `ultra_low` | 120ms | 40ms | 20ms | 4 (بدون CFG) | 180ms | أدنى تأخير ممكن |
| `balanced` | 180ms | 40ms | 20ms | 10 | 240ms + استدلال ≈ **430ms** | المكالمات (موثّق رسمياً) |
| `quality` | 500ms | 100ms | 60ms | 25 | 660ms | جلسات غير تفاعلية |

> الجدول مشتق من منحنى القياس المنشور في مستودع Seed-VC للنموذج tiny نفسه. لقياس جهازك: `make bench`.

## 6) ضبط الهجة العراقية

النموذج zero-shot أصلاً، لكن الضبط الدقيق على كلام عراقي يرفع جودة المخرج (طبيعية اللهجة + تشابه المتحدث). 

### أ. المسار الآلي الكامل (سكربتات مستقلة بأمر واحد — curl-bash مباشر)

سكربتات مستقلة تماماً (بيانات + تدريب + حفظ أوزان ذكي، مع أشرطة تقدم حية لكل مرحلة: 
التنزيلات بالنسبة والسرعة/ETA، والتدريب خطوة-بخطوة) — تُشغّل من المستودع نفسه دون استنساخ مسبق:

```bash
# Ubuntu 20.04+ (يتطلب NVIDIA GPU للتدريب؛ بدون GPU يجهّز البيئة والبيانات فقط):
curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_ubuntu.sh | bash

# Google Colab (خليّة واحدة بعد اختيار GPU T4):
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_colab.sh | bash
```

```powershell
# Windows 10/11 (PowerShell عادي — أمر واحد):
irm https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_windows.ps1 | iex
```

> مقابض التحكم (متغيرات بيئة قبل الأمر): `IRAQI_YODAS_HOURS=200|all|0`، `IRAQI_STEPS=3000`، 
> `IRAQI_SKIP_TRAIN=1` (تجهيز بيانات فقط)، `IRAQI_SKIP_DEPS=1` (إعادة استخدام بيئة جاهزة)، 
> `IRAQI_DATA_ROOT=...`، `HF_ENDPOINT=...` (مرايا). كل المراحل قابلة للاستئناف بإعادة التشغيل.

### ب. المسار اليدوي (داخل هذا المستودع)

```bash
# 1) جهّز البيانات (تحويل/تنظيف/تقطيع 1–30 ثانية + manifest)
uv run python finetune/prepare_dataset.py --input-dir ./raw_iraqi

# 2) التدريب — إما CLI مباشرة:
uv run python finetune/run_finetune.py --dataset-dir ./raw_iraqi/prepared \
    --run-name iraqi_v1 --max-steps 2000 --prepare

#    أو عبر الـ API (متابعة تقدم + سجلات + إلغاء):
curl -X POST http://host/api/v1/finetune/jobs -H "X-API-Key: ..." \
  -H "Content-Type: application/json" \
  -d '{"dataset_dir": "/srv/datasets/iraqi", "run_name": "iraqi_v1", "max_steps": 2000}'

# 3) قيّم ثم انشر:
uv run python evaluation/evaluate_conversion.py --source s.wav --reference r.wav --converted c.wav
uv run python evaluation/benchmark_stream.py --profile balanced
```

النشر بمتغيرَي البيئة `RAFIDAIN__ENGINE__LOCAL_CHECKPOINT` و `RAFIDAIN__ENGINE__LOCAL_CONFIG`. الدليل الاستراتيجي الكامل (حجم البيانات، الجرعة، التقييم، الأمانة): **[FINETUNING.md](FINETUNING.md)**.

## 7) الاختبارات والمراقبة

```bash
make test        # 42 اختباراً — بدون GPU (المحرك الوهمي)
make lint        # ruff
make bench       # قياس RTF/زمن الكتلة على جهازك الفعلي
```

- `GET /healthz` — حيوية، `GET /readyz` — جاهزية (النموذج محمّل + التخزين قابلة للكتابة)
- `GET /metrics` — Prometheus: عدد الجلسات النشطة، زمن معالجة الكتلة (p50/p95 ضمنياً)، RTF لكل جلسة، تجاوزات الضغط الخلفي
- السجلات: أسطر JSON تحمل `request_id` و`session_id`

## 8) هيكل المشروع

```
rafidain/                        # جذر المستودع (github.com/Alqudimi/rafidain)
├── rafidain/                     # حزمة الخدمة
│   ├── core/                     # config, logging, errors
│   ├── domain/                   # كيانات + منافذ (Ports) — بلا أي I/O
│   ├── application/              # SessionManager, StreamPipeline, VoiceService, Finetune
│   ├── infrastructure/           # محرك Seed-VC، المحرك الوهمي، صوتيات، تخزين، طوابير
│   ├── api/                      # REST + WebSocket + مصادقة + مقاييس
│   ├── workers/                  # عامل طابور Redis للضبط الدقيق
│   └── main.py                   # مصنع التطبيق + Lifespan
├── finetune/                     # تجهيز بيانات عراقية + غلاف التدريب + إعدادات
├── evaluation/                   # تقييم الجودة + قياس الأداء اللحظي
├── scripts/                      # تنزيل النماذج + عميل دخان + مثبّتات البيئات الثلاث
├── tests/                        # وحدات + تكامل (42)
├── docker/                       # Dockerfile (base/full) + docker-compose
├── .github/workflows/ci.yml      # lint + tests + smoke
├── seed_vc_iraqi_colab.sh        # تنزيل بيانات اللهجة العراقية + التدريب (Colab)
├── seed_vc_iraqi_ubuntu.sh       # تنزيل بيانات اللهجة العراقية + التدريب (Ubuntu)
└── seed_vc_iraqi_windows.ps1     # تنزيل بيانات اللهجة العراقية + التدريب (Windows)
```

## 9) الأمان والأخلاقيات ⚠️

استنساخ صوت شخص حقيقي يتطلب **موافقته الصريحة**. عند استخدام النظام في مكالمات حية:

1. لا تستخدم صوت أي شخص دون إذن موثّق منه.
2. افصح للطرف الآخر في المكالمة عند طلب القانون أو الأخلاق المهنية (بعض الأنظمة تلزم بذلك).
3. فعّل `RAFIDAIN__API_KEYS`، وشغّل خلف TLS، وقيّد CORS في الإنتاج.

## 10) خارطة الطريق

- [ ] بوابة SIP/WebRTC (FreeSWITCH media streams / Twilio) فوق نفس طبقة التطبيق
- [ ] عتبة VAD بـ Silero (خفيفة عبر ONNX) بدل webrtcvad
- [ ] استنساخ Streaming للفوكودر HiFT لخفض زمن الكتلة
- [ ] Opus على السلك (تنسيقات تفاوض في `session.ready`)
- [ ] لوحة مراقبة Grafana جاهزة فوق مقاييس Prometheus

## 11) المراجع

- مستودع Seed-VC: <https://github.com/Plachtta/seed-vc> (الرياضيات المنقولة في المحرك مأخوذة من `real-time-gui.py` و`inference.py` الرسميين)
- نقاط seed-uvit-tat-xlsr-tiny على HuggingFace: <https://huggingface.co/Plachta/Seed-VC>
- CAMPPlus: <https://huggingface.co/funasr/campplus> — HiFT: <https://huggingface.co/FunAudioLLM/CosyVoice-300M>
- XLSR: `facebook/wav2vec2-xls-r-300m` عبر transformers

</div>
