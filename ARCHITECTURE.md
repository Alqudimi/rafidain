<div dir="rtl">

# معمارية رافدَين — الدليل المرجعي

هذا المستند هو المرجع المعماري: **لماذا** بُني كل شيء بهذه الطريقة، وكيف تُوسّعه دون كسر شيء. للقراءة الأولى ابدأ بـ [README](README.md).

---

## 1) النمط المعماري: Clean Architecture + Ports & Adapters

```
                        ┌────────────────────────────┐
   HTTP / WebSocket ───▶│  api (الواجهات/التقديم)     │
                        │  http_*.py  ws/stream.py    │
                        └───────────┬────────────────┘
                                    ▼
                        ┌────────────────────────────┐
   حالات الاستخدام ─────▶│  application (التطبيق)      │
                        │  SessionManager             │
                        │  StreamPipeline             │
                        │  VoiceService / Finetune    │
                        └───────────┬────────────────┘
                                    ▼
                        ┌────────────────────────────┐
   كيانات + عقود ──────▶│  domain (النطاق — نقي)      │
                        │  entities, value_objects    │
                        │  ports (بروتوكولات)          │
                        └───────────▲────────────────┘
                                    │ تنفيذ الـ Ports فقط
                        ┌───────────┴────────────────┐
   العالم الخارجي ◀─────│  infrastructure (المحولات)  │
                        │  seedvc/  mock_engine       │
                        │  audio/  stores.py  jobs.py │
                        └────────────────────────────┘
```

**القاعدة الحديدية**: اتجاه الاستيراد نحو الداخل فقط.

| الطبقة | تعتمد على | ممنوع عليها |
|---|---|---|
| `domain` | لا شيء (Python قياسي + numpy) | أي framework، أي I/O |
| `application` | `domain` | استيراد FastAPI أو torch مباشرة |
| `infrastructure` | `domain` (تنفيذ منافذه) | أن يعرفه أحد من الداخل |
| `api` | `application` + `infrastructure` (جذر التركيب فقط) | منطق أعمال |

الربط الفعلي يحدث في مكان واحد: `rafidain/api/deps.py::build_components` — **DI يدوي بلا سحر**، يمكن تتبّع الرسم الكامل للكائنات بالعين.

## 2) مطابقة SOLID — ملفاً بملف

| المبدأ | التطبيق الملموس |
|---|---|
| **S** — مسؤولية واحدة | `seedvc/loader.py` يحمّل النماذج فقط؛ `seedvc/engine.py` يحوّل الصوت فقط؛ `session_manager.py` يدير دورة الحياة فقط؛ `stream_pipeline.py` ينسّق التدفق فقط؛ `ws/stream.py` ينقل فقط. |
| **O** — مفتوح/مغلق | محركات جديدة تُسجَّل في `engine_registry.register_engine()` دون تعديل أي مستدعٍ. برتفايلات جديدة = سطر واحد في `DEFAULT_PROFILES`. |
| **L** — لسباروف | `MockConversionEngine` و`SeedVCStreamingEngine` قابلان للتبديل التام تحت نفس عقد `ConversionEngine` — الاختبارات والـ CI دليل عملي (42 اختباراً تعمل بالمحرك الوهمي). |
| **I** — فصل الواجهات | `ports.py` يعرّف عقوداً صغيرة: `ConverterSession` (3 توابع)، `VoiceRepository`، `TaggedCache`، `JobQueue`، `Clock`. لا عقد عملاقة. |
| **D** — عكس الاعتماد | `StreamPipeline` يستقبل تبعياته عبر `PipelineDependencies`؛ لا يعرف شيئاً عن torch أو seed-vc. `SessionManager` يعرف `Clock` كبروتوكول (سهولة تجميد الزمن في الاختبار). |

## 3) تدفق المكالمة اللحظية (تسلسل)

```
Client          api/ws/stream.py        application            infrastructure
  │  WSS connect   │                       │                       │
  │───────────────▶│ attach(session)       │                       │
  │                │ load_reference ───────▶ VoiceService ──────────▶ FileSystemVoiceRepository
  │                │ engine.new_session ──▶ (thread) ───────────────▶ SeedVCConverterSession
  │                │                       │   (شرط الصوت الهدف: prompt_condition, mel2, style2)
  │◀── session.ready ──│                  │                       │
  │  stream.start  │                       │                       │
  │───────────────▶│ transition(STREAMING) │                       │
  │◀── stream.started ─│                   │                       │
  │  PCM 20ms      │                       │                       │
  │───────────────▶│ pipeline.feed ────────▶ BlockAccumulator       │
  │                │                       │  ├─ VAD (اختياري)      │
  │                │                       │  └─ to_thread ─────────▶ convert_block (torch)
  │                │                       │     resample + encode   │
  │◀── PCM محوّل ──│◀── send_queue ◀───────│                       │
  │  metrics.poll  │                       │                       │
  │───────────────▶│                       │                       │
  │◀── metrics.snapshot (rtf, p95, overflows) │                    │
```

## 4) نموذج التزامن (Concurrency)

- **حلقة الأحداث (asyncio)** واحدة: قراءة WS، كتابة WS (مهمة `_writer` مستقلة)، إدارة الطوابير — بلا حظر أبداً.
- **كل استدعاء نموذج** يجري في `asyncio.to_thread` (torch متزامن): حلقة الأحداث تبقى حرة لبقية الجلسات.
- **`engine.max_concurrency`** (افتراضي 1): Semaphore يمنع تنافر GPU بين الجلسات — يرفَع إلى 2+ على بطاقات قوية بعد قياس `make bench`.
- **طابور إرسال محدود** لكل جلسة (`send_queue_size`): عند امتلائه تُسقط أقدم كتلة **صوتية** فقط (رسائل التحكم محفوظة) وتُقاس `overflows` — هذا هو الضغط الخلفي الحقيقي بدل انفجار الذاكرة.
- **الغلق النظيف**: `finally` في المعالج يفرغ الخط (`pipeline.finish`)، يلغي الكاتب، يغلق المحول، يحدّث آلة الحالة، ويحرر عدّاد الجلسات.

## 5) آلة حالات الجلسة

```
 CREATED ──▶ READY ──▶ STREAMING ⇄ PAUSED
    │          │  ▲        │    │
    │          │  └─stop───┘    │
    ▼          ▼               ▼
  CLOSED ◀──── CLOSED ◀──── CLOSED     (وERROR طرفية)
```

الانتقالات محمية في `Session.transition()` — أي حركة غير قانونية ترفع `InvalidSessionStateError` (409). «stop» يعيد الجلسة إلى READY لبدء مقطع اتصال جديد ضمن نفس الجلسة.

## 6) عزلة النموذج: المحول Adapter

`infrastructure/seedvc/loader.py` هو **المكان الوحيد** الذي يعرف تخطيط مستودع Seed-VC الرسمي:

- النقطة `DiT_uvit_tat_xlsr_ema.pth` + `config_dit_mel_seed_uvit_xlsr_tiny.yml` (Plachta/Seed-VC)
- مشفّر المحتوى XLSR عبر `transformers` (بتر الطبقات حسب `output_layer` في الإعداد)
- مشفّر المتحدث CAMPPlus (`funasr/campplus`)، فوكودر HiFT (`FunAudioLLM/CosyVoice-300M`)
- `model.cfm.estimator.setup_caches(1, 8192)` — مخابئ KV للبث (متطلب المنبع)

كل الاستيرادات الثقيلة **مؤجلة داخل التوابع** — باقي الخدمة (والاختبارات) تعمل بلا torch.

## 7) ميزانية زمن الوصول (Latency Budget)

لكل كتلة `block_ms` دخل، يدفع المستخدم:

| المكوّن | balanced | ملاحظة |
|---|---|---|
| حجم الكتلة (granularity) | 180ms | أدقة تحويل |
| crossfade محتجز | 40ms | دقة وصل الكتل |
| سياق أيمن lookahead | 20ms | يُحول قبل إصدار الكتلة |
| **خوارزمي جزئي** | **240ms** | `LatencyProfile.algorithmic_delay_ms` |
| استدلال الكتلة (GPU) | ~150–190ms | قياس رسمي للنموذج tiny (خطوات=10) |
| **الإجمالي النموذجي** | **~430ms** | مطابق للقياس المنشور في Seed-VC |
| شبكة + عميل | حسب النشر | خارض نطاق الخادم |

الرياضيات المنقولة في `SeedVCConverterSession`: كل الإطارات مقيّسة إلى وحدات 20ms (`zc = sr//50`)؛ نافذة انزلاقية `[سياق أيسر | crossfade | بحث | كتلة | سياق أيمن]`؛ قصّ `skip_head/skip_tail`؛ فرق إطارات CE↔DiT ثابت 2.0s؛ ثم crossfade متساوي القدرة على منطقة التراكب (`overlap = out_len − block`). **صون العينات**: كل كتلة دخل تُصدر صافي `block` من العينات — لا انزلاق ولا ازدواج.

## 8) التوسع (Scaling)

| المحور | الاستراتيجية |
|---|---|
| عمودي (GPU أقوى) | ارفع `engine.max_concurrency` بعد `make bench`؛ RTF<1 شرط أساسي لكل جلسة. |
| أفقي (عقد متعددة) | الجلسة مقيّدة بعقدة واحدة (ذاكرة محول محلية) → موازن حمل **sticky على session_id**. الأصوات على قرص مشترك/نسخ، الحالة المشتركة في Redis (`redis_url`). |
| فصل التدريب | طابور Redis + `python -m rafidain.workers.finetune_worker` على عقدة GPU منفصلة — التدريب لا ينافس الاستدلال. |
| استخراج خدمة النموذج | `ConversionEngine` منفذ عبر gRPC/triton دون تغيير `application` — هذا بالضبط ما يشتريه عزل المنافذ. |

## 9) سجلات قرارات معمارية (ADR)

- **ADR-001 — FastAPI + WebSocket أولاً**: بروتوكول واحد لوسائط النقل بلا وسيط، مع مستند OpenAPI مجاناً. SIP/WebRTC يضاف كبوابة فوق نفس طبقة التطبيق (خارطة الطريق).
- **ADR-002 — DI يدوي بدل حاوية**: رسم كائنات صريح في `build_components` — أسرع فهمًا، لا انعكاسات، كافٍ لحجم التبعيات.
- **ADR-003 — محول حول seed-vc بدل استنساخه**: نتبع المنبع (`Plachtaa/seed-vc`) ونثبّته في البناء؛ الرياضيات المستقرة (البث) أُعيدت هيكلتها لدينا لأنها نقطة التزامن الحرج. ملف واحد يحتضن تغيّرات المنبع.
- **ADR-004 — المحرك الوهمي من اليوم الأول**: جعل النظام قابلاً للاختبار بلا GPU (42 اختباراً + CI + عميل دخان) وأثبت مبدأ L بالفعل.
- **ADR-005 — ffmpeg كفاك تشفير**: تقليل سطح الهجوم وضخامة التبعيات؛ لا تحليل حاويات غير موثوقة في Python.
- **ADR-006 — تخزين ملفات بسيط (WAV+JSON)**: أصوات قابلة للتفتيش والنسخ الاحتياطي بأدوات النظام؛ الترقية إلى S3 تتطلب تنفيذ `VoiceRepository` فقط.
- **ADR-007 — سقوط أقدم الكتل عند الضغط**: في الزمن الحقيقي، الصوت القديم عديم القيمة؛ القياس (`overflows`) يجعل السياسة قابلة للمراقبة والتغيير.

## 10) الاختبارات كضمانة معمارية

```
tests/unit/       → النطاق والمنطق (FSM، codecs، بروتوكول، SessionManager، StreamPipeline)
tests/integration/→ دورة كاملة عبر TestClient: REST + WebSocket حقيقي + مصادقة + أخطاء
scripts/smoke_ws_client.py → خادم حي كامل (يشغّله الـ CI أيضاً)
```

حاجز الانحدار: أي تعديل معماري يكسر العقد يفشل فوراً في `make test` — قبل أن يصل إلى GPU.

</div>
