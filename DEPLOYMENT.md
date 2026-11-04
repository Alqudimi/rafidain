<div dir="rtl">

# النشر التشغيلي — رافدَين

دليل الإنتاج: المتطلبات، Docker، المراقبة، التوسع، والتحصين.

---

## 1) متطلبات العتاد

| العنصر | الحد الأدنى | موصى به |
|---|---|---|
| GPU | T4 16GB | A10G 24GB (جلسات متزامنة أكثر) |
| VRAM فعلي للمحرك tiny | ~2.5–3GB (XLSR نصف دقة + DiT 25M + HiFT + CAMPPlus) | نفسه؛ الفائض للمزامنة |
| CPU/RAM | 4 vCPU / 8GB | 8 vCPU / 16GB |
| القرص | 5GB (الصور) + 2GB (النقاط) + البيانات | SSD |

CPU-only **للنموذج الحقيقي غير عملي للبث اللحظي** (استدلال الكتلة يتجاوز ميزانية المكالمة) — استخدم `mock` للتطوير فقط، أو جرّب بروفايل `quality` بلا توقعات زمنية.

## 2) Docker (الطريق الموصى به)

```bash
# نسخ الجذر ثم:
export RAFIDAIN__API_KEYS='["مفتاح-سرّي-طويل-عشوائي"]'   # JSON array — pydantic-settings requirement
docker compose -f docker/docker-compose.yml up -d --build

# التحقق:
curl -s http://localhost:8000/healthz   # {"status":"ok",...}
curl -s http://localhost:8000/readyz    # "ready": true بعد تحميل النموذج (دقيقتان أول مرة)
```

- الصورة `full` تتضمن مستودع Seed-VC + حزمة ML. ثبّت `SEED_VC_REF` (argument بناء) على commit محدد للإنتاجية.
- لخبز النقاط داخل الصورة (بدون تنزيل أول تشغيل): أزل تعليق `RUN uv run python scripts/download_models.py` في `docker/Dockerfile`.
- التخزين الدائم: حجما Docker `rafidain-data` (الأصوات) و`rafidain-checkpoints` (النقاط).

### عامل الضبط الدقيق (عقدة GPU ثانية)

```bash
export RAFIDAIN_DATASETS_DIR=/data/datasets RAFIDAIN_RUNS_DIR=/data/runs
docker compose -f docker/docker-compose.yml --profile finetune up -d
```

التدريب يتنافس مع الاستدلال على الذاكرة — افصلهما إلا عند الضرورة.

## 3) التشغيل المباشر (بدون Docker)

```bash
make seedvc && make install ENGINE=seedvc
make download-models
RAFIDAIN__ENGINE__DEVICE=cuda \
RAFIDAIN__API_KEYS='["..."]' \
uvicorn rafidain.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> **عامل uvicorn واحد**: النموذج يُحمَّل مرة واحدة لكل عامل؛ العمال المتعددون يضاعفون VRAM بلا حاجة (المزامنة داخلية عبر `max_concurrency`). للتوسع الأفقي أضف عقداً خلف موازن حمل sticky.

## 4) المراقبة والتنبيهات

| ماذا | أين | تنبيه مقترح |
|---|---|---|
| الجاهزية | `GET /readyz` | `ready != true` لأكثر من دقيقة |
| RTF لكل جلسة | `rafidain_block_rtf` | RTF > 0.8 مستمر (شارف فقدان الزمن الحقيقي) |
| زمن الكتلة p95 | `rafidain_block_processing_seconds` | p95 > ميزانية البروفايل |
| تجاوزات الضغط | `rafidain_input_overflows_total` | أي نمو متصاعد (عميل بطيء/شبكة) |
| الجلسات النشطة | `rafidain_ws_sessions_active` | قرب `max_sessions` |
| أخطاء HTTP | `rafidain_http_requests_total` | 5xx > 1% |

السجلات أسطر JSON تحمل `request_id`/`session_id` — جاهزة لـ Loki/ELK. مستند OpenAPI على `/docs`.

## 5) التحصين (Production Hardening)

1. **المصادقة**: `RAFIDAIN__API_KEYS` إلزامية — مفاتيح طويلة عشوائية لكل عميل، وتدوير دوري.
2. **TLS**: أنهِ الاتصال عند reverse proxy (nginx/caddy/Traefik) — `ws://` داخلياً، `wss://` خارجياً. مضاعفة مهلة القراءة للبث اللحظي (مثال nginx: `proxy_read_timeout 3600s;`).
3. **CORS**: `RAFIDAIN__CORS_ORIGINS=["https://app.example.com"]` (لا `"*"` في الإنتاج).
4. **الشبكة**: عزل Redis عن الإنترنت (شبكة Docker داخلية)، ومنفذ 8000 خلف الـ proxy فقط.
5. **البيانات**: انسخ حجمة `rafidain-data` دورياً؛ الأصوات أصول حساسة — خزّنها مشفرة على القرص إن أمكن وقيّد الوصول.
6. **الحدود**: اضبط `max_sessions` و`send_queue_size` على سعة عتادك (ابدأ بقيم الإعداد الافتراضية واضبط بقياس `make bench`).

## 6) التحديث والتراجع

- صور Docker بعلامات إصدار؛ حدّث عقدة واحدة أولاً (blue/green) وقارن `readyz` والمقاييس قبل الدفع الكامل.
- نقاط النموذج المضبوطة تُدار كأصول مستقلة عن كود التطبيق (`LOCAL_CHECKPOINT` path) — التراجع للنموذج لا يتطلب تراجع الكود.

## 7) استمرارية الخدمة

- المكالمة اللحظية حساسة للانقطاع: أعد تشغيل العقد بإزاحة (rolling) وتجنّب إسقاط عقدة بها جلسات نشطة قدر الإمكان.
- إن فقد العميل الاتصال: تنتهي الجلسة (TTL) ويبدأ بجلسة جديدة — واجهة العميل يجب تعيد إنشاء الجلسة والصوت تلقائياً (`POST /sessions` + reconnect WS).

</div>
