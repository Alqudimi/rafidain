<div dir="rtl">

# دليل المثبّتات الجاهزة — رافدَين (Rafidain Install Scripts)

**ثلاثة سكربتات مكتملة تثبّت وتهيّئ وتشغّل نظام رافدَين بالكامل** (المشروع + مستودع Seed-VC + نموذج `seed-uvit-tat-xlsr-tiny` + نقاطه + خادم الـ API) في بيئة واحدة لكل من:

| السكربت | البيئة | النمط |
|---|---|---|
| `scripts/install/setup_colab.sh` | Google Colab (T4 GPU) | curl-bash / خلية واحدة |
| `scripts/install/setup_ubuntu.sh` | Ubuntu 20.04+ / Debian 12+ | curl-bash / تشغيل مباشر |
| `scripts/install/setup_windows.ps1` | Windows 10/11 (PowerShell 5.1+) | ملف مباشر |

كلها **مجربة فعلياً end-to-end**: التثبيت الكامل، الاختبارات الـ 42، رفع الخادم، بث WebSocket حقيقي عبر عميل الدخان، وحتى تحميل نموذج Seed-VC الحقيقي وإجراء تحويل صوتي فعلي بنفس الـ Pin المثبّت.

---

## 0) 🆕 المثبّت الموحّد الجديد (uv) — الطريق الأساسي

**سكربت واحد لكل البيئات**: `scripts/install/setup_uv.sh` — Linux / macOS / WSL /
Google Colab (ولويندوز: نفس خطوات `make` عبر `scripts/manage.py`).

- يكتشف النظام والـ GPU وإصدار CUDA **تلقائياً** (عبر `scripts/manage.py detect`).
- يثبّت **uv** إن لم يكن موجوداً، ثم يزامن البيئة **المقفولة** من `uv.lock`
  (يثبّت القفل كل متغيرات torch: cpu / cu118 / cu124 / cu126 — فلا تعطّل بين
  بيئة وأخرى أبداً).
- يستنسخ Seed-VC، ينزّل النقاط، يكتب `.env` بمفتاح API عشوائي، يشغّل الاختبارات،
  يرفع الخادم مع **لوحة التحكم `/web` وصفحة البث الحي `/live`**، ويفتح **نفقاً
  عاماً مجانياً** `trycloudflare.com` (أو نفقك المسمّى عبر `CLOUDFLARED_TOKEN`).

```bash
# من المستودع:
bash scripts/install/setup_uv.sh

# curl-bash (بعد الاستضافة على GitHub):
curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_uv.sh | bash

# أبرز الخيارات:
#   --engine mock|seedvc   --variant cpu|cu118|cu124|cu126
#   --api-key KEY          --no-auth  --no-models  --no-tests
#   --no-tunnel            --no-serve (تثبيت فقط)   --port N
```

بعد التثبيت، كل دورة الحياة عبر `make` (أو `scripts/manage.py` على ويندوز):
`make web` · `make web-tunnel` · `make train ARGS="…"` · `make status` · `make stop`.

> مثبّتات pip الثلاثة القديمة (أدناه) ما زالت تعمل كما هي للتوافق الرجعي، لكن
> مسار uv أعلاه هو **الموصى به والموثّق** — أسرع، مقفول، وعابر للأنظمة.

---

## 1) ماذا تفعل السكربتات بالضبط؟ (10 خطوات)

1. **فحص البيئة**: نظام التشغيل، Python (يجب ≥ 3.10)، GPU وإصدار تعريف CUDA (عبر `nvidia-smi`).
2. **اختيار المحرك تلقائياً**: يوجد GPU → المحرك الحقيقي `seedvc` على CUDA · لا يوجد GPU → المحرك الوهمي `mock` (يعمل فوراً بدون أي مكتبات تعلم آلة — للتجربة والتطوير).
3. **الحزم النظامية**: `ffmpeg` (إلزامي لرفع الأصوات) و`git` وPython — تثبَّت تلقائياً عند الحاجة (apt على لينكس، تنزيل مباشر/winget على ويندوز).
4. **جلب المشروع**: من مجلد محلي أو Git URL أو Tarball/Zip URL (خيار `--source`)، أو بالاكتشاف التلقائي إن كان السكربت داخل المشروع.
5. **بيئة Python معزولة**: على Ubuntu/Windows تُنشأ `.venv` خاصة بالمشروع؛ على Colab يُستخدم Python النظامي (مع الاحتفاظ بـ torch المثبّت مسبقاً إن كان ≥ 2.6 وبـ CUDA).
6. **المكدس البرمجي الموثّق**:
   - الأساس: FastAPI / uvicorn / pydantic / soxr / prometheus-client … (من `requirements.txt`).
   - عند المحرك الحقيقي فقط: **`torch==2.6.0` + `torchaudio==2.6.0`** من الفهرس المطابق لكرتك (cu124 / cu118 / CPU) + `transformers>=4.40,<5` + `librosa` + `munch` + `einops` + `descript-audio-codec` …
7. **استنساخ مستودع Seed-VC** إلى `third_party/seed-vc` (عبر git أو تنزيل zip على ويندوز بدون git).
8. **تنزيل النقاط مسبقاً** (~1.5 GB): DiT tiny (142MB) + CAMPPlus (28MB) + HiFT (82MB) + مشفّر المحتوى XLSR (1.27GB).
9. **توليد `.env` كامل** مع مفتاح API عشوائي (أو مفتاحك عبر `--api-key`) + ملف `run_server.sh`/`run_server.ps1`، **ثم تشغيل الاختبارات الـ 42** (بالمحرك الوهمي — تتحقق من كل شيء بلا GPU).
10. **رفع الخادم** بالخلفية، انتظار `/healthz` ثم `/readyz` (تحميل النموذج + الإحماء)، ثم طباعة ملخص فيه الرابط، المفتاح، وأمثلة التجربة الجاهزة للنسخ. على Colab يُفتح أيضاً **نفق عام مجاني `trycloudflare.com`**.

السكربتات **متكررة الآثار (idempotent)**: إعادة التشغيل توقف الخادم القديم، تعيد استخدام النقاط والـ venv، وتحتفظ بنسخة احتياطية من `.env` القديم.

---

## 2) Google Colab

### أ. الطريقة المحلية (الأسرع للتجربة)

```python
# 1) ارفع zip المستودع (rafidain-main.zip) إلى Colab ثم:
!unzip -q rafidain-main.zip
# 2) من قائمة Runtime ▸ Change runtime type ▸ اختر GPU (T4)
# 3) شغّل:
!bash rafidain-main/scripts/install/setup_colab.sh
```

### ب. عبر URL (بعد رفع المشروع إلى GitHub — نمط curl-bash)

```python
# الخلية الأولى (المستودع الرسمي — يستنسخه السكربت تلقائياً إن لم يجده):
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_colab.sh | bash
# أو مع تحديد المصدر صراحةً:
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_colab.sh | bash -s -- --source https://github.com/Alqudimi/rafidain
```

> عند الاستضافة، يستطيع السكربت استنساخ المشروع تلقائياً؛ كما يدعم `--source <رابط tar.gz>`.

### ج. الخيارات المتاحة

```python
!bash setup_colab.sh --engine mock --no-tunnel --no-tests --api-key MY-KEY --port 8080
```

| الخيار | القيمة | الوصف |
|---|---|---|
| `--engine` | `seedvc` \| `mock` | افتراضياً `seedvc` مع GPU و`mock` بدونه |
| `--device` | `auto` \| `cuda` \| `cpu` | افتراضياً `auto` |
| `--port` | رقم | الافتراضي 8000 |
| `--api-key` | نص | مفتاح ثابت بدلاً من العشوائي |
| `--no-auth` | — | تعطيل المصادقة (**للتطوير فقط**) |
| `--no-models` | — | تخطي تنزيل النقاط (ستُنزَّل عند أول تشغيل) |
| `--no-tests` | — | تخطي الاختبارات |
| `--no-tunnel` | — | بدون النفق العام |
| `--foreground` | — | تشغيل الخادم بالمقدمة (Ctrl+C للإيقاف) |
| `--source` | مسار/URL | مصدر المشروع |
| `--ref` | اسم فرع | فرع Git (افتراضياً `main`) |

> **ملاحظات Colab**: الآلة تُعاد تدويرها دورياً — أعد تشغيل الخلية لإرجاع كل شيء (النقاط تبقى طالما لم تُعاد تهيئة المشغل). رابط `trycloudflare.com` يغيّر مع كل تشغيل.

---

## 3) Ubuntu / Debian

### أ. محلي (المشروع مستخرج بجانبك)

```bash
tar -xzf rafidain.tar.gz && cd rafidain
bash scripts/install/setup_ubuntu.sh                     # يتكفل بكل شيء
```

### ب. عبر curl-bash (بعد الاستضافة على GitHub)

```bash
curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_ubuntu.sh \
  | bash -s -- --source https://github.com/Alqudimi/rafidain
# أو بدونه — يستنسخ السكربت المستودع الرسمي تلقائياً:
curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_ubuntu.sh | bash
```

### ج. كخدمة نظام systemd (إنتاجي — يتطلب sudo)

```bash
sudo bash scripts/install/setup_ubuntu.sh --engine seedvc --service
# ثم:
systemctl status rafidain && journalctl -u rafidain -f
```

### د. نفق عام مؤقت (اختياري — لخادم بلا IP عام)

```bash
bash scripts/install/setup_ubuntu.sh --tunnel
```

**خيارات إضافية خاصة بأوبونتو**: `--service` (خدمة systemd)، `--tunnel`، `--skip-apt` (لبيئات مقيدة بلا apt)، إضافة إلى نفس خيارات Colab أعلاه.
يدعم السكربت **Ubuntu 20.04** تلقائياً (يثبّت Python 3.11 من PPA deadsnakes عند الحاجة).

### ما يُثبّت من حزم نظام

`ca-certificates` · `curl` · `git` · `ffmpeg` · `python3` · `python3-venv` — فقط ما ينقص منها فعلياً.

---

## 4) Windows 10/11

```powershell
# من PowerShell عادي (لا حاجة لصلاحيات مدير) بعد استخراج المشروع:
cd rafidain
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install\setup_windows.ps1
```

- يكتشف Python عبر مشغّل `py` أو PATH أو `%LOCALAPPDATA%` — وإن لم يجده ثبّته عبر **winget** تلقائياً.
- يثبّت **ffmpeg** بتنزيل مباشر (نسخة gyan.dev الرسمية) إلى `tools\ffmpeg` ويضيفه لمسارك.
- يستنسخ Seed-VC عبر git أو **كـ zip بدون git**.
- NVIDIA: يقرأ `nvidia-smi` ويختار عجلة CUDA المناسبة (cu124/cu118)، وإلا فعجلات CPU.
- الخيارات نفسها بصيغة PowerShell: `-Engine mock -NoTests -ApiKey KEY -Port 8080 -Foreground -Tunnel -NoModels -NoAuth -Source <dir|url>`
- للوصول من الشبكة المحلية يجب فتح المنفذ (مرة واحدة بصلاحيات مدير):

```powershell
netsh advfirewall firewall add rule name="Rafidain 8000" dir=in action=allow protocol=TCP localport=8000
```

---

## 5) استضافة السكربتات بنمط curl-bash

1. المشروع مستضاف أصلاً على المستودع الرسمي — ويمكنك استضافة نسختك المعدّلة إلى مستودع GitHub خاص بك (`git init && git add . && git commit -m "rafidain" && git remote add origin <url> && git push`).
2. روابط السكربتات الرسمية الجاهزة:

```
https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_colab.sh
https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_ubuntu.sh
```

3. ولنسخة PowerShell على ويندوز (النمط المكافئ لـ curl-bash):

```powershell
irm https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_windows.ps1 -OutFile setup.ps1
powershell -ExecutionPolicy Bypass -File setup.ps1          # يستنسخ المستودع الرسمي تلقائياً
# أو مع تحديد المصدر:
powershell -ExecutionPolicy Bypass -File setup.ps1 -Source https://github.com/Alqudimi/rafidain
```

> يمكن أيضاً استضافة `rafidain.tar.gz` على أي رابط مباشر واستخدام `--source <رابط>`.
> وعند التشغيل بنمط curl-bash بدون مشروع محلي، يستنسخ السكربت المستودع الرسمي `Alqudimi/rafidain` تلقائياً.

---

## 5-ب) سكربتات البيانات والتدريب (أمر واحد من المستودع نفسه)

سكربتات الضبط الدقيق مستقلة تماماً عن المستودع (تبني بيئتها الخاصة وتنزّل Seed-VC والبيانات بنفسها)،
وتحمل **أشرطة تقدم حية** لكل المراحل: تنزيل البيانات (نسبة + غيغابايت متدفقة + سرعة + ETA)،
التجهيز (مقاطع/ثانية)، وتنزيل نقاط التدريب — وشريط **خطوة-بخطوة للتدريب نفسه** مع loss وepoch وETA.

```bash
# Ubuntu 20.04/22.04/24.04 (NVIDIA GPU للتدريب؛ بدون GPU: بيئة + بيانات فقط):
curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_ubuntu.sh | bash
```

```python
# Google Colab (خليّة واحدة بعد اختيار Runtime ▸ GPU T4):
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_colab.sh | bash
```

```powershell
# Windows 10/11 (PowerShell عادي — أمر واحد):
irm https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_windows.ps1 | iex
```

مقابض التحكم (متغيرات بيئة قبل الأمر، نفسها للثلاث بيئات):

| المتغير | الافتراضي | الوظيفة |
|---|---|---|
| `IRAQI_YODAS_HOURS` | `200` | ساعات اللهجة العراقية من كتالوج YODAS-IQ (`all` = 339h، `0` = تعطيل) |
| `IRAQI_STEPS` | `3000` | خطوات التدريب |
| `IRAQI_SKIP_SMALL=1` | — | تخطي المجموعات الصغيرة النظيفة (youssef/shiekh/safaa) |
| `IRAQI_SKIP_TRAIN=1` | — | تجهيز البيانات فقط دون تدريب |
| `IRAQI_SKIP_DEPS=1` | — | إعادة استخدام بيئة Python جاهزة (تخطي venv/pip) |
| `IRAQI_DATA_ROOT` | `~/iraqi_seedvc` | جذر البيانات والأوزان (`/content/iraqi_seedvc` على Colab) |

كل المراحل قابلة للاستئناف: إعادة تشغيل السكربت تكمل من حيث توقف (علامات إنجاز + تخطّي الموجود).

---

## 6) لماذا `torch==2.6.0` تحديداً؟ (قرار هندسي موثّق)

- مشفّر المحتوى **XLSR** (`facebook/wav2vec2-xls-r-300m`) متوفر على HuggingFace بصيغة `pytorch_model.bin` فقط (بدون safetensors)، وإصدارات transformers الحديثة (≥ 4.53) **ترفض** `torch.load` على ملفات `.bin` مع torch أقدم من 2.6 بسبب CVE-2025-32434 — لذا أي Pin أقدم سيكسر التحميل.
- ملفات `.pth` الخاصة بـ Seed-VC تُحمَّل بالمسار الكلاسيكي (pickle)، وtorch ≥ 2.6 غيّر الافتراضي إلى `weights_only=True` — لهذا **تصدّر السكربتات متغير `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`** عند تشغيل الخادم (مضبوط داخل `run_server.*` وفي systemd).
- `2.6.0` يدعم Python 3.9–3.13، وعجلاته: `cu118` / `cu124` / `cu126` — لذا يختار السكربت `cu124` لتعريفات ≥ CUDA 12.4 و`cu118` لتعريفات 11.8–12.3.
- على Colab: إن كان المثبّت مسبقاً torch ≥ 2.6 وشغّالاً بـ CUDA يُترك كما هو (توفير ~2.5GB تنزيل).

**على GPU حقيقية (T4)**: بروفايل `balanced` يحقق زمن كتلة ~430ms (قياس منشور رسمياً للنموذج نفسه) — مناسب للمكالمات. على CPU يعمل المحرك لكن بزمن يتجاوز ميزانية الوقت الحقيقي (RTF ≈ 6 في اختبارنا) — استخدمه للتجربة فقط.

---

## 7) استكشاف الأخطاء

| المشكلة | الحل |
|---|---|
| `no usable GPU` على Colab | Runtime ▸ Change runtime type ▸ T4 GPU ثم أعد تشغيل الخلية |
| الخادم لا يرتفع (`server did not come up`) | راجع آخر السجلات: `tail -30 server.log` (وعلى ويندوز `server.err.log`) — غالباً منفذ مشغول: جرّب `--port 8080` |
| `/readyz` لا يصبح true | أول تشغيل ينزّل XLSR (~1.3GB) — انتظر أو راقب `tail -f server.log` |
| `estimator.input_pos` shape mismatch عند التحميل | **تحذير طبيعي متوقع** من مستودع Seed-VC نفسه (يُعاد بناء تضمين المواضع) — ليس خطأً |
| بطء تنزيل النقاط | ضبط `HF_TOKEN` يرفع حد المعدل على HuggingFace |
| ويندوز: أمر `powershell` مرفوض | شغّله بالصيغة المذكورة حرفياً مع `-ExecutionPolicy Bypass` |
| ويندوز: torch لا يجد كيرنلات CUDA بعد تثبيت | أعد التشغيل بعد تحديث تعريف NVIDIA؛ كروت RTX 50xx الجديدة تحتاج torch ≥ 2.7 بعجلة cu128: `pip install torch==2.7.x torchaudio==2.7.x --index-url https://download.pytorch.org/whl/cu128` |
| رسالة `weights_only` تحذيرية في السجلات | طبيعية تماماً — تدل أن متغير الأمان القديم يعمل كما صُمم |

---

## 8) ماذا بعد التشغيل؟

```bash
# صحة الخادم:
curl -s http://127.0.0.1:8000/healthz && curl -s http://127.0.0.1:8000/readyz

# تجربة بث كاملة (يرفع صوتاً اصطناعياً ويحوّله عبر WebSocket):
python scripts/smoke_ws_client.py --url http://127.0.0.1:8000 --seconds 3 --api-key <KEY>

# رفع صوت هدف حقيقي (5–30 ثانية صوت نظيف):
curl -X POST http://127.0.0.1:8000/api/v1/voices -H "X-API-Key: <KEY>" \
  -F "file=@reference.wav" -F "name=الصوت-الهدف"

# ثم أنشئ جلسة واربط WebSocket — التفاصيل في README القسم 4
```

- **ضبط اللهجة العراقية** (بياناتك الخاصة): `python finetune/run_finetune.py --help` ثم الدليل الكامل في `FINETUNING.md`.
- **النشر الإنتاجي الحقيقي**: `DEPLOYMENT.md` (TLS، systemd، المراقبة، التحصين).

## 9) الأمان ⚠

- المفتاح المولَّد يُطبع مرة واحدة في نهاية السكربت — احفظه. المصادقة إلزامية إلا مع `--no-auth`.
- نفق `trycloudflare.com` عام: من يملك الرابط يستطيع الوصول للـ API (المصادقة تحمي العمليات — أبقِ المفتاح سرياً).
- **استنساخ صوت أي شخص يتطلب موافقته الصريحة** — راجع قسم الأخلاقيات في README.

</div>
