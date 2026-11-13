<div dir="rtl">

# رافِدَين | Rafidain

**منصة مفتوحة المصدر لتحويل الصوت إلى صوت مستهدف وبثه لحظياً، مع أدوات لضبط نموذج Seed-VC على اللهجة العراقية.** تتضمن خدمة FastAPI، واجهة ويب، REST وWebSocket، دفاتر تدريب لـ Google Colab وRunPod، وسكربتات تثبيت لأنظمة Linux وWindows.

[المستودع](https://github.com/Alqudimi/rafidain) · [الدفاتر](https://github.com/Alqudimi/rafidain/tree/main/notbooks) · [إصدارات المشروع](https://github.com/Alqudimi/rafidain/releases) · [المشكلات](https://github.com/Alqudimi/rafidain/issues) · [المناقشات](https://github.com/Alqudimi/rafidain/discussions)

> **تنبيه:** تحويل الصوت أو محاكاة صوت شخص حقيقي يتطلب موافقته الصريحة. لا تستخدم الخدمة للتضليل أو انتحال الهوية، واحترم تراخيص النماذج ومجموعات البيانات.

## المحتويات

- [ما هو المشروع؟](#ما-هو-المشروع)
- [الميزات](#الميزات)
- [دفاتر التدريب: Colab وRunPod والمنصات الأخرى](#دفاتر-التدريب-colab-وrunpod-والمنصات-الأخرى)
- [اختيار طريقة التشغيل](#اختيار-طريقة-التشغيل)
- [التثبيت والتشغيل](#التثبيت-والتشغيل)
- [التشغيل السريع بالمحرك الوهمي](#التشغيل-السريع-بالمحرك-الوهمي-دون-gpu)
- [تشغيل نموذج Seed-VC الحقيقي](#تشغيل-نموذج-seed-vc-الحقيقي)
- [المثبتات الجاهزة](#المثبتات-الجاهزة)
- [Docker](#docker)
- [واجهة الويب](#واجهة-الويب)
- [استخدام REST وWebSocket](#استخدام-rest-وwebsocket)
- [ضبط النموذج للهجة العراقية](#ضبط-النموذج-للهجة-العراقية)
- [الاختبارات وجودة الكود](#الاختبارات-وجودة-الكود)
- [الإعدادات والأمان](#الإعدادات-والأمان)
- [هيكل المستودع](#هيكل-المستودع)
- [استكشاف الأخطاء](#استكشاف-الأخطاء)
- [الوثائق والمراجع](#الوثائق-والمراجع)
- [المساهمة والترخيص](#المساهمة-والترخيص)

## ما هو المشروع؟

رافِدَين مشروع Python لتحويل الصوت (Voice Conversion) في الزمن الحقيقي، مبني حول نموذج [Seed-VC](https://github.com/Plachtaa/seed-vc) الصغير `seed-uvit-tat-xlsr-tiny`. يوفّر طبقة خدمة قابلة للتشغيل والتطوير والاختبار؛ كما يضم مساراً لتجهيز بيانات الكلام وضبط النموذج وتقييمه.

| العنصر | التفاصيل |
|---|---|
| لغة التشغيل | Python 3.10 أو أحدث؛ يوصى Python 3.11 |
| واجهة الخدمة | FastAPI / Uvicorn، توثيق OpenAPI على `/docs` |
| نقل الصوت | REST لدورة حياة الصوت والجلسات، وWebSocket للبث الحي |
| المحركات | `seedvc` للنموذج الحقيقي، و`mock` للتطوير والاختبارات دون GPU |
| النموذج الأساسي | `Plachta/Seed-VC`، ملف tiny XLSR؛ تحقّق من متطلبات الأوزان والتراخيص لدى المصدر |
| أدوات التدريب | تجهيز البيانات، الضبط الدقيق، التقييم، ودفاتر Colab/RunPod |
| المراقبة | نقاط صحة وجاهزية ومقاييس Prometheus وسجلات منظمة |

زمن الاستجابة الفعلي يعتمد على بطاقة GPU وإعدادات النموذج والشبكة والملف الصوتي؛ لا يُعدّ أي رقم مرجعياً لكل جهاز. نفّذ القياس على عتادك باستخدام `make bench`.

## الميزات

- بث صوتي لحظي عبر WebSocket مع ملفات زمن وصول قابلة للاختيار.
- واجهة API لإدارة الأصوات والجلسات، وواجهة ويب مضمّنة للتجربة.
- محرك `mock` خفيف لتشغيل الخدمة والاختبارات دون تنزيل الأوزان أو وجود GPU.
- تكامل اختياري مع Seed-VC ونقاطه على Hugging Face، مع إمكانية تحديد نقطة محلية مضبوطة.
- مسار تدريب وإعداد بيانات وتقييم، مع دعم نسخ checkpoints إلى Google Drive في أدوات التدريب المدعومة.
- نشر باستخدام Docker Compose، ومقاييس Prometheus وسجلات JSON.
- سكربتات تثبيت لـ Google Colab وUbuntu/Debian وWindows.

## دفاتر التدريب: Colab وRunPod والمنصات الأخرى

توجد الدفاتر داخل [مجلد `notbooks/`](https://github.com/Alqudimi/rafidain/tree/main/notbooks) (الاسم مكتوب هكذا في المستودع). استخدم **فتح في Colab** للتنفيذ السحابي، أو **GitHub** للعرض، أو **nbviewer** لعرض محتوى الدفتر. روابط Colab تفتح الدفتر مباشرة؛ قد تحتاج إلى تسجيل الدخول واختيار Runtime مزود بـ GPU. تختلف الإصدارات في البيانات والإعدادات، لذا اقرأ أول خلايا الدفتر قبل التشغيل.

| الدفتر | الاستخدام المقصود | Colab | GitHub | nbviewer |
|---|---|---|---|---|
| `rafidain_colab_training.ipynb` | دفتر Colab متكامل لتجهيز البيانات والتدريب والتصدير | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/rafidain_colab_training.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/rafidain_colab_training.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/rafidain_colab_training.ipynb) |
| `colab_training.ipynb` | دفتر Colab للتدريب؛ يتضمن دليلاً خاصاً به | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/colab_training.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/colab_training.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/colab_training.ipynb) |
| `rafidain_runpod_rtx5090_training.ipynb` | RunPod مع RTX 5090؛ راجع متطلبات إصدار PyTorch/CUDA في الدفتر | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/rafidain_runpod_rtx5090_training.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/rafidain_runpod_rtx5090_training.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/rafidain_runpod_rtx5090_training.ipynb) |
| `rafidain_runpod_training_v2.ipynb` | RunPod؛ يتضمن تقييم A/B وقياساً للتدفق وفق وصف الدفتر | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/rafidain_runpod_training_v2.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/rafidain_runpod_training_v2.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/rafidain_runpod_training_v2.ipynb) |
| `runpod_rtx5090_autopilot_training.ipynb` | تجارب آلية على RunPod RTX 5090؛ يوضح الدفتر نمط التشغيل والميزانية | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/runpod_rtx5090_autopilot_training.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/runpod_rtx5090_autopilot_training.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/runpod_rtx5090_autopilot_training.ipynb) |
| `runpod_rtx5090x2_parallel_training.ipynb` | تشغيل تجارب مستقلة بالتوازي على بطاقتي RTX 5090 | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/runpod_rtx5090x2_parallel_training.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/runpod_rtx5090x2_parallel_training.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/runpod_rtx5090x2_parallel_training.ipynb) |
| `runpod_training_V1.ipynb` | إصدار RunPod أولي لتدريب النموذج وتصديره | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V1.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V1.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V1.ipynb) |
| `runpod_training_V2.ipynb` | إصدار RunPod ثانٍ؛ راجع خلايا الإعداد والبيانات قبل التنفيذ | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V2.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V2.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V2.ipynb) |
| `runpod_training_V3.ipynb` | إصدار RunPod يتضمن تقييماً ومقارنة بحسب وصف الدفتر | [فتح](https://colab.research.google.com/github/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V3.ipynb) | [عرض](https://github.com/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V3.ipynb) | [عرض](https://nbviewer.org/github/Alqudimi/rafidain/blob/main/notbooks/runpod_training_V3.ipynb) |

### فتح الدفاتر في منصات أخرى

- **RunPod:** أنشئ Pod مناسباً، ثم افتح JupyterLab وارفع ملف `.ipynb` من مجلد `notbooks/` أو استنسخ المستودع داخل البيئة. اختر الدفتر المطابق لنوع GPU لديك؛ دفاتر RTX 5090 مخصصة لبيئة Blackwell، ولا تفترض أن الإعدادات تناسب كل بطاقة.
- **JupyterLab أو Jupyter Notebook محلياً:** استنسخ المشروع، ثم افتح الدفتر من واجهة Jupyter. يتطلب التدريب تثبيت الاعتماديات والعتاد المناسبين كما يوضح الدفتر.
- **VS Code:** افتح ملف `.ipynb` من المستودع باستخدام إضافة Jupyter، واختر Kernel/بيئة Python مناسبة.
- **Kaggle Notebooks:** نزّل ملف الدفتر من GitHub وارفعه/استورده عبر واجهة Kaggle لإنشاء Notebook جديد. قد تحتاج إلى تعديل مسارات الملفات، صلاحيات الإنترنت، التخزين، وتهيئة GPU؛ لم تُختبر هذه البيئة تلقائياً في المشروع.
- **عرض دون تشغيل:** استخدم روابط GitHub أو nbviewer في الجدول.

> دفاتر Colab المخصصة لـ Colab قد تعتمد على Google Drive. لا تشغّل خلية غير مفهومة قبل مراجعتها، ولا تضع رموز Hugging Face أو مفاتيح API داخل دفتر عام أو تشاركه مع الآخرين.

## اختيار طريقة التشغيل

| هدفك | ابدأ من |
|---|---|
| تجربة API دون GPU | [المحرك الوهمي](#التشغيل-السريع-بالمحرك-الوهمي-دون-gpu) |
| تشغيل تحويل صوت حقيقي على GPU | [Seed-VC](#تشغيل-نموذج-seed-vc-الحقيقي) أو [المثبتات الجاهزة](#المثبتات-الجاهزة) |
| واجهة ويب سريعة | `make web` أو `make web-mock` |
| تدريب نموذج عراقي | [دفاتر التدريب](#دفاتر-التدريب-colab-وrunpod-والمنصات-الأخرى) و[FINETUNING.md](FINETUNING.md) |
| تشغيل بالحاويات | [Docker](#docker) و[DEPLOYMENT.md](DEPLOYMENT.md) |

## التثبيت والتشغيل

### المتطلبات

- Python 3.10+؛ الموصى به Python 3.11.
- Git و[uv](https://docs.astral.sh/uv/) للتثبيت بالطريقة المعتمدة في المستودع.
- `ffmpeg` مطلوب لعمليات الصوت وبعض مسارات التجهيز. على Ubuntu/Debian: `sudo apt-get update && sudo apt-get install -y git curl ffmpeg`.
- للتحويل الفعلي: NVIDIA GPU وتعريف CUDA متوافقان مع نسخة PyTorch المختارة. قد تعمل بعض الأوضاع على CPU، لكنها ليست مناسبة عادةً للبث اللحظي. وضع `mock` لا يحتاج GPU.
- مساحة كافية لتنزيل مستودع Seed-VC والأوزان وبيانات التدريب. نزّل النماذج عند الحاجة فقط.

### استنساخ المستودع

```bash
git clone https://github.com/Alqudimi/rafidain.git
cd rafidain
```

### تثبيت uv

إن لم يكن `uv` مثبتاً، اتبع [دليل التثبيت الرسمي لـ uv](https://docs.astral.sh/uv/getting-started/installation/) أو استخدم مثبت البيئة الجاهز أدناه. بعد توفر `uv`:

```bash
uv --version
```

### تثبيت اعتماديات التطوير والاختبارات (دون PyTorch)

هذا الخيار مناسب لتطوير API وتشغيل الاختبارات بالمحرك الوهمي:

```bash
uv sync --frozen --group dev
# أو
make install-dev
```

### تثبيت بيئة المحرك

للتثبيت عبر أداة إدارة المشروع:

```bash
# اختيار تلقائي حسب البيئة
make install

# اختيار المحرك الحقيقي
make install ENGINE=seedvc

# تثبيت/تهيئة المحرك الوهمي
make install ENGINE=mock
```

بعد تثبيت محرك Seed-VC، جهّز المستودع والأوزان عند اللزوم:

```bash
make seedvc
make download-models
```

يمكن الاطلاع على الخيارات التفصيلية عبر:

```bash
uv run python scripts/manage.py --help
uv run python scripts/manage.py install --help
uv run python scripts/manage.py download --help
```

> المحرك الحقيقي يحتاج اعتماديات تعلم الآلة وPyTorch متوافقاً مع بيئتك. يفضّل استخدام `make install ENGINE=seedvc` أو المثبتات الرسمية للمشروع بدلاً من تثبيت `torch` عشوائياً، خصوصاً على Colab أو بطاقات RTX 50xx.

## التشغيل السريع بالمحرك الوهمي (دون GPU)

ابدأ خادم API محلياً:

```bash
make install-dev
make serve-mock
```

افتح في المتصفح:

- توثيق API: <http://127.0.0.1:8000/docs>
- الصحة: <http://127.0.0.1:8000/healthz>
- الجاهزية: <http://127.0.0.1:8000/readyz>
- المقاييس: <http://127.0.0.1:8000/metrics>

من طرفية ثانية اختبر مسار WebSocket:

```bash
uv run python scripts/smoke_ws_client.py --url http://127.0.0.1:8000 --seconds 3
```

أوقف الخادم بـ `Ctrl+C` إذا كان يعمل في الواجهة الأمامية.

## تشغيل نموذج Seed-VC الحقيقي

بعد الاستنساخ وتثبيت المحرك الحقيقي:

```bash
make seedvc
make install ENGINE=seedvc
make download-models
cp .env.example .env
```

حرّر `.env` لضبط الجهاز والمصادقة، أو عيّن متغيرات البيئة مباشرة:

```bash
export RAFIDAIN__ENGINE__IMPL=seedvc
export RAFIDAIN__ENGINE__DEVICE=cuda
export RAFIDAIN__API_KEYS='["ضع-مفتاحاً-سرياً-قوياً-هنا"]'
make serve
```

الخادم افتراضياً على `0.0.0.0:8000`. تحقق من التحميل عبر `/readyz`؛ أول تشغيل قد يحتاج وقتاً لتنزيل الأوزان وتحميلها. لمزيد من المتغيرات راجع [`.env.example`](.env.example) و[ملف الإعداد](rafidain/core/config.py).

## المثبتات الجاهزة

يوفر المستودع سكربتات إعداد للمبتدئين والبيئات السحابية. راجع [INSTALL-SCRIPTS.md](INSTALL-SCRIPTS.md) قبل استخدامها لمعرفة الخيارات وتأثيراتها.

### Google Colab

في دفتر Colab اختر Runtime مزوداً بـ GPU، ثم نفّذ:

```python
!curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_colab.sh | bash
```

وللتشغيل من نسخة مستنسخة محلياً داخل الجلسة:

```python
!bash scripts/install/setup_colab.sh
```

### Ubuntu / Debian

```bash
bash scripts/install/setup_ubuntu.sh
```

يمكن تثبيت وتشغيل النسخة المنشورة من المصدر الرسمي:

```bash
curl -fsSL https://raw.githubusercontent.com/Alqudimi/rafidain/main/scripts/install/setup_ubuntu.sh | bash
```

لتثبيت خدمة systemd (يتطلب صلاحيات مناسبة):

```bash
sudo bash scripts/install/setup_ubuntu.sh --engine seedvc --service
systemctl status rafidain
journalctl -u rafidain -f
```

### Windows 10/11

من PowerShell داخل مجلد المشروع:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\install\setup_windows.ps1
```

تدعم المثبتات أعلاماً مثل اختيار `--engine`, `--device`, `--port`, `--api-key`, `--no-models`, `--no-tests` و`--foreground`. راجع `bash scripts/install/setup_ubuntu.sh --help` أو [دليل المثبتات](INSTALL-SCRIPTS.md) للتفاصيل المتاحة لكل نظام.

> لا تنفّذ سكربتاً من الإنترنت بصلاحيات مدير قبل قراءة محتواه ومراجعته. مفاتيح API والأنفاق العامة حساسة؛ لا تشاركها.

## Docker

يتطلب Docker وDocker Compose. شغّل من جذر المستودع:

```bash
export RAFIDAIN__API_KEYS='["استبدلني-بمفتاح-سري-قوي"]'
docker compose -f docker/docker-compose.yml up -d --build
```

تحقق من الخدمة:

```bash
curl -fsS http://127.0.0.1:8000/healthz
curl -fsS http://127.0.0.1:8000/readyz
```

لعامل التدريب ضمن profile الخاص به:

```bash
export RAFIDAIN_DATASETS_DIR=./datasets
export RAFIDAIN_RUNS_DIR=./runs
docker compose -f docker/docker-compose.yml --profile finetune up -d
```

راجع [DEPLOYMENT.md](DEPLOYMENT.md) لمعرفة تفاصيل GPU، التخزين، المراقبة، TLS، التوسع والتحصين. لا تعرض منفذ الخدمة مباشرة للإنترنت دون مصادقة وTLS وضبط الشبكة.

## واجهة الويب

لتشغيل الواجهة مع المحرك الحقيقي:

```bash
make web
```

أو بالمحرك الوهمي دون GPU:

```bash
make web-mock
```

أوامر الإدارة الإضافية:

```bash
make status
make stop
make web-tunnel   # بدء واجهة مع نفق عام مؤقت عند دعم البيئة
make tunnel-stop
make tunnel-url
```

قد يكون رابط النفق عاماً ومتاحاً لمن يحصل عليه؛ استخدم مفتاح API، ولا تضع بيانات حساسة في بيئة مكشوفة.

## استخدام REST وWebSocket

تظهر جميع النهايات ومخطط الطلبات المحدث في Swagger UI على `/docs` وOpenAPI JSON على `/openapi.json`.

### 1. رفع صوت مرجعي

الصوت المرجعي هو صوت المتحدث المستهدف، ويُفضّل أن يكون واضحاً وخالياً من الضجيج. مثال:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/voices \
  -H 'X-API-Key: ضع-مفتاحك-هنا' \
  -F 'file=@reference.wav' \
  -F 'name=الصوت-الهدف'
```

احتفظ بمعرّف الصوت (`voice_id`) الذي يرجع من الاستجابة.

### 2. إنشاء جلسة تحويل

```bash
curl -X POST http://127.0.0.1:8000/api/v1/sessions \
  -H 'X-API-Key: ضع-مفتاحك-هنا' \
  -H 'Content-Type: application/json' \
  -d '{"voice_id":"VOICE_ID","latency_profile":"balanced"}'
```

تتضمن الاستجابة عنوان WebSocket للجلسة. عند وجود مفتاح API، أرسله بالطريقة المدعومة في إعدادات الخدمة/العميل، ولا تضعه في روابط عامة أو سجلات.

### 3. تدفق الصوت

يرسل العميل إطارات PCM أحادية القناة بصيغة `s16le` ومعدل 16 kHz، مع رسائل التحكم JSON. دورة الرسائل تشمل `stream.start` و`stream.pause` و`stream.resume` و`stream.stop` و`metrics.poll`. استقبل رسائل الحالة والصوت والمقاييس من الخادم. ينبغي على العميل الالتزام بإيقاع الزمن الحقيقي وحجم الإطار المتوقع وعدم إرسال تسجيل كامل بسرعة دفعة واحدة.

مثال عميل اختبار مضمّن:

```bash
uv run python scripts/smoke_ws_client.py \
  --url http://127.0.0.1:8000 \
  --seconds 3 \
  --api-key 'ضع-مفتاحك-هنا'
```

راجع [توثيق API التفاعلي](http://127.0.0.1:8000/docs) و[ملف البروتوكول والمعمارية](ARCHITECTURE.md) لمعرفة مخطط رسائل WebSocket ورموز الإغلاق.

### ملفات زمن الوصول

تتوفر ملفات `ultra_low` و`balanced` و`quality` بمفاضلات مختلفة بين حجم الكتلة وجودة التحويل والتأخير. القيم الفعلية تعتمد على الجهاز وحمولة الجلسات؛ راجع تعريفات الإعداد وقياس `make bench` قبل اختيار ملف الإنتاج.

## ضبط النموذج للهجة العراقية

يتضمن المشروع مساراً لإعداد البيانات ثم تدريب Seed-VC وتقييمه. التدريب الثقيل يحتاج GPU ومساحة تخزين واتصالاً بمصادر البيانات. للمسار الموجّه عبر دفاتر التدريب استخدم الروابط في [قسم الدفاتر](#دفاتر-التدريب-colab-وrunpod-والمنصات-الأخرى)، وللمسار المحلي راجع [FINETUNING.md](FINETUNING.md).

### مسار CLI مختصر

```bash
# حضّر بياناتك الصوتية في مجلد مناسب، ثم أنشئ البيانات المهيأة
uv run python finetune/prepare_dataset.py --input-dir ./raw_iraqi

# تدريب النموذج على البيانات المهيأة
uv run python finetune/run_finetune.py \
  --dataset-dir ./raw_iraqi/prepared \
  --run-name iraqi_v1 \
  --max-steps 2000 \
  --prepare

# اعرض الخيارات المتاحة بحسب إصدار الكود
uv run python finetune/prepare_dataset.py --help
uv run python finetune/run_finetune.py --help
```

لتقييم تحويل صوت وقياس الأداء:

```bash
uv run python evaluation/evaluate_conversion.py --help
uv run python evaluation/benchmark_stream.py --help
make bench
```

استخدم `--help` للتحقق من خيارات ملفات الإدخال والمخرجات في إصدارك قبل التشغيل. تتغير نتائج التدريب باختلاف التسجيلات، جودتها، إعداد GPU والإعدادات المستخدمة. لا ترفع بيانات صوتية دون إذن وتحقق من تراخيص مصادرها.

## الاختبارات وجودة الكود

تعمل اختبارات المشروع الافتراضية دون GPU باستخدام المحرك الوهمي:

```bash
make test
make lint
make typecheck
```

أو مباشرة:

```bash
uv run pytest -q
uv run ruff check rafidain tests finetune evaluation scripts
uv run mypy rafidain
```

اختبار دخان متكامل للخادم وWebSocket:

```bash
bash scripts/smoke_e2e.sh
```

وتشغيل فحص أداء التدفق على الجهاز:

```bash
make bench
```

يفعّل CI في GitHub Actions فحوص Ruff والاختبارات وفحصاً سريعاً للمحرك الوهمي عند الدفع إلى `main` وعند فتح Pull Request.

## الإعدادات والأمان

تُقرأ الإعدادات من متغيرات البيئة ذات السابقة `RAFIDAIN__`، ثم ملف `.env`، ثم القيم الافتراضية. للحقول المتداخلة استخدم `__`، مثلاً `RAFIDAIN__ENGINE__DEVICE`.

| المتغير | مثال | الغرض |
|---|---|---|
| `RAFIDAIN__ENGINE__IMPL` | `seedvc` أو `mock` | اختيار المحرك |
| `RAFIDAIN__ENGINE__DEVICE` | `cuda` أو `cpu` أو `auto` | الجهاز المستهدف |
| `RAFIDAIN__API_KEYS` | `'["مفتاح-سري"]'` | قائمة مفاتيح المصادقة بصيغة JSON |
| `RAFIDAIN__CORS_ORIGINS` | `'["https://example.com"]'` | أصول CORS المسموحة بصيغة JSON |
| `RAFIDAIN__DATA_DIR` | `./data` | مجلد بيانات التطبيق |
| `RAFIDAIN__ENGINE__LOCAL_CHECKPOINT` | مسار ملف | نقطة نموذج محلية مضبوطة |
| `RAFIDAIN__ENGINE__LOCAL_CONFIG` | مسار ملف | إعداد النموذج المحلي |
| `RAFIDAIN__REDIS_URL` | `redis://localhost:6379/0` | إعداد Redis عند الحاجة إليه |

إرشادات أساسية قبل النشر:

1. أنشئ مفتاحاً عشوائياً قوياً، وفعّل المصادقة. لا تستخدم `--no-auth` في الإنتاج.
2. لا ترفع `.env` أو الرموز أو تسجيلات الصوت إلى GitHub. تحقق من `.gitignore` قبل الالتزام.
3. استخدم HTTPS/WSS عبر reverse proxy موثوق، وقيّد CORS على نطاقاتك بدلاً من `*`.
4. لا تعرض Redis أو لوحة الإدارة للإنترنت دون حماية. قيّد الوصول إلى ملفات الأصوات والنماذج وخذ نسخاً احتياطية آمنة.
5. احصل على موافقة صريحة قبل تحويل صوت شخص أو نشر نموذج قادر على محاكاته، ووضّح الاستخدام عند الاقتضاء.

## هيكل المستودع

```text
rafidain/
├── rafidain/                 # تطبيق الخدمة: API، تطبيق، نطاق، بنية تحتية، واجهة
├── finetune/                 # إعداد البيانات والتدريب وإعداداته
├── evaluation/               # تقييم التحويل وقياس التدفق
├── notbooks/                 # دفاتر Colab وRunPod
├── scripts/                  # إدارة المشروع والتثبيت والتشغيل والاختبار
├── tests/                    # اختبارات الوحدات والتكامل
├── docker/                   # Dockerfile وDocker Compose
├── ARCHITECTURE.md           # المعمارية وتدفق المكونات
├── DEPLOYMENT.md             # النشر والتشغيل الإنتاجي
├── FINETUNING.md             # دليل الضبط الدقيق
├── INSTALL-SCRIPTS.md        # دليل المثبتات الجاهزة
├── WEBUI.md                  # الواجهة
├── requirements.txt          # اعتماديات الخدمة الأساسية
├── requirements-finetune.txt # اعتماديات التدريب
├── pyproject.toml            # إعداد الحزمة وأدوات التطوير
└── Makefile                  # أوامر مختصرة للمهمات الشائعة
```

## استكشاف الأخطاء

| المشكلة | ما الذي تتحقق منه؟ |
|---|---|
| لا يظهر GPU | تأكد من تفعيل GPU في Colab/RunPod وثبّت تعريف NVIDIA المناسب، ثم افحص `nvidia-smi`. |
| النموذج لا يصبح جاهزاً | راجع سجلات الخادم، اتصال Hugging Face، مساحة القرص، ومسارات الأوزان؛ افحص `/readyz`. |
| خطأ مفتاح API | استخدم `X-API-Key` وتحقق من أن `RAFIDAIN__API_KEYS` قائمة JSON صالحة. |
| المنفذ مستخدم | غيّر المنفذ عبر خيارات التشغيل أو إعداد `port`، ثم حدّث عنوان العميل. |
| نفاد ذاكرة GPU | خفّض التوازي/حجم الدفعة أو استخدم GPU بذاكرة أكبر؛ تدريب النموذج يستهلك موارد أكثر من المحرك الوهمي. |
| Colab فقد الملفات | خزّن المخرجات والنقاط في Google Drive أو مكان دائم؛ مساحة الجلسة المؤقتة قد تختفي عند إعادة ضبطها. |
| مشاكل CUDA/PyTorch | استخدم إصدارات متوافقة مع GPU والتعريف وPython؛ اتبع تعليمات الدفتر الخاص ببيئتك ولا تخلط عجلات CUDA. |
| فشل تثبيت حزمة صوت | ثبّت `ffmpeg` وأعد فحص إصدار Python وبيئة `uv`، ثم راجع رسالة الخطأ كاملة. |

للمساعدة، افتح [Issue جديداً](https://github.com/Alqudimi/rafidain/issues/new) وأرفق نظام التشغيل وإصدارات Python/PyTorch/CUDA ورسالة الخطأ بعد إزالة أي أسرار أو بيانات خاصة.

## الوثائق والمراجع

- [المعمارية](ARCHITECTURE.md) · [النشر](DEPLOYMENT.md) · [الضبط الدقيق](FINETUNING.md) · [المثبتات](INSTALL-SCRIPTS.md) · [الواجهة](WEBUI.md)
- [دليل دفاتر Colab](notbooks/README_colab_training.md) · [دليل RunPod V3](notbooks/README_runpod_training_V3.md) · [دليل السكربتات](scripts/README.md)
- [Seed-VC](https://github.com/Plachtaa/seed-vc) · [أوزان Seed-VC على Hugging Face](https://huggingface.co/Plachta/Seed-VC)
- [CAMPPlus](https://huggingface.co/funasr/campplus) · [CosyVoice / HiFT](https://huggingface.co/FunAudioLLM/CosyVoice-300M) · [XLS-R](https://huggingface.co/facebook/wav2vec2-xls-r-300m)
- [FastAPI](https://fastapi.tiangolo.com/) · [uv](https://docs.astral.sh/uv/) · [PyTorch](https://pytorch.org/) · [Google Colab](https://colab.research.google.com/) · [RunPod](https://www.runpod.io/)

## المساهمة والترخيص

المساهمات عبر Pull Requests وIssues مرحب بها. قبل المساهمة شغّل الاختبارات والـ lint، ولا ترفع ملفات بيانات أو أوزان كبيرة أو أسراراً. تحقق من ملف الترخيص في المستودع ومن شروط تراخيص Seed-VC والأوزان والبيانات التابعة قبل إعادة الاستخدام أو التوزيع؛ تنطبق تراخيص الأطراف الثالثة بصورة مستقلة.

---

**المستودع:** [github.com/Alqudimi/rafidain](https://github.com/Alqudimi/rafidain) · **الفرع:** `main`

</div>
