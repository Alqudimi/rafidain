# Rafidain scripts

كل سكربتات التشغيل تستخدم `uv` و`Makefile`/`scripts/manage.py` كنقطة دخول موحدة. لا تستخدم هذه السكربتات `pip install` ولا تستدعي `inference.py` القديم.

## التثبيت

- `install/setup_uv.sh`: المثبت الرئيسي لنظام Linux/macOS/WSL وColab، مع كشف النظام وGPU وتهيئة checkpoints وCloudflare.
- `install/setup_ubuntu.sh`: اسم توافق لنفس المثبت.
- `install/setup_colab.sh`: نسخة Colab؛ لا تفتح tunnel تلقائيًا، وتدعم `--gdrive auto` عند التدريب.
- `../seed_vc_iraqi_ubuntu.sh` و`../seed_vc_iraqi_colab.sh`: أسماء قديمة محفوظة للتوافق وتفوض التنفيذ إلى المثبت الموحد.
- `../seed_vc_iraqi_windows.ps1`: مثبت PowerShell يستخدم `uv` ويدعم mock/Seed-VC وCloudflare.

## التشغيل

```bash
bash scripts/run_web.sh --engine mock --no-tunnel
bash scripts/web/launch_webapp.sh --engine seedvc --port 8000
bash scripts/web/launch_webapp.sh --status
bash scripts/web/launch_webapp.sh --stop
```

يشغّل المشغل `/web` و`/live`، ويستخدم `make tunnel` عند عدم تمرير `--no-tunnel`.

## التدريب وColab

```bash
bash scripts/colab_setup.sh --engine seedvc --no-models
bash scripts/train.sh \
  --dataset-dir ./datasets/iraqi \
  --run-name iraqi_v1 \
  --gdrive auto
```

يدعم `run_finetune.py` الاستعادة من أحدث checkpoint ومزامنة Google Drive ذريًا عبر `--gdrive auto` أو `--gdrive-dir`.

## الاختبار

```bash
bash scripts/smoke_e2e.sh
uv run python scripts/smoke_ws_client.py --url http://127.0.0.1:8000 --seconds 3
```
