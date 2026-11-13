# ============================================================ #
#  الخلية ٢١ — أرشيف tar.gz واحد جاهز للتنزيل                  #
# ============================================================ #
import shutil
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "شغّل خلية التصدير (20) أولاً"

base = EXPORTS / f"rafidain_seedvc_{CFG['RUN_NAME']}"
archive = shutil.make_archive(str(base), "gztar", root_dir=str(EXPORTS), base_dir=EXPORT_DIR.name)
size_mb = Path(archive).stat().st_size / 1e6
print(f"الأرشيف جاهز: {archive}  ({size_mb:.1f} MB)")
print(f'''
لتنزيله الآن:
  • من Jupyter: لوحة الملفات (يساراً) → مجلد exports → كليك يمين على الملف → Download
  • أو من جهازك عبر SSH (تجد الأمر في لوحة RunPod → Connect → SSH over TLS):
      scp -P <المنفذ> root@<العنوان>:{archive} .
''')
