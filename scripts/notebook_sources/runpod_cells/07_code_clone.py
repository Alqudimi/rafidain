# ============================================================ #
#  الخلية ٥ — استنساخ المستودعات + بصمات قابلية إعادة الإنتاج   #
#  Seed-VC (المدرب الرسمي) + رافِدَين (خط البيانات العراقية)    #
#  (على القرص الدائم — يُحدثان إن وُجدا بدلاً من إعادة التنزيل) #
# ============================================================ #
import subprocess, json
from pathlib import Path

SEEDVC_COMMIT = "51383efd921027683c89e5348211d93ff12ac2a8"

def git_clone(url, dst, revision=None):
    if (dst / ".git").exists():
        print(f"تحديث {dst.name} ...")
        if revision:
            sh(["git", "-C", str(dst), "fetch", "--depth", "1", "origin", revision])
            sh(["git", "-C", str(dst), "checkout", "--detach", revision])
        else:
            sh(["git", "-C", str(dst), "pull", "--ff-only"], check=False)
    else:
        print(f"استنساخ {dst.name} من {url} ...")
        sh(["git", "clone", "--depth", "1", url, str(dst)])
        if revision:
            sh(["git", "-C", str(dst), "fetch", "--depth", "1", "origin", revision])
            sh(["git", "-C", str(dst), "checkout", "--detach", revision])

git_clone("https://github.com/Plachtaa/seed-vc.git", SEEDVC, revision=SEEDVC_COMMIT)
git_clone("https://github.com/Alqudimi/rafidain.git", RAFIDAIN)

# تحقق من الملفات الحرجة
for f in ("train.py", "inference.py", "configs/presets/config_dit_mel_seed_uvit_xlsr_tiny.yml"):
    assert (SEEDVC / f).exists(), f"ملف مفقود في seed-vc: {f}"
assert (RAFIDAIN / "seed_vc_iraqi_ubuntu.sh").exists(), "سكربت خط البيانات مفقود في rafidain"

# ---- بصمات الإصدارات (تُحفظ مع حزمة النموذج لاحقاً — قابلية إعادة الإنتاج) ----
def git_hash(path):
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short", "HEAD"],
                           capture_output=True, text=True, timeout=30)
        return r.stdout.strip() or "?"
    except Exception:
        return "?"

repro = {
    "notebook_version": NB_VERSION,
    "seedvc_commit": git_hash(SEEDVC),
    "seedvc_expected_commit": SEEDVC_COMMIT,
    "rafidain_commit": git_hash(RAFIDAIN),
    "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
(ROOT / "repro.json").write_text(json.dumps(repro, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n[بصمات] seed-vc @ {repro['seedvc_commit']} | rafidain @ {repro['rafidain_commit']}")
print(f"[جاهز] seed-vc   : {SEEDVC}")
print(f"[جاهز] rafidain  : {RAFIDAIN}")
