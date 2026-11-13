# ============================================================ #
#  الخلية ٢٢ — الرفع الدائم إلى HuggingFace Hub                #
#  النموذج يُخزَّن في حسابك أنت، مجاناً، بلا علاقة بـ RunPod     #
#  (أعد تشغيلها في أي وقت مستقبلاً — ولو بعد شهور)              #
# ============================================================ #
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
HF_SCRIPT = ROOT / "hf_upload.py"
HF_SCRIPT.write_text(r'''
import os, time
from huggingface_hub import HfApi

token = os.environ["HF_TOKEN"]
export_dir = os.environ["NB_EXPORT_DIR"]
api = HfApi(token=token)
user = api.whoami()["name"]
repo_id = os.environ.get("NB_REPO_ID", "").strip() or f"{user}/rafidain-seedvc-iraqi"
private = os.environ.get("NB_PRIVATE", "1") == "1"
api.create_repo(repo_id, private=private, exist_ok=True)
print(f"[hf] رفع {export_dir} إلى {repo_id} (خاص = {private}) ...")
api.upload_folder(
    folder_path=export_dir,
    repo_id=repo_id,
    repo_type="model",
    commit_message=f"Rafidain Iraqi fine-tune ({time.strftime('%Y-%m-%d %H:%M')})",
)
print("[hf] REPO_URL: https://huggingface.co/" + repo_id)
''', encoding="utf-8")

if not CFG["HF_TOKEN"]:
    print("لا يوجد رمز HuggingFace بعد — لا مشكلة: كل أصولك محفوظة على القرص الشبكي،")
    print("ويمكنك تنفيذ الرفع الدائم في أي وقت لاحق (الخلية تعمل مستقلة):\n")
    print("  1) أنشئ حساباً مجانياً في huggingface.co إن لم يكن لديك")
    print("  2) من hf.co/settings/tokens أنشئ New Token بصلاحية Write وانسخه")
    print('  3) في الخلية 1 اضبط:  CFG["HF_TOKEN"] = "hf_ضع-الرمز-هنا"  ثم أعد تشغيلها')
    print("  4) أعد تشغيل هذه الخلية — سترفع الحزمة كاملة إلى مستودع خاص في حسابك")
else:
    assert (EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth").exists(), "شغّل خلية التصدير (20) أولاً"
    env = dict(ENV)
    env["HF_TOKEN"] = str(CFG["HF_TOKEN"])
    env["NB_EXPORT_DIR"] = str(EXPORT_DIR)
    env["NB_REPO_ID"] = str(CFG.get("HF_REPO_ID", ""))
    env["NB_PRIVATE"] = "1" if CFG["HF_PRIVATE"] else "0"
    sh([VENV_PY, HF_SCRIPT], env=env)
    print('''
[تم] النموذج الآن في حسابك على HuggingFace — بشكل دائم.
لتنزيله في أي وقت مستقبلاً من أي جهاز (بلا RunPod):
    pip install -U "huggingface_hub[cli]"
    hf download <اسم-حسابك>/rafidain-seedvc-iraqi --local-dir ./mymodel
أو من بايثون:
    from huggingface_hub import snapshot_download
    snapshot_download("<اسم-حسابك>/rafidain-seedvc-iraqi", local_dir="./mymodel")
أو مباشرة من صفحة المستودع في المتصفح.''')
