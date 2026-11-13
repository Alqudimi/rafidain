# ============================================================ #
#  الخلية ٢٥ — النشر: كيف تستخدم النموذج المضبوط؟              #
#  (خلية إرشادية تطبع الأوامر الجاهزة بمساراتك الفعلية)        #
# ============================================================ #
from pathlib import Path

EXPORT_DIR = EXPORTS / CFG["RUN_NAME"]
W = EXPORT_DIR / "seed_uvit_tat_xlsr_tiny_iraqi.pth"
C = EXPORT_DIR / "config_dit_mel_seed_uvit_xlsr_tiny.yml"

if not W.exists():
    print("شغّل خلية التصدير (20) أولاً حتى تتوفر الحزمة.")
else:
    # لوحة نتائج التقييم إن وُجدت (تذكير بأرقام هذا التشغيل)
    import json
    ab = EXPORT_DIR / "ab_eval" / "ab_eval.json"
    bench = EXPORT_DIR / "bench_stream" / "bench_stream.json"
    if ab.exists():
        s = json.loads(ab.read_text(encoding="utf-8"))["summary"]
        print(f"أرقام هذا التشغيل: تشابه متحدث {s['base_speaker_sim_mean']:.4f} ← {s['ft_speaker_sim_mean']:.4f} "
              f"(فرق {s['speaker_sim_delta']:+.4f}) | حفظ محتوى {s['ft_content_sim_mean']:.4f}")
    if bench.exists():
        b = json.loads(bench.read_text(encoding="utf-8"))["streaming"].get("balanced", {})
        if b:
            print(f"                   زمن وصول شامل (balanced): {b['estimated_e2e_ms']:.0f}ms | RTF {b['rtf']:.2f}")
    print(f'''
==========================================================
 نشر النموذج المضبوط — ثلاثة مسارات جاهزة
==========================================================

(أ) استدلال فوري في هذه الحاوية (البيئة جاهزة الآن):
    cd {SEEDVC}
    {VENV_PY} inference.py \\
        --checkpoint {W} \\
        --config     {C} \\
        --source <صوت-المتكلم-الأصلي.wav> \\
        --target  <صوت-مرجعي-عراقي.wav> \\
        --output  ./out

(ب) واجهة ويب Gradio للتجربة السريعة (تحتاج تثبيت gradio أولاً):
    uv pip install --python {VENV_PY} gradio   # أو: {VENV_PY} -m pip install gradio
    cd {SEEDVC}
    {VENV_PY} app_vc.py --checkpoint {W} --config {C}
    ثم افتح المنفذ 7860 (فعّل بروكسي HTTP من إعدادات الحاوية في لوحة RunPod)

(ج) خدمة رافِدَين الكاملة (REST + WebSocket لبث لحظي في المكالمات):
    المستودع منسوخ هنا: {RAFIDAIN}
    انسخ ما يلي في ملف .env الخاص بالخدمة (انظر .env.example داخل المستودع):

      RAFIDAIN__ENGINE__LOCAL_CHECKPOINT={W}
      RAFIDAIN__ENGINE__LOCAL_CONFIG={C}
      RAFIDAIN__ENGINE__DEVICE=cuda
      RAFIDAIN__API_KEYS=["<مفتاح-API-عشوائي-خاص-بك>"]

    التفاصيل الكاملة (Docker/GPU، الأمان، TLS، القياس):
      {RAFIDAIN}/DEPLOYMENT.md   و   {RAFIDAIN}/WEBUI.md

تذكير أخلاقي: استنساخ صوت شخص حقيقي يتطلب موافقته الصريحة الموثقة،
وأفصح للطرف الآخر عند ما يوجبه القانون — راجع قسم الأمان في README المستودع.
''')
