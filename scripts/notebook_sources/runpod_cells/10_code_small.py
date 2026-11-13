# ============================================================ #
#  الخلية ٧ — تنزيل المجموعات العراقية النظيفة                  #
#  youssef / shiekh / safaa (HuggingFace)                       #
#  + lahgtna: ≈50 ساعة عراقية من Lahgtna v2 (الجزء الأكبر)     #
#    عبر datasets-server بترقيم صفحات كامل                     #
#  + kharrufa: 450 مقطع استوديو من Google Drive                #
#  قابلة للاستئناف: أعد التشغيل لتكملة ما فشل فقط              #
# ============================================================ #
if CFG["SKIP_SMALL"]:
    print("تخطي المجموعات النظيفة الخمس (SKIP_SMALL=True) — إن أردت فعّلها من الخلية 1.")
else:
    import shutil
    free0 = shutil.disk_usage(ROOT).free / 1e9
    print(f"المساحة الحرة قبل التنزيل: {free0:.1f}GB (المرحلة كلها ≈7.5GB، أكبر عنصر فيها lahgtna)")
    sh([VENV_PY, PIPE, "small",
        "--root", ROOT,
        "--workers", int(CFG["FETCH_WORKERS"])])
    AUD = {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".webm"}
    for name, note in (("youssef", "مقاطع قصيرة نظيفة"),
                       ("shiekh", "متحدث ديني واحد"),
                       ("safaa", "صوت أنثى عراقي"),
                       ("lahgtna", "≈50 ساعة عراقية إضافية — جديد ٢.١"),
                       ("kharrufa", "450 مقطع استوديو — جديد ٢.١")):
        d = ROOT / "raw" / name
        n = len([p for p in d.iterdir() if p.suffix.lower() in AUD]) if d.exists() else 0
        print(f"  {name:<8}: {n:,} ملفاً | {note}")
    free1 = shutil.disk_usage(ROOT).free / 1e9
    print(f"استُهلك {free0-free1:.1f}GB | الحر الآن {free1:.1f}GB")
    st = ROOT / "raw" / "_lahgtna_state.json"
    if st.exists():
        import json
        s = json.loads(st.read_text())
        fp, fa = len(s.get("failed_pages", [])), len(s.get("failed_audio", []))
        if fp or fa:
            print(f"[ملاحظة] lahgtna: {fp} صفحة و{fa} مقطعاً فشلا مؤقتاً — "
                  "أعد هذه الخلية لإعادة المحاولة فقط (الموجود لا يُعاد)")
print("\n[جاهز] تابع إلى الخلية 8 (YODAS) — أو الخلية 9 لدمج ملفاتك الخاصة.")
