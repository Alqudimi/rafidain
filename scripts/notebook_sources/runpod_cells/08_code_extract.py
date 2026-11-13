# ============================================================ #
#  الخلية ٦ — استخراج خط بيانات العراق المدمج                   #
#  (الكتالوج العراقي كله مضمّن داخل سكربت المستودع نفسه:         #
#   49,194 مقطعاً / 339 ساعة / خريطة إزاحات بايت-دقيقة           #
#   داخل أرشيفات espnet/yodas3 — نستخرجه هنا كوحدة مستقلة)      #
# ============================================================ #
import re, gzip, base64, json, ast, urllib.request
from pathlib import Path

SRC_SH = RAFIDAIN / "seed_vc_iraqi_ubuntu.sh"
if not SRC_SH.exists():   # احتياط: المستودع غير منسوخ — ننزّل السكربت مباشرة من GitHub
    print("تنزيل السكربت من GitHub مباشرة ...")
    SRC_SH.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        "https://raw.githubusercontent.com/Alqudimi/rafidain/main/seed_vc_iraqi_ubuntu.sh", SRC_SH)

text = SRC_SH.read_text(encoding="utf-8")
m = re.search(r"cat > \"\$ROOT/iraqi_pipeline\.py\" <<'PYEOF'\n(.*?)\nPYEOF\n", text, re.S)
assert m, "لم يُعثر على خط البيانات المدمج داخل السكربت!"
code = m.group(1)
ast.parse(code)   # تحقق من سلامة الكود المستخرج قبل الكتابة
PIPE.write_text(code, encoding="utf-8")
print(f"[خط البيانات] استُخرج إلى: {PIPE}  ({len(code):,} بايت)")

# ---- إحصاءات الكتالوج العراقي المدمج (بدون أي تنزيل) ----
def _grab(name):
    blob = re.search(name + r' = r"""(.*?)"""', code, re.S).group(1)
    return json.loads(gzip.decompress(base64.b64decode(blob)))

clips, offs, srcrank = _grab("_IQ_CLIPS_B64"), _grab("_IQ_OFFS_B64"), _grab("_IQ_SRC_B64")
hours_total = sum(c["du"] for c in clips) / 3600
resolvable = [s for s, v in offs.items() if v.get("off")]
print(f"[كتالوج اللهجة العراقية] {len(clips):,} مقطعاً | {hours_total:.0f} ساعة | "
      f"{len(resolvable):,} تسجيل مصدر قابل للجلب | {sum(offs[s]['size'] for s in resolvable)/1e9:.1f}GB لو جُلب كله")

# ---- حجم هدفك الحالي (نفس منطق الاختيار في السكربت: الجودة أولاً) ----
t = str(CFG["YODAS_HOURS"]).strip().lower()
if t in ("0", ""):
    print(f"\n[هدفك] YODAS معطّل (YODAS_HOURS=0) — الخلية 8 ستتخطاه. الأصل المتاح: {hours_total:.0f} ساعة.")
else:
    target = None if t in ("all", "كل") else float(t)
    ranked = sorted(srcrank.items(), key=lambda kv: -kv[1][2])
    chosen, h = [], 0.0
    for sid, (tar, ch, q) in ranked:
        if sid not in offs or offs[sid].get("off") is None:
            continue
        chosen.append(sid); h += ch
        if target and h >= target:
            break
    size_gb = sum(offs[s]["size"] for s in chosen) / 1e9
    label = f"{target:.0f} ساعة" if target else "الكتالوج كله"
    print(f"[هدفك] {label}: {len(chosen):,} تسجيلاً مصدراً ≈ {size_gb:.1f}GB تنزيلاً خاماً (سيُشفَّر إلى Opus للتدريب)")
print("\n[جاهز] تابع إلى خلايا البيانات (7 وما بعدها).")
