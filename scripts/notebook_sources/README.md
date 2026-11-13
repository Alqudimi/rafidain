<div dir="rtl">

# مصادر بناء دفترَي التدريب

الدفتران المُسلّمان مبنيان من هذه المصادر — استخدمها لإعادة التوليد أو التعديل مستقبلاً
بدل تحرير ملف `.ipynb` مباشرة.

## دفتر RunPod — `runpod_cells/`

33 ملفاً مرقّماً (`00`–`32`)، ملف واحد لكل خلية بالترتيب نفسه الذي يظهر في
`rafidain/rafidain_runpod_training.ipynb`:

- `NN_md_*.md` → خلايا توثيق (Markdown)
- `NN_code_*.py` → خلايا كود

التحقق من صحة الكود المضمّن داخل الخلايا: `python3 check_embedded.py`

## دفتر Google Colab — `colab_cells_a/b/c.py`

ثلاث وحدات Python تصف الخلايا كسلاسل نصية (`CELLS_A` + `CELLS_B` + `CELLS_C`)،
يجمعها منشئ الدفتر ويتحقق منه تحققاً كاملاً (JSON، صحة نحو كل خلية كود، تسلسل
الترقيم، ثوابت خلية الإعداد، استخراج خط البيانات من سكربت المستودع):

```bash
python3 build_colab_nb.py   # يكتب ../rafidain/rafidain_colab_training.ipynb
```

## اختبار منطق الخلايا

```bash
python3 test_notebook_logic.py   # تشغيل جاف لمنطق الخلايا الحرجة (المزامنة، الاسترجاع، السجلات)
```
</div>
