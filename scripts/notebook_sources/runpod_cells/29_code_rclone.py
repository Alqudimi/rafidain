# ============================================================ #
#  الخلية ٢٣ — (اختيارية) طبقة رابعة: rclone سحابي             #
#  Google Drive / S3 / Dropbox / OneDrive ...                  #
#  خلية مرجعية — نفّذ الأوامر يدوياً وقت الحاجة                 #
# ============================================================ #

# 1) التثبيت (مرة واحدة على الحاوية):
#      apt-get install -y rclone          # أو أحدث إصدار من rclone.org/install
#
# 2) الإعداد التفاعلي (مرة واحدة — يربط حسابك السحابي):
#      rclone config                      # أنشئ remote باسم مثلاً: mydrive
#      (ملف الإعداد يبقى في /root/.config/rclone/rclone.conf — انسخه إلى
#       /workspace/iraqi_seedvc/ ليبقى مع القرص الدائم بين الحاويات)
#
# 3) رفع حزمة النموذج كاملة (أوزان + تقارير تقييم + بنشمارك):
#      rclone copy /workspace/iraqi_seedvc/exports/{run} mydrive:SeedVC_Iraqi/ -P
#
# 4) تنزيله لاحقاً من أي جهاز آخر فيه rclone مضبوط على الحساب نفسه:
#      rclone copy mydrive:SeedVC_Iraqi ./mymodel -P

print("خلية مرجعية — الأوامر جاهزة في التعليقات أعلاه.")
print("الطبقة الدائمة الأساسية تبقى HuggingFace Hub (الخلية 22) — الأسهل والأضمن.")
