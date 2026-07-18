# Local-LLM-Trainer
Training an LLM locally
# 🧠 Local LLM Trainer (Selective & Memory-Efficient)

منصة تدريب نماذج اللغة الكبيرة (LLMs) محلياً، مُصممة خصيصاً للعمل بكفاءة على الأجهزة ذات الموارد المحدودة (مثل GTX 1080 Ti بـ 11GB VRAM و 16GB RAM).

## ✨ الميزات الرئيسية
- 🚀 **تدريب انتقائي (Selective Training):** تحميل البيانات من السحابة (HuggingFace) وتصفيتها ذكياً للاحتفاظ بالعينات ذات الأهمية العالية فقط.
- 💾 **ذاكرة خارجية ذكية (Smart External Memory):** استخدام تقنيات Memory-Mapped (mmap) لتخزين البيانات على الـ SSD باستخدام ~0% من الـ RAM.
- ⚡ **تحسينات الذاكرة القصوى:** دمج 4-bit Quantization (NF4)، DoRA، و Gradient Checkpointing.
- 🔄 **توافق مع معالجات متعددة الأنوية:** تحسينات مخصصة لاستغلال معالجات مثل Xeon بأمان دون استهلاك ذاكرة عشوائية زائدة.

## 🖥️ متطلبات الجهاز الموصى بها
- **GPU:** NVIDIA GTX 1080 Ti (11GB VRAM) أو أعلى.
- **RAM:** 16 GB فأكثر.
- **OS:** Linux (CachyOS, Ubuntu, Arch) موصى به بشدة.
- **Storage:** SSD بسعة 50GB+ على الأقل للذاكرة الخارجية.

## 🚀 البدء السريع

### 1. استنساخ المستودع
```bash
git clone https://github.com/samkoli145/Local-LLM-Trainer.git
cd Local-LLM-Trainer
