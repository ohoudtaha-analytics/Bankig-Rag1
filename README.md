# مساعد سياسة الائتمان الذكي (Credit Policy RAG Assistant)

مساعد RAG بيجاوب على أسئلة موظفي الائتمان بناءً على نصوص رسمية من كتاب اللوائح
الصادر عن **البنك المركزي المصري** — الباب الرابع: ضوابط منح الائتمان (6 أقسام حالياً).

## هيكل المشروع

```
01_documents.py              -> تحميل واستخراج نص الـ PDFs الخام
02_preprocessing.py          -> تنظيف النص (تصحيح ترتيب الحروف + تنظيف عام)
03_chunking.py                -> تقسيم النص لأجزاء (chunks) قابلة للاسترجاع
04_vector_representation.py  -> تحويل الأجزاء لـ embeddings (sentence-transformers)
05_create_chroma_store.py    -> بناء الـ vector store (ChromaDB)
06_retrieve_context.py       -> استرجاع أقرب الأجزاء لسؤال المستخدم + مقاييس تقييم
07_prompting.py               -> بناء الـ prompt واستدعاء الموديل عبر OpenRouter
streamlit_app.py              -> واجهة Streamlit النهائية
requirements.txt
data/raw_pdfs/                -> ملفات الـ PDF المصدر (6 أقسام من البنك المركزي)
```

## ملاحظة مهمة عن جودة النصوص المصدر

أثناء استخراج النص من ملفات الـ PDF، تم اكتشاف مشكلتين حقيقيتين تم توثيقهما ومعالجتهما في `02_preprocessing.py`:

1. **انعكاس ترتيب الحروف**: كل الأقسام الستة تظهر بترتيب حروف معكوس عند
   الاستخراج بـ `pdfplumber` (مشكلة شائعة مع بعض خطوط PDF العربية). تم إصلاحها
   تلقائياً، مع الحفاظ على ترتيب الأرقام الصحيح (تواريخ، نسب مئوية).
2. **جودة OCR منخفضة في قسم واحد فقط** (القسم الثامن - ضوابط التمويل العقاري):
   يظهر أن هذا الملف تحديداً مصدره مسح ضوئي أقدم بجودة أقل، وتبقى بعض الكلمات
   غير واضحة حتى بعد إصلاح ترتيب الحروف. تم وسم هذا المستند بـ
   `text_quality: "low_quality"` بدلاً من التظاهر بأنه نظيف، ويظهر تحذير عنه
   في واجهة Streamlit عند الاسترجاع منه.

## تشغيل الـ pipeline (بالترتيب)

**ملاحظة:** الخطوتين 04 و05 يحتاجان اتصال إنترنت لتحميل موديل الـ embeddings
من HuggingFace في أول مرة. شغليهم على Google Colab أو Lightning AI (أو جهازك
المحلي) بدل أي بيئة محدودة الشبكة، ثم ارفعي مجلد `chroma_store/` الناتج مع
باقي المشروع.

```bash
pip install -r requirements.txt

python 01_documents.py
python 02_preprocessing.py
python 03_chunking.py
python 04_vector_representation.py   # يحتاج إنترنت لتحميل الموديل
python 05_create_chroma_store.py     # ينشئ مجلد chroma_store/
```

بعد كده شغلي التطبيق محلياً للتجربة:

```bash
export OPENROUTER_API_KEY="your_key_here"
streamlit run streamlit_app.py
```

## النشر على Streamlit Cloud

1. ادفعي المشروع (بما فيه مجلد `chroma_store/` الجاهز) على GitHub - **بدون** ملف `.env` الحقيقي.
2. من Streamlit Cloud: New app -> اختاري الـ repo -> Deploy.
3. من صفحة التطبيق: Manage app -> Secrets -> أضيفي:

```toml
OPENROUTER_API_KEY = "your_openrouter_key_here"
OPENROUTER_MODEL = "openai/gpt-4o-mini"
```

## تقييم جودة الاسترجاع

`06_retrieve_context.py` فيه دوال جاهزة لحساب Precision@k وRecall@k وHit Rate
وMean Reciprocal Rank على مجموعة أسئلة اختبارية معنونة يدوياً (question + قائمة
chunk_ids الصحيحة) — مفيدة لقسم التقييم في تقرير المشروع.
