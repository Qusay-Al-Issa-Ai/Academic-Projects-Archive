import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pyttsx3
from PIL import Image
import os

# ===============================
# إعدادات الصفحة (يجب أن تكون أول أمر)
# ===============================
st.set_page_config(
    page_title="AI Skin Diagnostics",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# إضافة لمسات CSS لتحسين المظهر
# ===============================
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #007bff;
        color: white;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# تحميل النموذج (استخدام التخزين المؤقت للسرعة)
# ===============================
@st.cache_resource
def load_skin_model():
    return load_model('skin_disease_classifier_6_classes.h5')

model = load_skin_model()

classes = [
    'Actinic keratosis',
    'Benign keratosis',
    'Melanoma'
]

# ===============================
# الدوال البرمجية
# ===============================
def classify_image(img):
    # تحويل الصورة مباشرة دون الحاجة لحفظها كملف مؤقت
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index] * 100

    return classes[class_index], confidence

def speak_result(text):
    # ملاحظة: pyttsx3 قد يواجه صعوبة في بيئات السيرفر (Cloud) 
    # ولكنه سيعمل بشكل ممتاز على جهازك الشخصي
    try:
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    except:
        pass

# ===============================
# تصميم الواجهة (Layout)
# ===============================

# --- الشريط الجانبي ---
with st.sidebar:
    st.image("image1.jpg", width=100)
    st.title("معلومات النظام")
    st.info("هذا النظام يستخدم تقنيات التعلم العميق (CNN) لتحليل الصور الجلدية المرفوعة.")
    st.warning("⚠️ تنبيه: هذا التطبيق لأغراض تعليمية فقط ولا يغني عن استشارة الطبيب المختص.")

# --- الصفحة الرئيسية ---
st.title("🩺 نظام تشخيص أمراض الجلد الذكي")
st.write("قم برفع صورة واضحة للمنطقة المصابة للحصول على تحليل فوري.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 تحميل الصورة")
    uploaded_file = st.file_uploader(
        "اختر صورة (JPG, PNG, JPEG)",
        type=['jpg', 'png', 'jpeg'],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption='الصورة التي تم رفعها', use_container_width=True)

with col2:
    st.subheader("📊 نتيجة التحليل")
    
    if uploaded_file is not None:
        with st.spinner('جاري التحليل...'):
            result, confidence = classify_image(img)
            
            # عرض النتائج بشكل جذاب
            st.markdown(f"""
                <div class="result-card">
                    <h3 style='color: #007bff;'>التشخيص المحتمل:</h3>
                    <h2 style='text-align: center;'>{result}</h2>
                    <hr>
                    <p>نسبة الثقة: <b>{confidence:.2f}%</b></p>
                </div>
            """, unsafe_allow_html=True)
            
            # شريط التقدم للثقة
            st.progress(int(confidence))

            # زر النطق
            if st.button("🔊 استماع للنتيجة"):
                speak_result(f"{result} with {confidence:.2f} percent confidence")
    else:
        st.info("انتظار رفع الصورة لبدء التحليل...")

# --- التذييل ---
st.markdown("---")
st.caption("تطوير : حسن العصفور © 2025")