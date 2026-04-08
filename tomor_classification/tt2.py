import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# تحميل النموذج المدرب
model = load_model('tumor_classifier_model.h5')  # تأكد من وجود ملف النموذج في نفس المجلد
classes = ['ورم حميد', 'ورم خبيث', 'ليست صورة ورم']  # تسميات الفئات

# دالة لتصنيف الصور
def classify_image(image):
    img = image.resize((224, 224))  # تغيير الحجم ليتناسب مع النموذج
    img_array = img_to_array(img) / 255.0  # تحويل الصورة إلى مصفوفة وتقسيم القيم
    img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد لتوافق الإدخال
    predictions = model.predict(img_array)  # التنبؤ باستخدام النموذج
    class_index = np.argmax(predictions)  # تحديد الفئة المتوقعة
    confidence = predictions[0][class_index] * 100  # نسبة الثقة
    return classes[class_index], confidence

# دالة لتصنيف الإطارات من الفيديو
def classify_frame(frame):
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  # تحويل الإطار إلى PIL Image
    return classify_image(img)

# واجهة المستخدم
st.title("تصنيف الأورام: صور وفيديوهات")
st.write("اختر نوع الإدخال لتحليل الصور أو الفيديوهات لتصنيف الأورام.")

# اختيار نوع الإدخال
input_type = st.radio("اختر نوع الإدخال:", ["صورة", "فيديو"])

if input_type == "صورة":
    # رفع صورة للتصنيف
    uploaded_image = st.file_uploader("ارفع صورة للتصنيف", type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        # فتح الصورة باستخدام PIL
        img = Image.open(uploaded_image)
        st.image(img, caption="الصورة المدخلة", use_container_width=True)

        # تصنيف الصورة
        result, confidence = classify_image(img)
        st.success(f"النتيجة: {result}")
        st.info(f"نسبة الثقة: {confidence:.2f}%")

elif input_type == "فيديو":
    # رفع فيديو لتحليل الإطارات
    uploaded_video = st.file_uploader("ارفع فيديو لتحليل الإطارات", type=['mp4', 'avi', 'mov'])

    if uploaded_video is not None:
        # قراءة الفيديو باستخدام OpenCV
        tfile = open("temp_video.mp4", "wb")  # حفظ الفيديو مؤقتاً
        tfile.write(uploaded_video.read())  # كتابة الفيديو المؤقت
        cap = cv2.VideoCapture("temp_video.mp4")

        if not cap.isOpened():
            st.error("تعذر فتح الفيديو. تأكد من الملف!")
        else:
            frame_count = 0
            stframe = st.empty()  # عنصر لعرض الإطارات

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    st.warning("انتهى الفيديو أو تعذر قراءة إطار.")
                    break

                frame_count += 1

                # تصنيف إطار واحد كل 10 إطارات (لتقليل الجهد)
                if frame_count % 200 == 0:
                    result, confidence = classify_frame(frame)
                    st.write(f"الإطار {frame_count}: {result} (الثقة: {confidence:.2f}%)")
                    frame = cv2.putText(frame, f"{result}: {confidence:.2f}%", (10, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

                # عرض الإطار
                stframe.image(frame, channels="BGR", use_container_width=True)

            cap.release()
            st.success("تم تحليل الفيديو بالكامل!")