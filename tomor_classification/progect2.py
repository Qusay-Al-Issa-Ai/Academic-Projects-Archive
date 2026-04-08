import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# تحميل ومعالجة البيانات
data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_data = data_gen.flow_from_directory(
    'C:\\Users\\C E C\\Desktop\\قصي قصي\\data4',  # ضع مسار البيانات هنا
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    'C:\\Users\\C E C\\Desktop\\قصي قصي\\data4',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(3, activation='softmax')  # ثلاث فئات: ورم حميد، ورم خبيث، وصورة ليست ورم
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# تدريب النموذج
history = model.fit(train_data, validation_data=val_data, epochs=10)

# حفظ النموذج
model.save('tumor_classifier_model.h5')


import matplotlib.pyplot as plt

# رسم الخسارة والدقة
def plot_metrics(history):
    plt.figure(figsize=(12, 5))

    # خسارة التدريب والتحقق
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='خسارة التدريب')
    plt.plot(history.history['val_loss'], label='خسارة التحقق')
    plt.legend()
    plt.title('الخسارة عبر الحقب')

    # دقة التدريب والتحقق
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='دقة التدريب')
    plt.plot(history.history['val_accuracy'], label='دقة التحقق')
    plt.legend()
    plt.title('الدقة عبر الحقب')

    plt.show()

plot_metrics(history)


import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pyttsx3
from PIL import Image

# تحميل النموذج
model = load_model('tumor_classifier_model.h5')
classes = ['ورم حميد', 'ورم خبيث', 'ليست صورة ورم']

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    predictions = model.predict(img_array)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index] * 100
    return classes[class_index], confidence

def speak_result(result_text):
    engine = pyttsx3.init()
    engine.say(result_text)
    engine.runAndWait()

# واجهة المستخدم
st.title("تصنيف الأورام")
uploaded_file = st.file_uploader("ارفع صورة للتصنيف", type=['jpg', 'png'])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='الصورة المدخلة', use_column_width=True)

    # حفظ الصورة مؤقتاً
    img.save("temp.jpg")

    # تصنيف الصورة
    result, confidence = classify_image("temp.jpg")
    st.write(f"النتيجة: {result} بنسبة {confidence:.2f}%")
    
    # نطق النتيجة
    speak_result(f"{result} بنسبة {confidence:.2f} بالمئة")