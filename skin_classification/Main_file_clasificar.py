import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ===============================
# مسار البيانات
# ===============================

data_dir = r'C:\Users\C E C\Desktop\skin_clasification2\Split_smol\train'

# ===============================
# تجهيز البيانات
# ===============================

data_gen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=15,
    horizontal_flip=True,
    zoom_range=0.1
)

train_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='training'
)

val_data = data_gen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    batch_size=8,
    class_mode='categorical',
    subset='validation'
)

num_classes = train_data.num_classes
print("عدد الفئات:", num_classes)
print("ترميز الفئات:", train_data.class_indices)

# ===============================
# بناء النموذج
# ===============================

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ===============================
# تدريب النموذج
# ===============================

epochs =30

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=epochs
)

# ===============================
# حفظ النموذج
# ===============================

model.save('skin_disease_classifier_6_classes.h5')

# ===============================
# رسم النتائج
# ===============================

def plot_metrics(history):
    # المخطط الأول: الخسارة (Loss)
    plt.figure(figsize=(5, 4))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    # تصغير حجم الخط في العنوان والتسميات
    plt.title('Loss', fontsize=10)
    plt.legend(fontsize=8)
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Loss', fontsize=8)
    plt.show()

    # المخطط الثاني: الدقة (Accuracy)
    plt.figure(figsize=(5, 4))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    # تصغير حجم الخط في العنوان والتسميات
    plt.title('Accuracy', fontsize=10)
    plt.legend(fontsize=8)
    plt.xlabel('Epochs', fontsize=8)
    plt.ylabel('Accuracy', fontsize=8)
    plt.show()

# استدعاء الدالة
plot_metrics(history)







