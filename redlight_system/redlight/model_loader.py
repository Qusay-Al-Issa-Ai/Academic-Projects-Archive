# -*- coding: utf-8 -*-
# تحميل نموذج YOLO (ultralytics)
from ultralytics import YOLO

def load_yolo(model_path: str):
    """
    يعيد كائن YOLO الجاهز للاستخدام.
    model_path: مسار ملف النموذج (مثال: "yolov8s.pt")
    """
    return YOLO(model_path)