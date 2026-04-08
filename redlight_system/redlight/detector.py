# -*- coding: utf-8 -*-
from typing import List, Tuple
import numpy as np
import torch

class YoloDetector:
    """
    غلاف بسيط لـ YOLO (ultralytics).
    يحافظ على طريقة الاستخدام كما في الكود الأصلي.
    """

    def __init__(self, model, resize_width=960, conf_thresh=0.35):
        # -------- device selection --------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # النموذج قادم جاهز من load_yolo
        self.model = model
        try:
            # نقل النموذج إلى GPU / CPU
            self.model.to(self.device)
        except Exception:
            # في حال كان model ليس كائن torch مباشر (احتياط)
            pass

        self.resize_width = resize_width
        self.conf_thresh = conf_thresh

        print(f"[YOLO-DETECTOR] Running on device: {self.device}")

    def predict(self, frame):
        """
        frame: numpy array (frame_proc resized already عادةً)
        يعيد list من (bbox, cls, conf)
        bbox = (x1,y1,x2,y2) كأعداد صحيحة
        """

        # لا نغيّر أي شيء في منطق الاستدعاء
        results = self.model.predict(
            source=frame,
            imgsz=self.resize_width,
            conf=self.conf_thresh,
            device=0 if self.device == "cuda" else "cpu",
            verbose=False
        )

        res = results[0]
        dets = []

        if hasattr(res, 'boxes') and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy().astype(int)
            classes = res.boxes.cls.cpu().numpy().astype(int)
            scores = res.boxes.conf.cpu().numpy().astype(float)

            for (b, cls, sc) in zip(boxes, classes, scores):
                x1, y1, x2, y2 = int(b[0]), int(b[1]), int(b[2]), int(b[3])
                dets.append(((x1, y1, x2, y2), int(cls), float(sc)))

        return dets