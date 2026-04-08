# plate_detector_module.py
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PlateDetectorModule:
    def __init__(self, model_path="license_plate_detector.pt", pad=0):
        # -------- device selection --------
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # تحميل النموذج
        self.model = YOLO(model_path)

        # نقل النموذج إلى GPU / CPU
        try:
            self.model.to(self.device)
        except Exception:
            pass

        self.pad = pad

        print(f"[PLATE-DETECTOR] Running on device: {self.device}")

    def detect(self, image):
        """
        Input:
            image = numpy array (BGR)
        Output:
            list of crops (numpy images)
            list of bounding boxes [(x1,y1,x2,y2), ...]
        """

        # inference
        results = self.model.predict(
            source=image,
            device=0 if self.device == "cuda" else "cpu",
            verbose=False
        )

        boxes_obj = results[0].boxes

        crops = []
        boxes = []

        if boxes_obj is None or len(boxes_obj) == 0:
            return crops, boxes

        # نقل النتائج إلى CPU مرة واحدة فقط
        xyxy = boxes_obj.xyxy.cpu().numpy().astype(int)

        for (x1, y1, x2, y2) in xyxy:
            # padding (آمن)
            x1 = max(0, x1 - self.pad)
            y1 = max(0, y1 - self.pad)
            x2 = min(image.shape[1], x2 + self.pad)
            y2 = min(image.shape[0], y2 + self.pad)

            crop = image[y1:y2, x1:x2]

            if crop.size == 0:
                continue

            crops.append(crop)
            boxes.append((x1, y1, x2, y2))

        return crops, boxes