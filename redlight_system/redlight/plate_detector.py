import cv2
from ultralytics import YOLO

class PlateDetector:
    def __init__(self, model_path, conf_thresh=0.5, pad=5):
        """
        model_path: مسار نموذج كشف اللوحة license_plate_detector.pt
        conf_thresh: عتبة الثقة للكشف
        pad: هامش قص حول اللوحة
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.pad = pad

    def detect_plate(self, image):
        """
        image: مصفوفة صورة (BGR)
        return:
            - crop: صورة اللوحة المقصوصة (أو None)
            - bbox: الإحداثيات [x1, y1, x2, y2] (أو None)
            - conf: الثقة (float) (أو None)
        """

        if image is None:
            return None, None, None

        results = self.model.predict(image, conf=self.conf_thresh, verbose=False)
        boxes = results[0].boxes

        if len(boxes) == 0:
            return None, None, None

        # أخذ أعلى صندوق ثقة
        best_box = max(boxes, key=lambda b: float(b.conf[0].item()))

        x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy().astype(int)
        conf = float(best_box.conf[0].item())

        # إضافة هامش قص
        h, w = image.shape[:2]
        pad = self.pad

        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = image[y1:y2, x1:x2]
        bbox = [int(x1), int(y1), int(x2), int(y2)]

        return crop, bbox, conf