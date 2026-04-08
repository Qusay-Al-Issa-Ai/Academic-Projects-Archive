# -*- coding: utf-8 -*-
import numpy as np
import cv2

# ثوابت افتراضية (يمكن تجاوزها في processor إن شئت)
VEHICLE_CLASSES_COCO = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
TRAFFIC_LIGHT_CLASS = 10
RESIZE_WIDTH = 1280
SKIP_FRAMES = 1
CONF_THRESH = 0.35
CONFIG_FILE = "camera_config.json"
EVENTS_DIR = "events"

def euclidean_distance(a, b):
    """
    دالة مسافة تستخدم مع Norfair wrapper.
    a, b: مصفوفات/نقاط numpy (على شكل [ [x, y] ]) أو كائنات متوافقة.
    نسعى للحفاظ على توافق مع التوقيع الأصلي.
    """
    try:
        pa = np.array(a).reshape(-1)
        pb = np.array(b).reshape(-1)
        return float(np.linalg.norm(pa - pb))
    except Exception:
        return 1e6

def centroid_from_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0], dtype=float)

def point_line_side(pt, a, b):
    (x, y) = pt
    (x1, y1), (x2, y2) = a, b
    return (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)

def crossed_line(prev_pt, curr_pt, line_a, line_b):
    side_prev = point_line_side(prev_pt, line_a, line_b)
    side_curr = point_line_side(curr_pt, line_a, line_b)
    return side_prev * side_curr < 0

def detect_traffic_light_color(img):
    """
    تحليل بسيط للون الإشارة داخل صورة مُقتطعة (ROI).
    يرجع ('red'|'green'|'yellow'|'unknown', confidence_float)
    """
    if img is None or img.size == 0:
        return 'unknown', 0.0
    try:
        h, w = img.shape[:2]
        if h < 6 or w < 6:
            return 'unknown', 0.0
        small = cv2.GaussianBlur(img, (5, 5), 0)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        lower_red1, upper_red1 = (0, 100, 50), (10, 255, 255)
        lower_red2, upper_red2 = (160, 100, 50), (180, 255, 255)
        lower_green, upper_green = (35, 60, 40), (90, 255, 255)
        lower_yellow, upper_yellow = (12, 100, 60), (35, 255, 255)

        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, np.array(lower_red1), np.array(upper_red1)),
            cv2.inRange(hsv, np.array(lower_red2), np.array(upper_red2))
        )
        mask_green = cv2.inRange(hsv, np.array(lower_green), np.array(upper_green))
        mask_yellow = cv2.inRange(hsv, np.array(lower_yellow), np.array(upper_yellow))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
        mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)

        def largest_component_area(mask):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            return max((cv2.contourArea(c) for c in contours), default=0)

        red_area = largest_component_area(mask_red)
        green_area = largest_component_area(mask_green)
        yellow_area = largest_component_area(mask_yellow)
        total_area = max(1, img.shape[0] * img.shape[1])

        scores = {
            'red': red_area / total_area,
            'green': green_area / total_area,
            'yellow': yellow_area / total_area
        }
        color = max(scores, key=scores.get)
        conf = scores[color]
        if conf < 0.005:
            return 'unknown', conf
        return color, float(conf)
    except Exception:
        return 'unknown', 0.0