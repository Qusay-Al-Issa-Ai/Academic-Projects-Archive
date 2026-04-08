# -*- coding: utf-8 -*-
import json
import cv2

class ROIManager:
    """
    واجهة لإدارة stop_line و light_roi بدون UI افتراضي.
    """
    def __init__(self):
        self.stop_line = None
        self.light_roi = None

    def set_stop_line(self, p1, p2):
        self.stop_line = (tuple(p1), tuple(p2))

    def set_light_roi(self, roi):
        # roi = (x1,y1,x2,y2)
        self.light_roi = tuple(roi)

    def load_config(self, path="camera_config.json"):
        try:
            with open(path, 'r') as f:
                cfg = json.load(f)
                if 'stop_line' in cfg:
                    self.stop_line = tuple([tuple(pt) for pt in cfg['stop_line']])
                if 'light_roi' in cfg:
                    self.light_roi = tuple(cfg['light_roi'])
                return True
        except Exception:
            return False

    def save_config(self, path="camera_config.json"):
        cfg = {}
        if self.stop_line is not None:
            cfg['stop_line'] = [list(self.stop_line[0]), list(self.stop_line[1])]
        if self.light_roi is not None:
            cfg['light_roi'] = list(self.light_roi)
        with open(path, 'w') as f:
            json.dump(cfg, f)

class ROISelector:
    """
    نسخة UI من selector (تتعامل مع mouse callbacks و draw_on).
    تماثل تمامًا وظيفة الـ UI في الكود الأصلي.
    """
    def __init__(self, window_name):
        self.window_name = window_name
        self.reset()
        self.mode = None
        self.temp_pt = None

    def reset(self):
        self.stop_line = None
        self.light_roi = None

    def set_mode(self, mode):
        self.mode = mode
        print(f"[UI] Mode set to: {mode}. Use mouse to draw/select. Press same key to exit mode.")

    def mouse_callback(self, event, x, y, flags, param):
        frame = param
        if self.mode == 'stopline':
            if event == cv2.EVENT_LBUTTONDOWN:
                if self.temp_pt is None:
                    self.temp_pt = (x, y)
                    print(f"[UI] Stop line first point: {self.temp_pt}")
                else:
                    p1 = self.temp_pt
                    p2 = (x, y)
                    self.stop_line = (p1, p2)
                    print(f"[UI] Stop line set: {self.stop_line}")
                    self.temp_pt = None
        elif self.mode == 'light':
            if event == cv2.EVENT_LBUTTONDOWN:
                self.temp_pt = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.temp_pt is not None:
                x1, y1 = self.temp_pt
                x2, y2 = (x, y)
                x1c, x2c = min(x1, x2), max(x1, x2)
                y1c, y2c = min(y1, y2), max(y1, y2)
                self.light_roi = (x1c, y1c, x2c, y2c)
                self.temp_pt = None
                print(f"[UI] Light ROI set: {self.light_roi}")

    def draw_on(self, img):
        if self.stop_line is not None:
            cv2.line(img, self.stop_line[0], self.stop_line[1], (0, 0, 255), 2)
        if self.light_roi is not None:
            x1, y1, x2, y2 = self.light_roi
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)
        if self.temp_pt is not None and self.mode == 'stopline':
            cv2.circle(img, self.temp_pt, 4, (255, 0, 0), -1)