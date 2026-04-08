# -*- coding: utf-8 -*-
import uuid
from collections import deque
from datetime import datetime
import numpy as np
from norfair import Detection

from .utils import (
    VEHICLE_CLASSES_COCO,
    TRAFFIC_LIGHT_CLASS,
    centroid_from_bbox,
    detect_traffic_light_color,
    crossed_line
)

class RedLightProcessor:
    def __init__(
        self,
        model,
        tracker,
        roi_manager,
        resize_width=960,
        skip_frames=1,
        conf_thresh=0.35,
        signal_buffer_len=7,
        pad_preview=6,
        event_queue=None,
        required_frames=5
    ):
        self.model = model
        self.tracker = tracker
        self.roi = roi_manager
        self.resize_width = resize_width
        self.skip_frames = skip_frames
        self.conf_thresh = conf_thresh

        self.signal_buffer = deque(maxlen=signal_buffer_len)
        self.track_centroids = {}
        self.reported_tracks = set()
        
        # مخزن لتجميع الفريمات (الحزم)
        self.track_packets = {} 
        self.required_frames = required_frames

        self.frame_idx = 0
        self.pad_preview = pad_preview
        self.event_queue = event_queue

    def process_frame(self, frame_proc):
        events = []
        self.frame_idx += 1

        # 1) كشف الأشياء في الفريم
        results = self.model.predict(frame_proc)
        vehicle_detections = []
        
        for (b, cls, sc) in results:
            x1, y1, x2, y2 = map(int, b)
            if (x2 - x1) * (y2 - y1) < 400: continue

            if int(cls) in VEHICLE_CLASSES_COCO:
                vehicle_detections.append(((x1, y1, x2, y2), float(sc), VEHICLE_CLASSES_COCO[int(cls)]))

        # 2) تحديد حالة الإشارة
        signal_state = {'state': 'unknown', 'confidence': 0.0}
        if self.roi.light_roi is not None:
            x1, y1, x2, y2 = self.roi.light_roi
            crop = frame_proc[y1:y2, x1:x2]
            color, conf = detect_traffic_light_color(crop)
            signal_state = {'state': color, 'confidence': float(conf)}

        self.signal_buffer.append(signal_state)
        states = [s['state'] for s in self.signal_buffer]
        majority = max(set(states), key=states.count)
        current_signal = majority

        # 3) التتبع (Tracking)
        norfair_dets = [Detection(points=centroid_from_bbox(d[0]).reshape(1, 2), scores=np.array([d[1]])) for d in vehicle_detections]
        tracked_objects = self.tracker.update(norfair_dets)

        # 4) معالجة كل كائن يتم تتبعه
        for tobj in tracked_objects:
            tid = getattr(tobj, 'id', None)
            est = getattr(tobj, 'estimate', None)
            if tid is None or est is None: continue

            cx, cy = int(est[0][0]), int(est[0][1])
            
            # البحث عن أفضل صندوق محيط (BBox) لهذا التتبع
            best_bbox = None
            min_dist = 1e12
            for b, sc, vtype in vehicle_detections:
                bx, by = (b[0] + b[2]) // 2, (b[1] + b[3]) // 2
                dist = (bx - cx)**2 + (by - cy)**2
                if dist < min_dist:
                    min_dist, best_bbox = dist, (b, sc, vtype)

            prev = self.track_centroids.get(tid)
            self.track_centroids[tid] = (cx, cy)

            # اكتشاف لحظة عبور الخط
            is_crossing = False
            if prev and self.roi.stop_line:
                is_crossing = crossed_line(prev, (cx, cy), *self.roi.stop_line)

            # إذا عبر السيارة الخط والإشارة حمراء، نفتح "حزمة" تجميع
            if is_crossing and current_signal == 'red' and tid not in self.reported_tracks:
                self.reported_tracks.add(tid)
                self.track_packets[tid] = {
                    "incident_id": str(uuid.uuid4()),
                    "previews": [],
                    "metadata": {
                        "track_id": int(tid),
                        "vehicle_type": best_bbox[2] if best_bbox else "unknown",
                        "timestamp": datetime.utcnow().isoformat() + "Z",
                        "signal_state": current_signal
                    }
                }

            # إذا كانت السيارة تحت المراقبة (مخالفة)، نجمع لها الفريمات
            if tid in self.track_packets:
                packet = self.track_packets[tid]
                
                if best_bbox and len(packet["previews"]) < self.required_frames:
                    b, sc, vtype = best_bbox
                    pad = self.pad_preview
                    x1, y1, x2, y2 = b
                    x1c, y1c = max(0, x1-pad), max(0, y1-pad)
                    x2c, y2c = min(frame_proc.shape[1], x2+pad), min(frame_proc.shape[0], y2+pad)
                    
                    # نأخذ لقطة للسيارة
                    packet["previews"].append(frame_proc[y1c:y2c, x1c:x2c].copy())

                # عند اكتمال الـ 5 فريمات، نرسل الحدث
                if len(packet["previews"]) >= self.required_frames:
                    final_event = {
                        "incident_id": packet["incident_id"],
                        "track_id": packet["metadata"]["track_id"],
                        "vehicle_type": packet["metadata"]["vehicle_type"],
                        "timestamp": packet["metadata"]["timestamp"],
                        "signal_state": {"state": packet["metadata"]["signal_state"]},
                        "preview_images": packet["previews"], # قائمة المصفوفات
                        "is_batch": True
                    }
                    
                    if self.event_queue:
                        # نستخدم put_nowait لضمان عدم تعليق النظام الرئيسي
                        self.event_queue.put_nowait(final_event)
                    
                    print(f"[PROCESSOR] Sent Batch Event: Track {tid} with {len(packet['previews'])} frames.")
                    del self.track_packets[tid] # تنظيف الذاكرة فوراً

        return frame_proc.copy(), events

