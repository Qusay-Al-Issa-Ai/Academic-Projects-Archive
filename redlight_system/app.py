# -*- coding: utf-8 -*-
"""
تحسين شامل لنظام التقاط مخالفات الإشارات الحمراء مع حفظ الصور والبيانات في قاعدة بيانات
تم التعديل: إضافة شرط التحقق من النص قبل الحفظ في قاعدة البيانات
"""

import os
import argparse
import queue
import cv2
import uuid
import json
import threading
import sqlite3
import time
import logging
from datetime import datetime

# استورد ما تحتاجه من مشروعك
# (تأكد أن هذه الملفات موجودة في مسارك كما في الكود الأصلي)
from redlight.model_loader import load_yolo
from redlight.detector import YoloDetector
from redlight.tracker_module import TrackerWrapper
from redlight.utils import (
    euclidean_distance, RESIZE_WIDTH, SKIP_FRAMES, CONF_THRESH,
    CONFIG_FILE, EVENTS_DIR
)
from redlight.roi_manager import ROISelector, ROIManager
from redlight.processor import RedLightProcessor
from plate_pipeline import PlatePipeline

# ---------- إعداد logging ----------
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# ---------- EventFanout آمن ----------
class EventFanout:
    def __init__(self, *queues, put_timeout=0.5):
        self.queues = queues
        self.put_timeout = put_timeout

    def put(self, item):
        for q in self.queues:
            try:
                q.put(item, block=True, timeout=self.put_timeout)
            except queue.Full:
                logger.warning("EventFanout: queue full; dropped event for one queue")

    def put_nowait(self, item):
        ok = True
        for q in self.queues:
            try:
                q.put_nowait(item)
            except queue.Full:
                logger.warning("EventFanout.put_nowait: queue full; dropped event for one queue")
                ok = False
        return ok

# ---------- BackgroundSaver (تم التعديل هنا) ----------
class BackgroundSaver(threading.Thread):
    def __init__(self, in_queue, events_dir=EVENTS_DIR, db_path="violations.db"):
        super().__init__(daemon=True)
        self.q = in_queue
        self.events_dir = events_dir
        self.db_path = db_path
        os.makedirs(self.events_dir, exist_ok=True)
        self._ensure_db()

    def _ensure_db(self):
        try:
            with sqlite3.connect(self.db_path, timeout=30) as conn:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA foreign_keys = ON;")
                conn.execute("""CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    incident_id TEXT UNIQUE,
                    vehicle_image_path TEXT,
                    plate_crop_path TEXT,
                    plate_enhanced_path TEXT,
                    plate_text TEXT,
                    confidence TEXT,
                    event_folder TEXT,
                    timestamp TEXT
                )""")
                conn.commit()
        except Exception as e:
            logger.exception("Failed to initialize DB: %s", e)

    # دالة الحفظ الجديدة المشروطة
    def _db_save_if_valid(self, meta_data):
        plate_text = meta_data.get("plate_text")
        incident_id = meta_data.get("incident_id")

        # --- الشرط المطلوب ---
        # إذا كان النص غير موجود (None) أو فارغاً أو يساوي "unknown" (بأي حالة أحرف)
        if not plate_text or str(plate_text).strip().lower() == "unknown" or str(plate_text).strip() == "":
            logger.info("DB Skipped: Text is '%s' for incident %s", plate_text, incident_id)
            return  # لا تفعل شيئاً، اخرج من الدالة

        # إذا تحقق الشرط (النص موجود وصحيح)، نقوم بالحفظ
        tries = 3
        while tries:
            try:
                with sqlite3.connect(self.db_path, timeout=30) as conn:
                    conn.execute("PRAGMA busy_timeout = 5000;")
                    cursor = conn.cursor()
                    # نستخدم INSERT OR REPLACE لضمان عدم تكرار البيانات
                    cursor.execute(
                        """INSERT OR REPLACE INTO violations (
                            incident_id, vehicle_image_path, plate_crop_path, 
                            plate_enhanced_path, plate_text, confidence, 
                            event_folder, timestamp
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            incident_id,
                            meta_data.get("vehicle_image_path"),
                            meta_data.get("plate_crop_path"),
                            meta_data.get("plate_enhanced_path"),
                            plate_text,
                            meta_data.get("confidence"),
                            meta_data.get("event_folder"),
                            meta_data.get("timestamp")
                        )
                    )
                    conn.commit()
                logger.info("DB Saved: Valid plate '%s' for incident %s", plate_text, incident_id)
                return
            except sqlite3.OperationalError as e:
                tries -= 1
                logger.warning("DB insert error: %s — retries left %s", e, tries)
                time.sleep(0.3)
        logger.error("DB insert failed for %s", incident_id)

    def run(self):
        while True:
            item = self.q.get()
            try:
                if item is None:
                    logger.info("BackgroundSaver received sentinel, exiting")
                    break

                kind = item.get("_kind", "incident")
                incident_id = item.get("incident_id", str(uuid.uuid4()))
                event_folder = os.path.join(self.events_dir, incident_id)
                os.makedirs(event_folder, exist_ok=True)

                if kind == "incident":
                    # --- المرحلة الأولى: حفظ الملفات فقط (بدون قاعدة بيانات) ---
                    previews = item.get("preview_images", [])
                    vehicle_path = None
                    if previews:
                        vehicle_path = os.path.join(event_folder, "vehicle.jpg")
                        try:
                            cv2.imwrite(vehicle_path, previews[0])
                        except Exception:
                            logger.exception("Failed to write vehicle image for %s", incident_id)
                            vehicle_path = None

                        meta = item.copy()
                        meta.pop('preview_images', None)
                        meta.update({
                            "vehicle_image_path": vehicle_path,
                            "plate_crop_path": None,
                            "plate_enhanced_path": None,
                            "plate_text": None,
                            "confidence": None,
                            "event_folder": event_folder,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        })
                        try:
                            with open(os.path.join(event_folder, "metadata.json"), "w", encoding="utf-8") as f:
                                json.dump(meta, f, indent=2, ensure_ascii=False)
                        except Exception:
                            logger.exception("Failed to write metadata.json for %s", incident_id)

                        # ملاحظة: تم إزالة _db_insert_initial من هنا لتعليق الحفظ في القاعدة

                    else:
                        logger.debug("BackgroundSaver: incident with no previews, skipping vehicle save")

                elif kind == "plate":
                    # --- المرحلة الثانية: التحقق والحفظ في القاعدة ---
                    plate_crop = item.get("plate_crop_path")
                    plate_enhanced = item.get("plate_enhanced_path")
                    plate_text = item.get("plate_text")
                    confidence = item.get("confidence")

                    meta_path = os.path.join(event_folder, "metadata.json")
                    meta = {}
                    
                    # تحميل البيانات القديمة (لنحصل على التوقيت ومسار السيارة)
                    try:
                        if os.path.exists(meta_path):
                            with open(meta_path, "r", encoding="utf-8") as f:
                                meta = json.load(f)
                    except Exception:
                        pass # إذا فشل التحميل، نعتمد على البيانات الجديدة فقط

                    # تحديث البيانات
                    meta.update({
                        "incident_id": incident_id,
                        "plate_crop_path": plate_crop,
                        "plate_enhanced_path": plate_enhanced,
                        "plate_text": plate_text,
                        "confidence": confidence
                    })

                    # تحديث ملف JSON (يبقى دائماً كسجل احتياطي)
                    try:
                        with open(meta_path, "w", encoding="utf-8") as f:
                            json.dump(meta, f, indent=2, ensure_ascii=False)
                    except Exception:
                        logger.exception("Failed to update metadata.json for %s", incident_id)

                    # محاولة الحفظ في قاعدة البيانات (سيتم تطبيق الشرط هنا)
                    self._db_save_if_valid(meta)
                
                else:
                    logger.warning("Unknown item kind in BackgroundSaver: %s", kind)

            except Exception:
                logger.exception("Unexpected error in BackgroundSaver loop")
            finally:
                try:
                    self.q.task_done()
                except Exception:
                    logger.exception("task_done error in BackgroundSaver")

# ---------- PipelineWorker ----------
class PipelineWorker(threading.Thread):
    def __init__(self, pipeline, in_queue, out_queue, events_dir=EVENTS_DIR):
        super().__init__(daemon=True)
        self.pipeline = pipeline
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.events_dir = events_dir

    def run(self):
        while True:
            item = self.in_queue.get()
            try:
                if item is None:
                    logger.info("PipelineWorker received sentinel, exiting")
                    break

                incident_id = item.get("incident_id", str(uuid.uuid4()))
                previews = item.get("preview_images", [])
                if not previews:
                    logger.debug("PipelineWorker: no previews for %s", incident_id)
                    continue

                try:
                    results = self.pipeline.run(previews, event_folder=os.path.join(self.events_dir, incident_id))
                except Exception:
                    logger.exception("Pipeline run failed for %s", incident_id)
                    results = None

                if results:
                    res = results[0]
                    event_folder = os.path.join(self.events_dir, incident_id)
                    plates_dir = os.path.join(event_folder, "plates_pipeline")
                    os.makedirs(plates_dir, exist_ok=True)

                    plate_crop_path = None
                    plate_enhanced_path = None
                    try:
                        if res.get("crop") is not None:
                            plate_crop_path = os.path.join(plates_dir, "crop.png")
                            cv2.imwrite(plate_crop_path, res["crop"])
                        if res.get("enhanced") is not None:
                            plate_enhanced_path = os.path.join(plates_dir, "enhanced.png")
                            cv2.imwrite(plate_enhanced_path, res["enhanced"])
                    except Exception:
                        logger.exception("Failed to write plate images for %s", incident_id)

                    plate_text = res.get("text") if res.get("text") else None
                    confidence = res.get("confidence_score")

                    try:
                        with open(os.path.join(event_folder, "plate_result.json"), "w", encoding="utf-8") as f:
                            json.dump({
                                "incident_id": incident_id,
                                "plate_text": plate_text,
                                "confidence": confidence
                            }, f, ensure_ascii=False, indent=2)
                    except Exception:
                        logger.exception("Failed to write plate_result.json for %s", incident_id)

                    out_item = {
                        "_kind": "plate",
                        "incident_id": incident_id,
                        "plate_crop_path": plate_crop_path,
                        "plate_enhanced_path": plate_enhanced_path,
                        "plate_text": plate_text,
                        "confidence": confidence
                    }
                    try:
                        self.out_queue.put(out_item, block=True, timeout=1.0)
                    except queue.Full:
                        logger.warning("BackgroundSaver queue full; dropping plate update for %s", incident_id)
                else:
                    logger.info("PipelineWorker: no plate results for %s", incident_id)

            except Exception:
                logger.exception("Unexpected error in PipelineWorker loop")
            finally:
                try:
                    self.in_queue.task_done()
                except Exception:
                    logger.exception("task_done error in PipelineWorker")

# ---------- Main ----------
def main(source=0):
    logger.info("Starting improved redlight system...")

    model = load_yolo("yolov8s.pt")
    detector = YoloDetector(model, resize_width=RESIZE_WIDTH, conf_thresh=CONF_THRESH)
    tracker = TrackerWrapper(distance_function=euclidean_distance, distance_threshold=40)

    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        logger.error("Cannot open source: %s", source)
        return

    selector = ROISelector("Core Processing")
    roi_manager = ROIManager()
    if os.path.exists(CONFIG_FILE):
        roi_manager.load_config(CONFIG_FILE)
        selector.stop_line = roi_manager.stop_line
        selector.light_roi = roi_manager.light_roi

    cv2.namedWindow(selector.window_name, cv2.WINDOW_NORMAL)
    ret, first_frame = cap.read()
    if not ret:
        logger.error("Failed to read first frame")
        return

    h0, w0 = first_frame.shape[:2]
    scale = RESIZE_WIDTH / float(w0)
    first_preview = cv2.resize(first_frame, (RESIZE_WIDTH, int(h0 * scale)), interpolation=cv2.INTER_LANCZOS4)
    cv2.setMouseCallback(selector.window_name, selector.mouse_callback, param=first_preview)

    while True:
        preview_copy = first_preview.copy()
        selector.draw_on(preview_copy)
        cv2.imshow(selector.window_name, preview_copy)
        key = cv2.waitKey(10) & 0xFF
        if key == ord('q') and selector.stop_line and selector.light_roi:
            roi_manager.stop_line = selector.stop_line
            roi_manager.light_roi = selector.light_roi
            break
        elif key == ord('s'):
            selector.set_mode('stopline')
        elif key == ord('l'):
            selector.set_mode('light')
        elif key == ord('w'):
            cfg = {
                'stop_line': [list(pt) for pt in selector.stop_line],
                'light_roi': list(selector.light_roi)
            }
            with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
                json.dump(cfg, f)

    bg_queue = queue.Queue(maxsize=1000)
    pipeline_queue = queue.Queue(maxsize=200)
    plate_pipeline = PlatePipeline()
    fanout = EventFanout(pipeline_queue, bg_queue)

    bg_saver = BackgroundSaver(bg_queue, events_dir=EVENTS_DIR, db_path=os.path.join(EVENTS_DIR, 'violations.db'))
    bg_saver.start()
    pw = PipelineWorker(plate_pipeline, pipeline_queue, bg_queue, events_dir=EVENTS_DIR)
    pw.start()
    processor = RedLightProcessor(detector, tracker, roi_manager, event_queue=fanout, required_frames=10)

    frame_idx = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.info("End of stream or cannot read frame")
                break

            frame_idx += 1
            if frame_idx % SKIP_FRAMES != 0:
                continue

            frame_proc = cv2.resize(frame, (RESIZE_WIDTH, int(frame.shape[0] * (RESIZE_WIDTH / frame.shape[1]))), interpolation=cv2.INTER_LANCZOS4)
            display_frame, _ = processor.process_frame(frame_proc)
            selector.draw_on(display_frame)
            cv2.imshow(selector.window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested quit")
                break

    except KeyboardInterrupt:
        logger.info("KeyboardInterrupt received, shutting down")
    finally:
        cap.release()
        cv2.destroyAllWindows()

        try:
            pipeline_queue.put(None, block=True, timeout=1.0)
        except Exception:
            logger.exception("Failed to put sentinel to pipeline_queue")
        try:
            bg_queue.put(None, block=True, timeout=1.0)
        except Exception:
            logger.exception("Failed to put sentinel to bg_queue")

        pw.join(timeout=5.0)
        bg_saver.join(timeout=5.0)
        logger.info("Shutdown complete")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # تعديل: جعل المسار يستقبل قيمة من الخارج
    parser.add_argument("--source", type=str, default="0") # 0 تعني الكاميرا الافتراضية
    args = parser.parse_args()
    
    # تحويل القيمة إلى رقم إذا كانت كاميرا، أو تركها كمسار إذا كانت فيديو
    source_input = args.source
    if source_input.isdigit():
        source_input = int(source_input)
        
    main(source=source_input)
