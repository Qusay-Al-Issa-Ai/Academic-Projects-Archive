# -*- coding: utf-8 -*-
import threading
import os
import uuid
import json
import cv2

class BackgroundSaver(threading.Thread):
    def __init__(self, q, pipeline=None, events_dir="events"):
        """
        أضفنا pipeline لكي يتمكن الـ Saver من استدعاء منطق الحسم (Consensus)
        """
        super().__init__(daemon=True)
        self.q = q
        self.pipeline = pipeline
        self.events_dir = events_dir
        os.makedirs(self.events_dir, exist_ok=True)

    def run(self):
        while True:
            item = self.q.get()
            if item is None:
                self.q.task_done()
                break

            try:
                incident_id = item.get('incident_id', str(uuid.uuid4()))
                event_folder = os.path.join(self.events_dir, incident_id)
                os.makedirs(event_folder, exist_ok=True)

                # ===== 1) استلام الصور =====
                batch_images = item.get('preview_images', [])
                if not batch_images and 'preview_image' in item:
                    batch_images = [item['preview_image']]

                # ===== 2) تشغيل الـ Pipeline للحسم =====
                final_text = "N/A"
                plate_crop_path = None
                plate_enhanced_path = None

                if self.pipeline and batch_images:
                    results = self.pipeline.run(batch_images, event_folder=event_folder)
                    if results:
                        res = results[0]  # النتيجة الحاسمة بعد التصويت
                        final_text = res['text']

                        plate_crop_path = res.get('plate_crop_path')
                        plate_enhanced_path = res.get('plate_enhanced_path')

                # ===== 3) حفظ الصورة الأصلية للمركبة =====
                vehicle_image_path = None
                if batch_images:
                    vehicle_image_path = os.path.join(event_folder, "vehicle.jpg")
                    cv2.imwrite(vehicle_image_path, batch_images[0])

                # ===== 4) تنظيف البيانات وكتابة JSON =====
                meta = item.copy()
                meta.pop('preview_images', None)
                meta.pop('preview_image', None)

                meta['final_plate_text'] = final_text
                meta['event_folder'] = event_folder
                meta['vehicle_image_path'] = vehicle_image_path
                meta['plate_crop_path'] = plate_crop_path
                meta['plate_enhanced_path'] = plate_enhanced_path

                meta_fname = os.path.join(event_folder, "metadata.json")
                with open(meta_fname, 'w', encoding='utf-8') as mf:
                    json.dump(meta, mf, indent=2, ensure_ascii=False)

                print(f"[BACKGROUND] Processed incident {incident_id} | Result: {final_text}")

            except Exception as e:
                print(f"[BACKGROUND] Error in saver: {e}")

            finally:
                self.q.task_done()