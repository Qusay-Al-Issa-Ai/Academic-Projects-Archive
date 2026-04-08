# -*- coding: utf-8 -*-
import cv2
import os
from collections import Counter
from plate_detector_module import PlateDetectorModule
from plate_enhancer_module import PlateEnhancerModule
from plate_ocr_module import PlateOCRModule

class PlatePipeline:
    def __init__(
        self,
        detector_model="license_plate_detector.pt",
        enhancer_model="experiments/pretrained_models/RealESRGAN_x2plus.pth"
    ):
        self.detector = PlateDetectorModule(detector_model)
        self.enhancer = PlateEnhancerModule(model_path=enhancer_model)
        self.ocr = PlateOCRModule()

    def run(self, input_data, event_folder=None):
        """
        input_data: يمكن أن يكون صورة واحدة (BGR) أو قائمة من الصور (List of BGR)
        event_folder: مسار حفظ الصور الناتجة (سيتم تمريره من PipelineWorker)
        """
        if not isinstance(input_data, list):
            images = [input_data]
        else:
            images = input_data

        all_frames_texts = []
        final_results = []

        best_crop = None
        best_enhanced = None

        for img in images:
            crops, boxes = self.detector.detect(img)

            for crop in crops:
                enhanced = self.enhancer.enhance(crop)
                text = self.ocr.read(enhanced).strip()

                if text:
                    all_frames_texts.append(text)
                    if best_crop is None:
                        best_crop = crop
                        best_enhanced = enhanced

        if not all_frames_texts:
            return []

        counts = Counter(all_frames_texts)
        final_text, frequency = counts.most_common(1)[0]

        print(f"[PIPELINE] Consensus: {final_text} found in {frequency}/{len(images)} frames.")

        # ===== حفظ الصور المهمة في event_folder =====
        plate_crop_path = None
        plate_enhanced_path = None
        if event_folder:
            plates_dir = os.path.join(event_folder, "plates_pipeline")
            os.makedirs(plates_dir, exist_ok=True)

            if best_crop is not None:
                plate_crop_path = os.path.join(plates_dir, "crop.png")
                cv2.imwrite(plate_crop_path, best_crop)

            if best_enhanced is not None:
                plate_enhanced_path = os.path.join(plates_dir, "enhanced.png")
                cv2.imwrite(plate_enhanced_path, best_enhanced)

        final_results.append({
            "crop": best_crop,
            "enhanced": best_enhanced,
            "text": final_text,
            "confidence_score": f"{frequency}/{len(images)}",
            "all_candidates": counts,
            "plate_crop_path": plate_crop_path,
            "plate_enhanced_path": plate_enhanced_path
        })

        return final_results