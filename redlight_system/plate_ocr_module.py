# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from paddleocr import PaddleOCR
import easyocr
import re

class PlateOCRModule:
    """
    High-accuracy OCR for license plates
    - PaddleOCR + EasyOCR
    - GPU supported
    - Adaptive preprocessing
    - Strict numeric plate formatting
    """

    def __init__(self):
        self.use_cuda = torch.cuda.is_available()
        self.paddle = PaddleOCR(
            use_angle_cls=True,
            lang='en',
            det=False,
            rec=True,
            use_gpu=self.use_cuda,
            show_log=False
        )
        self.easy = easyocr.Reader(
            ["en"],
            gpu=self.use_cuda
        )
        device = "cuda" if self.use_cuda else "cpu"
        print(f"[PLATE-OCR] Running on: {device}")

    # -------------------------------------------------
    # Preprocessing variants (lightweight & Morphology)
    # -------------------------------------------------
    def _preprocess_variants(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variants = []

        # 1) raw gray
        variants.append(gray)

        # 2) Gaussian + Otsu
        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        variants.append(th)

        # 3) Adaptive threshold
        adapt = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        variants.append(adapt)

        # 4) Morphology (to connect broken digits)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel)
        variants.append(morph)

        return variants

    # -------------------------------------------------
    # Main read
    # -------------------------------------------------
    def read(self, img):
        if img is None or img.size == 0:
            return "UNKNOWN"

        candidates = []
        for prep in self._preprocess_variants(img):
            p_text, p_conf = self._read_paddle(prep)
            e_text, e_conf = self._read_easy(prep)
            if p_text:
                candidates.append((p_text, p_conf))
            if e_text:
                candidates.append((e_text, e_conf))

        best = self._select_best(candidates)
        return best if best else "UNKNOWN"

    # -------------------------------------------------
    # PaddleOCR (numeric only)
    # -------------------------------------------------
    def _read_paddle(self, img):
        try:
            result = self.paddle.ocr(img, cls=False)
        except Exception:
            return "", 0.0

        texts, confs = [], []
        for line in result or []:
            for item in line:
                if len(item) == 2:
                    _, text_info = item
                    if isinstance(text_info, (list, tuple)):
                        text = text_info[0]
                        conf = text_info[1]
                        digits = re.sub(r"\D", "", text)
                        if digits:
                            texts.append(digits)
                            confs.append(conf)
                elif len(item) == 3:
                    _, text, conf = item
                    digits = re.sub(r"\D", "", text)
                    if digits:
                        texts.append(digits)
                        confs.append(conf)

        combined_text = "".join(texts)
        combined_conf = float(np.mean(confs)) if confs else 0.0
        return self._clean(combined_text), combined_conf

    # -------------------------------------------------
    # EasyOCR (numeric allowlist)
    # -------------------------------------------------
    def _read_easy(self, img):
        try:
            result = self.easy.readtext(
                img,
                allowlist="0123456789",
                detail=1,
                paragraph=False
            )
        except Exception:
            return "", 0.0

        texts, confs = [], []
        for item in result:
            digits = re.sub(r"\D", "", item[1])
            if digits:
                texts.append(digits)
                confs.append(item[2])

        combined_text = "".join(texts)
        combined_conf = float(np.mean(confs)) if confs else 0.0
        return self._clean(combined_text), combined_conf

    # -------------------------------------------------
    # Smart selection based on confidence & digits
    # -------------------------------------------------
    def _select_best(self, candidates):
        if not candidates:
            return ""

        scored = []
        for text, conf in candidates:
            if not text or len(text) < 4:
                continue
            digits_count = sum(c.isdigit() for c in text)

            # Apply flexible scoring: reward exact 7 digits, small penalty otherwise
            length_penalty = abs(7 - digits_count) * 1.0
            score = conf*3.0 + digits_count*2.0 - length_penalty
            scored.append((score, text))

        if not scored:
            return ""

        # Return text with highest score
        scored.sort(reverse=True, key=lambda x: x[0])
        return scored[0][1]

    # -------------------------------------------------
    # Cleaning + enforce numeric plate format
    # -------------------------------------------------
    def _clean(self, text):
        if not text:
            return ""

        digits = [c for c in text if c.isdigit()]

        if len(digits) < 6:
            return ""

        # If more than 7 digits, apply original logic: remove 3rd digit to normalize
        if len(digits) > 7 and len(digits) >= 3:
            del digits[2]
        digits = digits[:7]

        if len(digits) != 7:
            return ""

        # Format: XX_XXXXX
        return f"{digits[0]}{digits[1]}_{''.join(digits[2:])}"
