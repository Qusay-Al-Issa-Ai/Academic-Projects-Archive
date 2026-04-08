# -*- coding: utf-8 -*-
import cv2
import numpy as np
from skimage.filters import threshold_sauvola

class PlateEnhancerModule:
    """
    Classical Enhancer (No-AI):
    مخصص للصور المفهومة بشرياً ولكنها مهتزة أو مغبشة.
    يعتمد على الرياضيات لاستعادة الحدة والتباين بدلاً من التوليد.
    """
    
    def __init__(self, model_path=None, scale=None, second_pass=None, min_size_for_second=None, **kwargs):
        """
        نستقبل جميع المعاملات القديمة (model_path, scale, ...)
        لضمان عدم توقف النظام، لكننا نستخدم إعداداتنا الخاصة.
        """
        print("[PLATE-ENHANCER] Mode: CLASSICAL (Sharpening & Contrast) - Active")
        print("[PLATE-ENHANCER] Ignoring AI models to prevent hallucinations.")

    def enhance(self, crop_img):
        if crop_img is None or crop_img.size == 0:
            return crop_img

        # 1. تصحيح الإضاءة (Gamma Correction)
        # يرفع سطوع المناطق المظلمة داخل الأرقام
        img_gamma = self._adjust_gamma(crop_img, gamma=1.2)

        # 2. التكبير الرياضي الصلب (Lanczos Upscaling)
        # نكبر الصورة 3 أضعاف لإعطاء مساحة للفلاتر للعمل
        h, w = img_gamma.shape[:2]
        scale_factor = 3
        img_large = cv2.resize(img_gamma, (w * scale_factor, h * scale_factor), interpolation=cv2.INTER_LANCZOS4)

        # 3. إزالة الضبابية (Deblurring / Sharpening)
        # تقنية Unsharp Masking لإبراز الحواف المخفية
        gaussian = cv2.GaussianBlur(img_large, (0, 0), 2.0)
        img_sharp = cv2.addWeighted(img_large, 1.6, gaussian, -0.6, 0)

        # 4. تعزيز التباين اللوني (CLAHE on L-channel)
        # نفصل الأرقام عن الخلفية بناءً على الإضاءة
        lab = cv2.cvtColor(img_sharp, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        img_contrast = cv2.merge((l, a, b))
        img_bgr_contrast = cv2.cvtColor(img_contrast, cv2.COLOR_LAB2BGR)

        # 5. التقطيع الذكي (Adaptive Binarization)
        # نستخدم Sauvola بدلاً من Otsu لأنها تتعامل بعبقرية مع الظلال
        gray = cv2.cvtColor(img_bgr_contrast, cv2.COLOR_BGR2GRAY)
        
        # حجم النافذة يعتمد على حجم الصورة المكبرة (25 ممتاز للوحات)
        window_size = 25
        thresh_sauvola = threshold_sauvola(gray, window_size=window_size)
        binary = (gray > thresh_sauvola).astype(np.uint8) * 255

        # 6. تنظيف الشوائب (Artifact Removal)
        # حذف أي نقطة سوداء صغيرة لا ترقى لتكون رقماً
        final_clean = self._clean_noise(binary)

        # إعادة الصورة بصيغة BGR لكي لا يرفضها النظام
        return cv2.cvtColor(final_clean, cv2.COLOR_GRAY2BGR)

    def _adjust_gamma(self, image, gamma=1.0):
        """تفتيح المناطق المظلمة دون حرق المناطق الفاتحة"""
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
            for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def _clean_noise(self, binary_img):
        """حذف النقاط السوداء الصغيرة (الفلفل) من الصورة البيضاء"""
        # نعكس الصورة لأن الدالة ConnectedComponents تبحث عن الأبيض
        inverted = cv2.bitwise_not(binary_img)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(inverted, connectivity=8)
        
        cleaned_inverted = np.zeros_like(inverted)
        
        # بما أننا كبرنا الصورة 3 أضعاف، فإن أي جسم مساحته أقل من 150 بكسل هو غالباً وسخ
        min_area = 150 

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                cleaned_inverted[labels == i] = 255
        
        # إعادة العكس للوضع الطبيعي (نص أسود على خلفية بيضاء)
        return cv2.bitwise_not(cleaned_inverted)

