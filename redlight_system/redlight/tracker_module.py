# -*- coding: utf-8 -*-
import numpy as np
from norfair import Detection, Tracker

class TrackerWrapper:
    """
    غلاف لنورفير: يستخدم دالة المسافة المقدمة بطريقة متوافقة.
    """
    def __init__(self, distance_function, distance_threshold=40):
        # norfair expects distance_function(det, trk) but in الأصل استعملنا توقيعًا أ، ب.
        def _df(detection, tracked):
            try:
                # detection.points is array-like; tracked.estimate is array-like
                pa = np.array(detection.points[0])
                pb = np.array(tracked.estimate[0])
                return float(distance_function(pa, pb))
            except Exception:
                return 1e6
        self.tracker = Tracker(distance_function=_df, distance_threshold=distance_threshold)

    def update(self, detections):
        """
        detections: قائمة من Detection (norfair.Detection) أو نقاط numpy مع شكل (1,2)
        في الكود الرئيسي سنمرر Norfair Detection أمامياً.
        """
        # إذا تم تمرير Norfair Detection بالفعل — استخدمها.
        # لكن لمعالجة عامة، نقبل قائمة من Norfair.Detection أو من numpy نقاط.
        norfair_dets = []
        for d in detections:
            if isinstance(d, Detection):
                norfair_dets.append(d)
            else:
                # نفترض d هو numpy array (2,) أو (1,2)
                pts = np.array(d).reshape(1, 2)
                norfair_dets.append(Detection(points=pts))
        tracked = self.tracker.update(detections=norfair_dets)
        return tracked