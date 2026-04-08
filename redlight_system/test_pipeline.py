# test_pipeline.py

import cv2
from plate_pipeline import PlatePipeline


# تحميل الصورة (أي صورة سيارة)
image_path = r"C:\Users\C E C\Desktop\qusay_progects\images\i1.jpg"
img = cv2.imread(image_path)

pipeline = PlatePipeline()

results = pipeline.run(img)

if not results:
    print("❌ لم يتم العثور على أي لوحة")
else:
    for i, res in enumerate(results):
        print(f"\n===== لوحة #{i+1} =====")
        print("النص المستخرج:", res["text"])

        # عرض الصور
        cv2.imshow(f"Crop {i+1}", res["crop"])
        cv2.imshow(f"Enhanced {i+1}", res["enhanced"])

    cv2.waitKey(0)