import sqlite3
from redlight.utils import EVENTS_DIR

db_path = r"C:\Users\C E C\Desktop\qusay_progects\redlight_system\events\violations.db"

conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# حذف جميع السجلات من جدول violations
cursor.execute("DELETE FROM violations")
conn.commit()

# إعادة ضبط عداد الـ AUTOINCREMENT (اختياري)
cursor.execute("DELETE FROM sqlite_sequence WHERE name='violations'")
conn.commit()

conn.close()

print("تم حذف جميع البيانات من قاعدة البيانات بنجاح!")