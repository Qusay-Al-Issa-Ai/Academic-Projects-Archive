import sqlite3
conn = sqlite3.connect(r"C:\Users\C E C\Desktop\qusay_progects\redlight_system\events\violations.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM violations")
print(cursor.fetchall())
conn.close()