import sqlite3

DB_PATH = "violations.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS violations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        incident_id TEXT UNIQUE,
        violation_type TEXT,
        violation_time TEXT,
        source_video TEXT,
        plate_number TEXT,
        confidence REAL,
        vehicle_image_path TEXT,
        plate_crop_path TEXT,
        plate_enhanced_path TEXT,
        evidence_dir TEXT,
        status TEXT DEFAULT 'CONFIRMED',
        created_at TEXT
    )
    """)

    conn.commit()
    conn.close()
if __name__ == "__main__":
    init_db()
    print("Database initialized")