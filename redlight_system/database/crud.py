from database.db import SessionLocal
from database.models import Violation
from datetime import datetime, timedelta

# Time window to prevent duplicate violations (seconds)
DUPLICATE_TIME_WINDOW = 10

def save_violation(
    plate_number,
    confidence,
    camera,
    violation_image,
    plate_image
):
    session = SessionLocal()

    # Check for recent duplicate violation
    recent_time = datetime.utcnow() - timedelta(seconds=DUPLICATE_TIME_WINDOW)
    duplicate = session.query(Violation).filter(
        Violation.plate_number == plate_number,
        Violation.timestamp >= recent_time
    ).first()

    if duplicate:
        session.close()
        return False

    violation = Violation(
        plate_number=plate_number,
        confidence=confidence,
        camera=camera,
        violation_image=violation_image,
        plate_image=plate_image
    )

    session.add(violation)
    session.commit()
    session.close()
    return True