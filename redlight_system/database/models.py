from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database.db import Base

class Violation(Base):
    __tablename__ = "violations"

    id = Column(Integer, primary_key=True, index=True)
    plate_number = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)
    camera = Column(String)
    violation_image = Column(String)
    plate_image = Column(String)