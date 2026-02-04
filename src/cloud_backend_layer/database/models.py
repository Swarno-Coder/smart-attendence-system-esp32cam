"""
Database Models for Smart Attendance System
============================================
SQLAlchemy ORM models for attendance tracking.

Tables:
- persons: Registered users (synced with face embeddings)
- attendance_logs: Immutable audit trail of all scan events
- daily_attendance: Aggregated daily summaries
- system_config: Configurable system parameters
"""

from datetime import datetime, date
from typing import Optional
from enum import Enum
from sqlalchemy import (
    Column, String, Integer, Float, Boolean, DateTime, Date, 
    Text, LargeBinary, ForeignKey, Enum as SQLEnum, create_engine
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()


class EventType(str, Enum):
    """Type of attendance event."""
    ENTRY = "ENTRY"
    EXIT = "EXIT"
    DUPLICATE = "DUPLICATE"
    INVALID = "INVALID"


class LogStatus(str, Enum):
    """Status of the attendance log entry."""
    SUCCESS = "SUCCESS"
    REJECTED = "REJECTED"
    ERROR = "ERROR"
    ALREADY_LOGGED = "ALREADY_LOGGED"


class AttendanceStatus(str, Enum):
    """Daily attendance status."""
    PRESENT = "PRESENT"
    HALF_DAY = "HALF_DAY"
    ABSENT = "ABSENT"
    IRREGULAR = "IRREGULAR"


class Person(Base):
    """
    Registered users table.
    Synced with face embeddings stored in .npz files.
    """
    __tablename__ = 'persons'
    
    person_id = Column(String(50), primary_key=True, index=True)
    person_name = Column(String(100), nullable=False)
    department = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    registered_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    last_seen = Column(DateTime, nullable=True)
    
    # Note: Relationships removed to avoid SQLAlchemy mapper recursion issues
    # Use explicit joins in queries when needed
    
    def __repr__(self):
        return f"<Person(id={self.person_id}, name={self.person_name}, active={self.is_active})>"


class AttendanceLog(Base):
    """
    Immutable audit trail of all recognition events.
    Every face recognition attempt is logged here.
    """
    __tablename__ = 'attendance_logs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(50), ForeignKey('persons.person_id'), nullable=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    event_type = Column(String(20), nullable=False)  # ENTRY, EXIT, DUPLICATE, INVALID
    confidence = Column(Float, nullable=True)  # Face match confidence (0-1)
    liveness_score = Column(Float, nullable=True)  # Anti-spoofing confidence
    device_id = Column(String(50), default="default", nullable=False)
    status = Column(String(20), nullable=False)  # SUCCESS, REJECTED, ERROR, ALREADY_LOGGED
    rejection_reason = Column(Text, nullable=True)
    face_snapshot = Column(LargeBinary, nullable=True)  # Optional captured image
    is_anomaly = Column(Boolean, default=False, nullable=False)  # Flagged as suspicious
    anomaly_reason = Column(String(200), nullable=True)  # Why flagged
    
    # Note: Relationships removed to avoid SQLAlchemy mapper recursion issues
    
    def __repr__(self):
        return f"<AttendanceLog(id={self.id}, person={self.person_id}, type={self.event_type}, status={self.status})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "id": self.id,
            "person_id": self.person_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "event_type": self.event_type,
            "confidence": self.confidence,
            "liveness_score": self.liveness_score,
            "device_id": self.device_id,
            "status": self.status,
            "rejection_reason": self.rejection_reason,
            "is_anomaly": self.is_anomaly,
            "anomaly_reason": self.anomaly_reason
        }


class DailyAttendance(Base):
    """
    Aggregated daily attendance summary.
    Updated after each successful ENTRY/EXIT.
    """
    __tablename__ = 'daily_attendance'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    person_id = Column(String(50), ForeignKey('persons.person_id'), nullable=False, index=True)
    attendance_date = Column(Date, nullable=False, index=True)
    first_entry = Column(DateTime, nullable=True)
    last_exit = Column(DateTime, nullable=True)
    entry_count = Column(Integer, default=0, nullable=False)
    exit_count = Column(Integer, default=0, nullable=False)
    total_hours = Column(Float, default=0.0, nullable=False)
    status = Column(String(20), default="PRESENT")  # PRESENT, HALF_DAY, ABSENT, IRREGULAR
    
    # Note: Relationships removed to avoid SQLAlchemy mapper recursion issues
    
    # Unique constraint: one record per person per day
    __table_args__ = (
        # SQLite doesn't enforce this well, we'll handle in code
    )
    
    def __repr__(self):
        return f"<DailyAttendance(person={self.person_id}, date={self.attendance_date}, status={self.status})>"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "person_id": self.person_id,
            "attendance_date": self.attendance_date.isoformat() if self.attendance_date else None,
            "first_entry": self.first_entry.strftime("%H:%M:%S") if self.first_entry else None,
            "last_exit": self.last_exit.strftime("%H:%M:%S") if self.last_exit else None,
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "total_hours": round(self.total_hours, 2),
            "status": self.status
        }


class SystemConfig(Base):
    """
    System configuration parameters.
    Allows runtime configuration without code changes.
    """
    __tablename__ = 'system_config'
    
    key = Column(String(100), primary_key=True)
    value = Column(String(500), nullable=False)
    description = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f"<SystemConfig(key={self.key}, value={self.value})>"


# Default configuration values
DEFAULT_CONFIG = {
    "cooldown_seconds": ("60", "Minimum seconds between valid scans for same person"),
    "work_start_hour": ("6", "Start of allowed attendance window (24h format)"),
    "work_end_hour": ("22", "End of allowed attendance window (24h format)"),
    "lunch_start_hour": ("12", "Lunch break start hour"),
    "lunch_end_hour": ("14", "Lunch break end hour"),
    "lunch_deduction_minutes": ("60", "Minutes to deduct for lunch if full day"),
    "min_hours_full_day": ("6.0", "Hours needed for PRESENT status"),
    "min_hours_half_day": ("3.0", "Hours needed for HALF_DAY status"),
    "max_daily_scans": ("10", "Scans above this = IRREGULAR flag"),
    "weekend_allowed": ("false", "Allow weekend attendance"),
    "save_failed_images": ("true", "Save face images for failed attempts"),
    "save_anomaly_images": ("true", "Save face images for anomalous attempts"),
}
