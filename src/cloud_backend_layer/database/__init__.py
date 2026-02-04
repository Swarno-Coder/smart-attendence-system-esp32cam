"""
Database Module for Smart Attendance System
============================================
Provides SQLite-based attendance tracking with:
- Entry/Exit logging
- Duplicate prevention (cooldown)
- Anomaly detection
"""

from .models import Person, AttendanceLog, DailyAttendance, SystemConfig
from .db_manager import DatabaseManager, get_db_manager
from .attendance_service import AttendanceService

__all__ = [
    'Person',
    'AttendanceLog', 
    'DailyAttendance',
    'SystemConfig',
    'DatabaseManager',
    'get_db_manager',
    'AttendanceService'
]
