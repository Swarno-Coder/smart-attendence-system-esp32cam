"""
Attendance Service for Smart Attendance System
===============================================
Core business logic for attendance tracking.

Features:
- Automatic Entry/Exit alternation
- 60-second cooldown duplicate prevention
- Anomaly detection (night access, holidays, excessive scans)
- Face image storage for failed/suspicious attempts
- Daily attendance aggregation
"""

import logging
from datetime import datetime, date, timedelta
from typing import Optional, Tuple, Dict, Any
from pathlib import Path

from .models import Person, AttendanceLog, DailyAttendance, EventType, LogStatus
from .db_manager import get_db_manager, DatabaseManager

# Configure logging
logger = logging.getLogger(__name__)


class AttendanceResult:
    """
    Result of an attendance operation.
    Provides a structured response for API endpoints.
    """
    
    def __init__(
        self,
        success: bool,
        event_type: str,
        status: str,
        message: str,
        person_id: Optional[str] = None,
        person_name: Optional[str] = None,
        daily_summary: Optional[dict] = None,
        log_id: Optional[int] = None,
        cooldown_remaining: Optional[int] = None,
        is_anomaly: bool = False,
        anomaly_reason: Optional[str] = None
    ):
        self.success = success
        self.event_type = event_type
        self.status = status
        self.message = message
        self.person_id = person_id
        self.person_name = person_name
        self.daily_summary = daily_summary
        self.log_id = log_id
        self.cooldown_remaining = cooldown_remaining
        self.is_anomaly = is_anomaly
        self.anomaly_reason = anomaly_reason
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        result = {
            "success": self.success,
            "event_type": self.event_type,
            "status": self.status,
            "message": self.message
        }
        
        if self.person_id:
            result["person_id"] = self.person_id
        if self.person_name:
            result["person_name"] = self.person_name
        if self.daily_summary:
            result["daily_summary"] = self.daily_summary
        if self.log_id:
            result["log_id"] = self.log_id
        if self.cooldown_remaining is not None:
            result["cooldown_remaining_seconds"] = self.cooldown_remaining
        if self.is_anomaly:
            result["is_anomaly"] = self.is_anomaly
            result["anomaly_reason"] = self.anomaly_reason
        
        return result


class AttendanceService:
    """
    Main service for handling attendance operations.
    
    Usage:
        service = AttendanceService()
        result = service.log_attendance(
            person_id="emp001",
            person_name="John Doe",
            confidence=0.87,
            liveness_score=0.95
        )
    """
    
    def __init__(self, db_manager: Optional[DatabaseManager] = None):
        """
        Initialize attendance service.
        
        Args:
            db_manager: Optional DatabaseManager instance. Uses global if not provided.
        """
        self.db = db_manager or get_db_manager()
        self._cache_config()
    
    def _cache_config(self):
        """Cache frequently used configuration values."""
        self.cooldown_seconds = self.db.get_config_int("cooldown_seconds", 60)
        self.work_start_hour = self.db.get_config_int("work_start_hour", 6)
        self.work_end_hour = self.db.get_config_int("work_end_hour", 22)
        self.lunch_start_hour = self.db.get_config_int("lunch_start_hour", 12)
        self.lunch_end_hour = self.db.get_config_int("lunch_end_hour", 14)
        self.lunch_deduction_minutes = self.db.get_config_int("lunch_deduction_minutes", 60)
        self.min_hours_full_day = self.db.get_config_float("min_hours_full_day", 6.0)
        self.min_hours_half_day = self.db.get_config_float("min_hours_half_day", 3.0)
        self.max_daily_scans = self.db.get_config_int("max_daily_scans", 10)
        self.weekend_allowed = self.db.get_config_bool("weekend_allowed", False)
        self.save_failed_images = self.db.get_config_bool("save_failed_images", True)
        self.save_anomaly_images = self.db.get_config_bool("save_anomaly_images", True)
        
        logger.debug(f"Config cached: cooldown={self.cooldown_seconds}s, work_hours={self.work_start_hour}-{self.work_end_hour}")
    
    def refresh_config(self):
        """Refresh cached configuration from database."""
        self._cache_config()
    
    def log_attendance(
        self,
        person_id: str,
        person_name: str,
        confidence: float,
        liveness_score: float,
        device_id: str = "default",
        face_image: Optional[bytes] = None,
        timestamp: Optional[datetime] = None
    ) -> AttendanceResult:
        """
        Log an attendance event for a recognized person.
        
        This is the main entry point for attendance logging, called after
        successful face recognition.
        
        Args:
            person_id: Unique identifier of the person
            person_name: Display name of the person  
            confidence: Face match confidence (0-1)
            liveness_score: Anti-spoofing score (0-1)
            device_id: Camera/terminal identifier
            face_image: Optional JPEG bytes of captured face
            timestamp: Optional timestamp (defaults to now)
            
        Returns:
            AttendanceResult with event details
        """
        timestamp = timestamp or datetime.now()
        today = timestamp.date()
        
        logger.info(f"[ATTENDANCE] Processing: {person_name} ({person_id}) at {timestamp}")
        
        try:
            # Step 1: Check if person exists and is active
            is_valid, rejection_reason = self._validate_person(person_id)
            if not is_valid:
                return self._log_rejected_event(
                    person_id=person_id,
                    timestamp=timestamp,
                    reason=rejection_reason,
                    confidence=confidence,
                    liveness_score=liveness_score,
                    device_id=device_id,
                    face_image=face_image if self.save_failed_images else None
                )
            
            # Step 2: Check for anomalies (night access, weekend, etc.)
            is_anomaly, anomaly_reason = self._check_anomalies(timestamp)
            
            # Step 3: Check cooldown period
            is_duplicate, cooldown_remaining, last_log = self._check_cooldown(person_id, timestamp)
            if is_duplicate:
                return self._log_duplicate_event(
                    person_id=person_id,
                    person_name=person_name,
                    timestamp=timestamp,
                    cooldown_remaining=cooldown_remaining,
                    confidence=confidence,
                    liveness_score=liveness_score,
                    device_id=device_id
                )
            
            # Step 4: Determine event type (ENTRY or EXIT)
            event_type = self._determine_event_type(person_id, today, last_log)
            
            # Step 5: Log the event
            log_entry = self._create_log_entry(
                person_id=person_id,
                timestamp=timestamp,
                event_type=event_type,
                confidence=confidence,
                liveness_score=liveness_score,
                device_id=device_id,
                status=LogStatus.SUCCESS.value,
                is_anomaly=is_anomaly,
                anomaly_reason=anomaly_reason,
                face_image=face_image if (is_anomaly and self.save_anomaly_images) else None
            )
            
            # Step 6: Update daily attendance
            daily_summary = self._update_daily_attendance(person_id, today, event_type, timestamp)
            
            # Step 7: Update person's last_seen
            self._update_person_last_seen(person_id, timestamp)
            
            # Step 8: Check for excessive scans (IRREGULAR flag)
            if daily_summary and daily_summary.get("entry_count", 0) + daily_summary.get("exit_count", 0) > self.max_daily_scans:
                is_anomaly = True
                anomaly_reason = f"Excessive scans: {daily_summary.get('entry_count', 0) + daily_summary.get('exit_count', 0)} today"
                self._flag_log_as_anomaly(log_entry.id, anomaly_reason)
            
            # Format time for message
            time_str = timestamp.strftime("%I:%M %p")
            message = f"{event_type} logged for {person_name} at {time_str}"
            
            logger.info(f"[ATTENDANCE] SUCCESS: {message}")
            
            return AttendanceResult(
                success=True,
                event_type=event_type,
                status=LogStatus.SUCCESS.value,
                message=message,
                person_id=person_id,
                person_name=person_name,
                daily_summary=daily_summary,
                log_id=log_entry.id,
                is_anomaly=is_anomaly,
                anomaly_reason=anomaly_reason
            )
            
        except Exception as e:
            logger.error(f"[ATTENDANCE] ERROR: {e}", exc_info=True)
            return AttendanceResult(
                success=False,
                event_type=EventType.INVALID.value,
                status=LogStatus.ERROR.value,
                message=f"Error logging attendance: {str(e)}",
                person_id=person_id,
                person_name=person_name
            )
    
    def log_failed_recognition(
        self,
        reason: str,
        confidence: Optional[float] = None,
        liveness_score: Optional[float] = None,
        device_id: str = "default",
        face_image: Optional[bytes] = None,
        timestamp: Optional[datetime] = None
    ) -> AttendanceResult:
        """
        Log a failed recognition attempt (no face detected, spoof detected, unknown face).
        
        Args:
            reason: Why recognition failed
            confidence: Face match confidence if available
            liveness_score: Liveness score if available
            device_id: Camera/terminal identifier
            face_image: Optional JPEG bytes of captured face
            timestamp: Optional timestamp
            
        Returns:
            AttendanceResult with failure details
        """
        timestamp = timestamp or datetime.now()
        
        logger.info(f"[ATTENDANCE] Failed recognition: {reason}")
        
        try:
            log_entry = self._create_log_entry(
                person_id=None,
                timestamp=timestamp,
                event_type=EventType.INVALID.value,
                confidence=confidence,
                liveness_score=liveness_score,
                device_id=device_id,
                status=LogStatus.REJECTED.value,
                rejection_reason=reason,
                is_anomaly=False,
                face_image=face_image if self.save_failed_images else None
            )
            
            return AttendanceResult(
                success=False,
                event_type=EventType.INVALID.value,
                status=LogStatus.REJECTED.value,
                message=reason,
                log_id=log_entry.id
            )
            
        except Exception as e:
            logger.error(f"[ATTENDANCE] ERROR logging failed recognition: {e}")
            return AttendanceResult(
                success=False,
                event_type=EventType.INVALID.value,
                status=LogStatus.ERROR.value,
                message=str(e)
            )
    
    def _validate_person(self, person_id: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that person exists and is active.
        
        Returns:
            (is_valid, rejection_reason)
        """
        with self.db.get_session() as session:
            person = session.query(Person).filter_by(person_id=person_id).first()
            
            if not person:
                # Person not in DB - auto-create if we're lenient
                # For strict mode, return rejection
                logger.warning(f"Person not found in database: {person_id}")
                # Auto-create person for now (can make configurable)
                return (True, None)  # Allow, will be created on first log
            
            if not person.is_active:
                return (False, f"Person {person_id} is inactive/deactivated")
            
            return (True, None)
    
    def _check_anomalies(self, timestamp: datetime) -> Tuple[bool, Optional[str]]:
        """
        Check for anomalous access patterns.
        
        Returns:
            (is_anomaly, anomaly_reason)
        """
        hour = timestamp.hour
        weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
        
        # Night access check
        if hour < self.work_start_hour or hour >= self.work_end_hour:
            return (True, f"Night/after-hours access at {timestamp.strftime('%H:%M')}")
        
        # Weekend check
        if weekday >= 5 and not self.weekend_allowed:  # Saturday(5) or Sunday(6)
            day_name = "Saturday" if weekday == 5 else "Sunday"
            return (True, f"Weekend access on {day_name}")
        
        return (False, None)
    
    def _check_cooldown(
        self, 
        person_id: str, 
        timestamp: datetime
    ) -> Tuple[bool, int, Optional[AttendanceLog]]:
        """
        Check if person has logged within cooldown period.
        
        Returns:
            (is_duplicate, cooldown_remaining_seconds, last_log)
        """
        with self.db.get_session() as session:
            # Get most recent log for this person today
            today_start = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            
            last_log = session.query(AttendanceLog).filter(
                AttendanceLog.person_id == person_id,
                AttendanceLog.timestamp >= today_start,
                AttendanceLog.status == LogStatus.SUCCESS.value
            ).order_by(AttendanceLog.timestamp.desc()).first()
            
            if not last_log:
                return (False, 0, None)
            
            # Calculate time since last log
            time_diff = (timestamp - last_log.timestamp).total_seconds()
            
            if time_diff < self.cooldown_seconds:
                cooldown_remaining = int(self.cooldown_seconds - time_diff)
                logger.debug(f"Within cooldown: {time_diff:.1f}s since last log, {cooldown_remaining}s remaining")
                return (True, cooldown_remaining, last_log)
            
            return (False, 0, last_log)
    
    def _determine_event_type(
        self, 
        person_id: str, 
        today: date,
        last_log: Optional[AttendanceLog]
    ) -> str:
        """
        Determine if this should be ENTRY or EXIT.
        
        Rules:
        - First scan of day → ENTRY
        - After ENTRY → EXIT  
        - After EXIT → ENTRY
        """
        if not last_log:
            return EventType.ENTRY.value
        
        if last_log.event_type == EventType.ENTRY.value:
            return EventType.EXIT.value
        else:
            return EventType.ENTRY.value
    
    def _create_log_entry(
        self,
        person_id: Optional[str],
        timestamp: datetime,
        event_type: str,
        confidence: Optional[float],
        liveness_score: Optional[float],
        device_id: str,
        status: str,
        rejection_reason: Optional[str] = None,
        is_anomaly: bool = False,
        anomaly_reason: Optional[str] = None,
        face_image: Optional[bytes] = None
    ) -> AttendanceLog:
        """Create and persist an attendance log entry."""
        
        with self.db.get_session() as session:
            log_entry = AttendanceLog(
                person_id=person_id,
                timestamp=timestamp,
                event_type=event_type,
                confidence=confidence,
                liveness_score=liveness_score,
                device_id=device_id,
                status=status,
                rejection_reason=rejection_reason,
                is_anomaly=is_anomaly,
                anomaly_reason=anomaly_reason,
                face_snapshot=face_image
            )
            session.add(log_entry)
            session.commit()
            session.refresh(log_entry)
            
            logger.debug(f"Created log entry: id={log_entry.id}, type={event_type}, status={status}")
            return log_entry
    
    def _update_daily_attendance(
        self, 
        person_id: str, 
        today: date, 
        event_type: str,
        timestamp: datetime
    ) -> dict:
        """
        Update or create daily attendance summary.
        
        Returns:
            Dictionary with daily summary
        """
        with self.db.get_session() as session:
            # Find or create daily record
            daily = session.query(DailyAttendance).filter_by(
                person_id=person_id,
                attendance_date=today
            ).first()
            
            if not daily:
                daily = DailyAttendance(
                    person_id=person_id,
                    attendance_date=today,
                    first_entry=timestamp if event_type == EventType.ENTRY.value else None,
                    entry_count=1 if event_type == EventType.ENTRY.value else 0,
                    exit_count=1 if event_type == EventType.EXIT.value else 0,
                    status="PRESENT"
                )
                session.add(daily)
            else:
                if event_type == EventType.ENTRY.value:
                    daily.entry_count += 1
                    if not daily.first_entry:
                        daily.first_entry = timestamp
                elif event_type == EventType.EXIT.value:
                    daily.exit_count += 1
                    daily.last_exit = timestamp
            
            # Calculate hours if we have entry and exit
            if daily.first_entry and daily.last_exit:
                daily.total_hours = self._calculate_work_hours(
                    daily.first_entry, 
                    daily.last_exit
                )
                
                # Update status based on hours
                if daily.total_hours >= self.min_hours_full_day:
                    daily.status = "PRESENT"
                elif daily.total_hours >= self.min_hours_half_day:
                    daily.status = "HALF_DAY"
                else:
                    daily.status = "HALF_DAY"  # Still counted as half day if any attendance
            
            session.commit()
            session.refresh(daily)
            
            return daily.to_dict()
    
    def _calculate_work_hours(self, first_entry: datetime, last_exit: datetime) -> float:
        """
        Calculate work hours between first entry and last exit.
        Deducts lunch time if applicable.
        """
        if not first_entry or not last_exit:
            return 0.0
        
        total_seconds = (last_exit - first_entry).total_seconds()
        total_hours = total_seconds / 3600
        
        # Deduct lunch if worked through lunch period
        entry_hour = first_entry.hour
        exit_hour = last_exit.hour
        
        # If entry before lunch and exit after lunch, deduct lunch time
        if entry_hour < self.lunch_end_hour and exit_hour >= self.lunch_start_hour:
            # Calculate overlap with lunch period
            lunch_start = first_entry.replace(hour=self.lunch_start_hour, minute=0, second=0)
            lunch_end = first_entry.replace(hour=self.lunch_end_hour, minute=0, second=0)
            
            # Only deduct if they actually worked through part of lunch
            if first_entry < lunch_end and last_exit > lunch_start:
                lunch_deduction = self.lunch_deduction_minutes / 60.0
                total_hours = max(0, total_hours - lunch_deduction)
        
        return round(total_hours, 2)
    
    def _update_person_last_seen(self, person_id: str, timestamp: datetime):
        """Update person's last_seen timestamp."""
        with self.db.get_session() as session:
            person = session.query(Person).filter_by(person_id=person_id).first()
            if person:
                person.last_seen = timestamp
                session.commit()
    
    def _flag_log_as_anomaly(self, log_id: int, reason: str):
        """Flag an existing log entry as anomaly."""
        with self.db.get_session() as session:
            log = session.query(AttendanceLog).filter_by(id=log_id).first()
            if log:
                log.is_anomaly = True
                log.anomaly_reason = reason
                session.commit()
    
    def _log_rejected_event(
        self,
        person_id: str,
        timestamp: datetime,
        reason: str,
        confidence: float,
        liveness_score: float,
        device_id: str,
        face_image: Optional[bytes]
    ) -> AttendanceResult:
        """Log a rejected attendance attempt."""
        
        log_entry = self._create_log_entry(
            person_id=person_id,
            timestamp=timestamp,
            event_type=EventType.INVALID.value,
            confidence=confidence,
            liveness_score=liveness_score,
            device_id=device_id,
            status=LogStatus.REJECTED.value,
            rejection_reason=reason,
            face_image=face_image
        )
        
        return AttendanceResult(
            success=False,
            event_type=EventType.INVALID.value,
            status=LogStatus.REJECTED.value,
            message=reason,
            person_id=person_id,
            log_id=log_entry.id
        )
    
    def _log_duplicate_event(
        self,
        person_id: str,
        person_name: str,
        timestamp: datetime,
        cooldown_remaining: int,
        confidence: float,
        liveness_score: float,
        device_id: str
    ) -> AttendanceResult:
        """Log a duplicate scan (within cooldown period)."""
        
        log_entry = self._create_log_entry(
            person_id=person_id,
            timestamp=timestamp,
            event_type=EventType.DUPLICATE.value,
            confidence=confidence,
            liveness_score=liveness_score,
            device_id=device_id,
            status=LogStatus.ALREADY_LOGGED.value,
            rejection_reason=f"Within {self.cooldown_seconds}s cooldown period"
        )
        
        message = f"{person_name} already logged {self.cooldown_seconds - cooldown_remaining} seconds ago"
        logger.info(f"[ATTENDANCE] DUPLICATE: {message}")
        
        return AttendanceResult(
            success=True,  # Not a failure, just already logged
            event_type=EventType.DUPLICATE.value,
            status=LogStatus.ALREADY_LOGGED.value,
            message=message,
            person_id=person_id,
            person_name=person_name,
            cooldown_remaining=cooldown_remaining
        )
    
    # ============== Query Methods ==============
    
    def get_person_today_summary(self, person_id: str) -> Optional[dict]:
        """Get today's attendance summary for a person."""
        today = date.today()
        
        with self.db.get_session() as session:
            daily = session.query(DailyAttendance).filter_by(
                person_id=person_id,
                attendance_date=today
            ).first()
            
            return daily.to_dict() if daily else None
    
    def get_today_logs(self, person_id: Optional[str] = None, limit: int = 100) -> list:
        """Get today's attendance logs, optionally filtered by person."""
        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        
        with self.db.get_session() as session:
            query = session.query(AttendanceLog).filter(
                AttendanceLog.timestamp >= today_start
            )
            
            if person_id:
                query = query.filter(AttendanceLog.person_id == person_id)
            
            logs = query.order_by(AttendanceLog.timestamp.desc()).limit(limit).all()
            return [log.to_dict() for log in logs]
    
    def get_daily_report(self, report_date: Optional[date] = None) -> dict:
        """Get attendance report for a specific date."""
        report_date = report_date or date.today()
        
        with self.db.get_session() as session:
            daily_records = session.query(DailyAttendance).filter_by(
                attendance_date=report_date
            ).all()
            
            total_persons = session.query(Person).filter_by(is_active=True).count()
            present_count = len([d for d in daily_records if d.status in ("PRESENT", "HALF_DAY")])
            
            return {
                "date": report_date.isoformat(),
                "total_registered": total_persons,
                "present_count": present_count,
                "absent_count": total_persons - present_count,
                "attendance_rate": round(present_count / total_persons * 100, 1) if total_persons > 0 else 0,
                "records": [d.to_dict() for d in daily_records]
            }
    
    def ensure_person_exists(self, person_id: str, person_name: str) -> Person:
        """
        Ensure a person exists in the database,creating if necessary.
        Called during face registration.
        """
        with self.db.get_session() as session:
            person = session.query(Person).filter_by(person_id=person_id).first()
            
            if not person:
                person = Person(
                    person_id=person_id,
                    person_name=person_name,
                    is_active=True,
                    registered_at=datetime.utcnow()
                )
                session.add(person)
                session.commit()
                session.refresh(person)
                logger.info(f"Created new person record: {person_name} ({person_id})")
            
            return person
