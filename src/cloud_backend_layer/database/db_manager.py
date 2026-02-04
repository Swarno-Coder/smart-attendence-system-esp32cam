"""
Database Manager for Smart Attendance System
=============================================
Handles database connection, initialization, and session management.

Features:
- SQLite database with async support ready
- Automatic table creation
- Default configuration seeding
- Connection pooling for concurrent requests
"""

import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Generator
from contextlib import contextmanager

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from .models import Base, SystemConfig, Person, DEFAULT_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

# Database file path (same directory as this module)
DATABASE_DIR = Path(__file__).parent
DATABASE_PATH = DATABASE_DIR / "attendance.db"


class DatabaseManager:
    """
    Manages database connections and provides session context.
    
    Usage:
        db = DatabaseManager()
        with db.get_session() as session:
            person = session.query(Person).filter_by(person_id="emp001").first()
    """
    
    _instance: Optional['DatabaseManager'] = None
    
    def __init__(self, db_path: Optional[Path] = None, echo: bool = False):
        """
        Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file. Defaults to attendance.db
            echo: If True, log all SQL statements (useful for debugging)
        """
        self.db_path = db_path or DATABASE_PATH
        self.echo = echo
        self.engine = None
        self.SessionLocal = None
        self._initialized = False
        
    def initialize(self) -> bool:
        """
        Initialize database connection and create tables.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            # Ensure database directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Create SQLite engine
            # check_same_thread=False needed for async/multi-threaded access
            # StaticPool for connection reuse
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=self.echo,
                connect_args={"check_same_thread": False},
                poolclass=StaticPool
            )
            
            # Enable foreign key support (SQLite has it disabled by default)
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
                cursor.close()
            
            # Create session factory
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine
            )
            
            # Create all tables
            Base.metadata.create_all(bind=self.engine)
            logger.info(f"Database initialized at: {self.db_path}")
            
            # Seed default configuration
            self._seed_default_config()
            
            self._initialized = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
    
    def _seed_default_config(self):
        """Insert default configuration values if not present."""
        with self.get_session() as session:
            for key, (value, description) in DEFAULT_CONFIG.items():
                existing = session.query(SystemConfig).filter_by(key=key).first()
                if not existing:
                    config = SystemConfig(
                        key=key,
                        value=value,
                        description=description
                    )
                    session.add(config)
                    logger.debug(f"Added default config: {key}={value}")
            session.commit()
            logger.info("Default configuration seeded")
    
    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """
        Get a database session with automatic cleanup.
        
        Usage:
            with db.get_session() as session:
                # do database operations
                session.commit()
        
        Yields:
            SQLAlchemy Session object
        """
        if not self._initialized:
            self.initialize()
        
        session = self.SessionLocal()
        try:
            yield session
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def get_config(self, key: str, default: str = None) -> Optional[str]:
        """
        Get a configuration value by key.
        
        Args:
            key: Configuration key name
            default: Default value if key not found
            
        Returns:
            Configuration value as string
        """
        with self.get_session() as session:
            config = session.query(SystemConfig).filter_by(key=key).first()
            return config.value if config else default
    
    def get_config_int(self, key: str, default: int = 0) -> int:
        """Get config value as integer."""
        value = self.get_config(key)
        try:
            return int(value) if value else default
        except ValueError:
            return default
    
    def get_config_float(self, key: str, default: float = 0.0) -> float:
        """Get config value as float."""
        value = self.get_config(key)
        try:
            return float(value) if value else default
        except ValueError:
            return default
    
    def get_config_bool(self, key: str, default: bool = False) -> bool:
        """Get config value as boolean."""
        value = self.get_config(key)
        if value is None:
            return default
        return value.lower() in ('true', '1', 'yes', 'on')
    
    def set_config(self, key: str, value: str, description: str = None):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key name
            value: Configuration value
            description: Optional description
        """
        with self.get_session() as session:
            config = session.query(SystemConfig).filter_by(key=key).first()
            if config:
                config.value = value
                config.updated_at = datetime.utcnow()
                if description:
                    config.description = description
            else:
                config = SystemConfig(
                    key=key,
                    value=value,
                    description=description
                )
                session.add(config)
            session.commit()
            logger.info(f"Config updated: {key}={value}")
    
    def sync_persons_from_embeddings(self, embeddings_dir: Path) -> int:
        """
        Sync persons table with existing face embeddings.
        Creates Person records for any embeddings not already in database.
        
        Args:
            embeddings_dir: Path to directory containing .npz embedding files
            
        Returns:
            Number of new persons added
        """
        import numpy as np
        
        added_count = 0
        embedding_files = list(embeddings_dir.glob("*.npz"))
        
        with self.get_session() as session:
            for emb_file in embedding_files:
                try:
                    data = np.load(emb_file, allow_pickle=True)
                    
                    # Extract person info from embedding file
                    person_id = str(data['person_id']) if 'person_id' in data.files else emb_file.stem
                    person_name = str(data['person_name']) if 'person_name' in data.files else emb_file.stem
                    
                    # Handle numpy scalar arrays
                    if hasattr(person_id, 'item'):
                        person_id = person_id.item()
                    if hasattr(person_name, 'item'):
                        person_name = person_name.item()
                    
                    # Check if person already exists
                    existing = session.query(Person).filter_by(person_id=person_id).first()
                    if not existing:
                        person = Person(
                            person_id=person_id,
                            person_name=person_name,
                            is_active=True,
                            registered_at=datetime.utcnow()
                        )
                        session.add(person)
                        added_count += 1
                        logger.info(f"Synced person from embedding: {person_name} ({person_id})")
                        
                except Exception as e:
                    logger.error(f"Failed to sync embedding {emb_file}: {e}")
            
            session.commit()
        
        logger.info(f"Sync complete: {added_count} new persons added from {len(embedding_files)} embeddings")
        return added_count
    
    def get_stats(self) -> dict:
        """
        Get database statistics.
        
        Returns:
            Dictionary with counts and status info
        """
        with self.get_session() as session:
            from .models import AttendanceLog, DailyAttendance
            
            person_count = session.query(Person).count()
            active_person_count = session.query(Person).filter_by(is_active=True).count()
            total_logs = session.query(AttendanceLog).count()
            today_logs = session.query(AttendanceLog).filter(
                AttendanceLog.timestamp >= datetime.now().replace(hour=0, minute=0, second=0)
            ).count()
            
            return {
                "database_path": str(self.db_path),
                "total_persons": person_count,
                "active_persons": active_person_count,
                "total_logs": total_logs,
                "logs_today": today_logs,
                "initialized": self._initialized
            }
    
    def close(self):
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            logger.info("Database connection closed")


# Global singleton instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """
    Get the global database manager instance.
    Creates and initializes if not already done.
    
    Returns:
        DatabaseManager singleton instance
    """
    global _db_manager
    
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.initialize()
    
    return _db_manager


def reset_db_manager():
    """Reset the global database manager (for testing)."""
    global _db_manager
    if _db_manager:
        _db_manager.close()
    _db_manager = None
