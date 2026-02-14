"""
Database Manager
Handles all SQLite database operations for driver enrollment and verification logging.
Refactored to use Repository pattern.
"""

import sqlite3
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

from database.models import Driver, VerificationLog, SystemAuditLog
from database.driver_repository import DriverRepository
from database.verification_repository import VerificationRepository
from database.audit_repository import AuditRepository
from utils.config import config

class DatabaseManager:
    """Manages SQLite database operations via repositories"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file (uses config if not provided)
        """
        self.db_path = db_path or config.database_path
        self._ensure_database_exists()
        self._create_tables()

        # Initialize repositories
        self.driver_repo = DriverRepository(self._get_connection)
        self.verification_repo = VerificationRepository(self._get_connection)
        self.audit_repo = AuditRepository(self._get_connection)
    
    def _ensure_database_exists(self):
        """Ensure database directory exists"""
        db_dir = Path(self.db_path).parent
        db_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.Connection(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _create_tables(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create drivers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                id_number TEXT,
                biometric_embedding BLOB NOT NULL,
                enrollment_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                email TEXT,
                status TEXT DEFAULT 'active'
            )
        """)
        
        # Create verification_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_logs (
                log_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                driver_id INTEGER,
                driver_name TEXT,
                similarity_score REAL,
                authorized BOOLEAN,
                processing_time_ms REAL,
                image_path TEXT,
                liveness_passed BOOLEAN,
                FOREIGN KEY (driver_id) REFERENCES drivers(driver_id)
            )
        """)
        
        # Create index on verification_logs timestamp for faster queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_verification_timestamp 
            ON verification_logs(timestamp DESC)
        """)
        
        # Create audit_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                action TEXT NOT NULL,
                user_email TEXT,
                details TEXT,
                ip_address TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    # ==================== Driver Operations ====================
    
    def enroll_driver(self, name: str, embedding: np.ndarray, email: str = None, id_number: str = None) -> int:
        """Enroll a new driver in the database"""
        return self.driver_repo.enroll(name, embedding, email, id_number)
    
    def get_driver(self, driver_id: int) -> Optional[Driver]:
        """Get driver by ID"""
        return self.driver_repo.get_by_id(driver_id)
    
    def get_driver_by_name(self, name: str) -> Optional[Driver]:
        """Get driver by name"""
        return self.driver_repo.get_by_name(name)
    
    def get_all_drivers(self, active_only: bool = True) -> List[Driver]:
        """Get all drivers"""
        return self.driver_repo.get_all(active_only)
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get all driver embeddings for verification"""
        return self.driver_repo.get_all_embeddings()
    
    def update_driver_status(self, driver_id: int, status: str) -> bool:
        """Update driver status (active/inactive)"""
        return self.driver_repo.update_status(driver_id, status)
    
    def delete_driver(self, driver_id: int) -> bool:
        """Delete a driver (soft delete by setting status to inactive)"""
        return self.update_driver_status(driver_id, 'inactive')
    
    def driver_exists(self, name: str) -> bool:
        """Check if a driver with the given name exists"""
        return self.driver_repo.exists(name)
    
    # ==================== Verification Log Operations ====================
    
    def log_verification(self, log: VerificationLog) -> int:
        """Log a verification attempt"""
        return self.verification_repo.log_verification(log)
    
    def get_recent_logs(self, limit: int = 100) -> List[VerificationLog]:
        """Get recent verification logs"""
        return self.verification_repo.get_recent(limit)
    
    def get_logs_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        """Get verification logs for a specific driver"""
        return self.verification_repo.get_by_driver(driver_id, limit)
    
    def get_unauthorized_attempts(self, limit: int = 50) -> List[VerificationLog]:
        """Get recent unauthorized access attempts"""
        return self.verification_repo.get_unauthorized(limit)
    
    def get_statistics(self) -> dict:
        """Get verification statistics"""
        # Combine statistics from both repositories if needed, 
        # but currently verification repo handles the bulk of stats
        repo_stats = self.verification_repo.get_statistics()
        
        # Add total drivers from driver repo (manual query for now as it's simple)
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) as count FROM drivers WHERE status = 'active'")
        total_drivers = cursor.fetchone()['count']
        conn.close()
        
        repo_stats['total_drivers'] = total_drivers
        return repo_stats

    def get_daily_statistics(self) -> dict:
        """Get verification statistics for the current day"""
        return self.verification_repo.get_daily_statistics()
    
    # ==================== Audit Log Operations ====================
    
    def log_audit(self, action: str, user_email: str, details: str = None, ip_address: str = None):
        """Log an administrative action"""
        return self.audit_repo.log_event(action, user_email, details, ip_address)
    
    def get_audit_logs(self, limit: int = 100) -> List[SystemAuditLog]:
        """Get recent system audit logs"""
        return self.audit_repo.get_recent_logs(limit)
    
    # ==================== Helper Methods (Kept for backward compatibility if needed internally) ====================
    
    def _row_to_driver(self, row: sqlite3.Row) -> Driver:
        """DEPRECATED: Use DriverRepository._row_to_driver instead"""
        return self.driver_repo._row_to_driver(row)
    
    def _row_to_log(self, row: sqlite3.Row) -> VerificationLog:
        """DEPRECATED: Use VerificationRepository._row_to_log instead"""
        return self.verification_repo._row_to_log(row)


if __name__ == "__main__":
    # Test database operations
    print("Testing refactored database operations...")
    
    db = DatabaseManager()
    print(f"Database initialized at: {db.db_path}")
    
    # Test statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total drivers: {stats['total_drivers']}")
    print(f"  Total verifications: {stats['total_verifications']}")
    print(f"  Authorization rate: {stats['authorization_rate']:.2f}%")
