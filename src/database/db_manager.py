"""
Database Manager
Manages PostgreSQL database operations for the driver verification system.
Uses psycopg2 directly (no ORM) to keep the interface identical to the old SQLite version.
Repository pattern is preserved.
"""

import psycopg2
import psycopg2.extras
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from urllib.parse import urlparse

from database.models import Driver, VerificationLog, SystemAuditLog
from database.driver_repository import DriverRepository
from database.verification_repository import VerificationRepository
from database.audit_repository import AuditRepository
from utils.config import config


class DatabaseManager:
    """Manages PostgreSQL database operations via repositories"""

    def __init__(self, database_url: str = None):
        """
        Initialize database manager.

        Args:
            database_url: Full PostgreSQL DSN (uses config if not provided).
        """
        self.database_url = database_url or config.database_url
        self._create_tables()

        # Initialize repositories with our connection factory
        self.driver_repo = DriverRepository(self._get_connection)
        self.verification_repo = VerificationRepository(self._get_connection)
        self.audit_repo = AuditRepository(self._get_connection)

    def _get_connection(self) -> psycopg2.extensions.connection:
        """Return a new psycopg2 connection with RealDictCursor as default."""
        conn = psycopg2.connect(self.database_url)
        return conn

    def _get_cursor(self, conn):
        """Return a dict-like cursor (similar to sqlite3.Row)."""
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _create_tables(self):
        """Create database tables if they don't exist."""
        conn = self._get_connection()
        cursor = self._get_cursor(conn)

        # drivers table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS drivers (
                driver_id   SERIAL PRIMARY KEY,
                name        TEXT NOT NULL,
                license_number TEXT,
                category    TEXT NOT NULL DEFAULT 'A',
                biometric_embedding BYTEA NOT NULL,
                enrollment_date TIMESTAMP DEFAULT NOW(),
                email       TEXT,
                status      TEXT DEFAULT 'active'
            )
        """)

        # verification_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS verification_logs (
                log_id              SERIAL PRIMARY KEY,
                timestamp           TIMESTAMP DEFAULT NOW(),
                driver_id           INTEGER REFERENCES drivers(driver_id),
                driver_name         TEXT,
                similarity_score    REAL,
                authorized          BOOLEAN,
                processing_time_ms  REAL,
                image_path          TEXT,
                liveness_passed     BOOLEAN
            )
        """)

        # index on timestamp
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_verification_timestamp
            ON verification_logs(timestamp DESC)
        """)

        # audit_logs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                audit_id    SERIAL PRIMARY KEY,
                timestamp   TIMESTAMP DEFAULT NOW(),
                action      TEXT NOT NULL,
                user_email  TEXT,
                details     TEXT,
                ip_address  TEXT
            )
        """)

        conn.commit()
        cursor.close()
        conn.close()

    # ==================== Driver Operations ====================

    def enroll_driver(self, name: str, embedding: np.ndarray,
                      email: str = None, license_number: str = None,
                      category: str = 'A') -> int:
        """Enroll a new driver in the database"""
        return self.driver_repo.enroll(name, embedding, email, license_number, category)

    def get_driver(self, driver_id: int) -> Optional[Driver]:
        return self.driver_repo.get_by_id(driver_id)

    def get_driver_by_name(self, name: str) -> Optional[Driver]:
        return self.driver_repo.get_by_name(name)

    def get_all_drivers(self, active_only: bool = True) -> List[Driver]:
        return self.driver_repo.get_all(active_only)

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        return self.driver_repo.get_all_embeddings()

    def update_driver_status(self, driver_id: int, status: str) -> bool:
        return self.driver_repo.update_status(driver_id, status)

    def delete_driver(self, driver_id: int) -> bool:
        return self.update_driver_status(driver_id, 'inactive')

    def driver_exists(self, name: str) -> bool:
        return self.driver_repo.exists(name)

    # ==================== Verification Log Operations ====================

    def log_verification(self, log: VerificationLog) -> int:
        return self.verification_repo.log_verification(log)

    def get_recent_logs(self, limit: int = 100) -> List[VerificationLog]:
        return self.verification_repo.get_recent(limit)

    def get_logs_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        return self.verification_repo.get_by_driver(driver_id, limit)

    def get_unauthorized_attempts(self, limit: int = 50) -> List[VerificationLog]:
        return self.verification_repo.get_unauthorized(limit)

    def get_statistics(self) -> dict:
        repo_stats = self.verification_repo.get_statistics()
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        cursor.execute("SELECT COUNT(*) AS count FROM drivers WHERE status = 'active'")
        total_drivers = cursor.fetchone()['count']
        cursor.close()
        conn.close()
        repo_stats['total_drivers'] = total_drivers
        return repo_stats

    def get_daily_statistics(self) -> dict:
        return self.verification_repo.get_daily_statistics()

    # ==================== Audit Log Operations ====================

    def log_audit(self, action: str, user_email: str,
                  details: str = None, ip_address: str = None):
        return self.audit_repo.log_event(action, user_email, details, ip_address)

    def get_audit_logs(self, limit: int = 100) -> List[SystemAuditLog]:
        return self.audit_repo.get_recent_logs(limit)


if __name__ == "__main__":
    print("Testing PostgreSQL database operations...")
    db = DatabaseManager()
    print(f"Connected to: {config.database_url}")
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total drivers: {stats.get('total_drivers', 0)}")
    print(f"  Total verifications: {stats.get('total_verifications', 0)}")
    print(f"  Authorization rate: {stats.get('authorization_rate', 0):.2f}%")
