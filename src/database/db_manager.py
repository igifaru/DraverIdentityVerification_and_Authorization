"""
Database Manager
Central façade for all PostgreSQL database operations.

Architecture:
  - DatabaseManager owns the connection-factory (a callable wrapping psycopg2.connect).
  - Three specialised repositories handle their own SQL:
      DriverRepository       → drivers table
      VerificationRepository → verification_logs table
      AuditRepository        → audit_logs table
  - Public methods on DatabaseManager delegate straight to the repositories,
    giving the rest of the application a single, stable interface while keeping
    SQL isolated and testable.

Connection strategy:
  - Each repository method opens and closes its own short-lived connection.
  - psycopg2 connections are not thread-safe, so this approach avoids sharing
    a connection across the Flask request/thread boundary.
"""

import psycopg2
import psycopg2.extras
import numpy as np
from contextlib import contextmanager
from typing import Callable, List, Optional, Tuple

from database.models import Driver, VerificationLog, SystemAuditLog
from database.driver_repository import DriverRepository
from database.verification_repository import VerificationRepository
from database.audit_repository import AuditRepository
from utils.config import config


class DatabaseManager:
    """
    Single entry-point for database access used by the rest of the application.

    Usage:
        db = DatabaseManager()
        driver_id = db.enroll_driver("Jane Smith", embedding, category="B,C")
        stats = db.get_statistics()
    """

    def __init__(self, database_url: Optional[str] = None):
        """
        Args:
            database_url: PostgreSQL DSN (falls back to config.database_url).
        """
        self._database_url: str = database_url or config.database_url
        self._ensure_schema()

        # Each repository receives the connection factory as a dependency.
        factory: Callable = self._new_connection
        self.driver_repo       = DriverRepository(factory)
        self.verification_repo = VerificationRepository(factory)
        self.audit_repo        = AuditRepository(factory)

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _new_connection(self) -> psycopg2.extensions.connection:
        """Open and return a new psycopg2 connection."""
        return psycopg2.connect(self._database_url)

    @contextmanager
    def _db(self):
        """Context manager: open a RealDictCursor connection, commit on exit."""
        conn = self._new_connection()
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cur
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cur.close()
            conn.close()

    # ------------------------------------------------------------------
    # Schema initialisation (runs once at startup)
    # ------------------------------------------------------------------

    def _ensure_schema(self) -> None:
        """Create all tables and indexes if they do not already exist."""
        with self._db() as cur:

            cur.execute("""
                CREATE TABLE IF NOT EXISTS drivers (
                    driver_id           SERIAL PRIMARY KEY,
                    name                TEXT    NOT NULL,
                    license_number      TEXT,
                    category            TEXT    NOT NULL DEFAULT 'A',
                    biometric_embedding BYTEA   NOT NULL,
                    enrollment_date     TIMESTAMP DEFAULT NOW(),
                    email               TEXT,
                    status              TEXT    DEFAULT 'active',
                    photo_path          TEXT
                )
            """)

            # Ensure photo_path column exists on older databases
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_name = 'drivers' AND column_name = 'photo_path'
                    ) THEN
                        ALTER TABLE drivers ADD COLUMN photo_path TEXT;
                    END IF;
                END $$;
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS verification_logs (
                    log_id             SERIAL PRIMARY KEY,
                    timestamp          TIMESTAMP DEFAULT NOW(),
                    driver_id          INTEGER REFERENCES drivers(driver_id),
                    driver_name        TEXT,
                    similarity_score   REAL,
                    authorized         BOOLEAN,
                    processing_time_ms REAL,
                    image_path         TEXT,
                    liveness_passed    BOOLEAN,
                    system_id          TEXT,
                    brightness         REAL,
                    location           TEXT,
                    retry_count        INTEGER DEFAULT 0
                )
            """)

            # Ensure new incident columns exist for older databases
            cur.execute("""
                DO $$
                BEGIN
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'verification_logs' AND column_name = 'system_id') THEN
                        ALTER TABLE verification_logs ADD COLUMN system_id TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'verification_logs' AND column_name = 'brightness') THEN
                        ALTER TABLE verification_logs ADD COLUMN brightness REAL;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'verification_logs' AND column_name = 'location') THEN
                        ALTER TABLE verification_logs ADD COLUMN location TEXT;
                    END IF;
                    IF NOT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'verification_logs' AND column_name = 'retry_count') THEN
                        ALTER TABLE verification_logs ADD COLUMN retry_count INTEGER DEFAULT 0;
                    END IF;
                END $$;
            """)

            cur.execute("""
                CREATE TABLE IF NOT EXISTS audit_logs (
                    audit_id   SERIAL PRIMARY KEY,
                    timestamp  TIMESTAMP DEFAULT NOW(),
                    action     TEXT NOT NULL,
                    user_email TEXT,
                    details    TEXT,
                    ip_address TEXT
                )
            """)

    # ==================================================================
    # Driver operations  (delegates to DriverRepository)
    # ==================================================================

    def enroll_driver(
        self,
        name: str,
        embedding: np.ndarray,
        email: Optional[str] = None,
        license_number: Optional[str] = None,
        category: str = 'A',
        photo_path: Optional[str] = None,
    ) -> int:
        """Enroll a new driver and return the assigned driver_id."""
        return self.driver_repo.enroll(name, embedding, email, license_number, category, photo_path)

    def get_driver(self, driver_id: int) -> Optional[Driver]:
        """Return a Driver by primary key, or None."""
        return self.driver_repo.get_by_id(driver_id)

    def get_driver_by_name(self, name: str) -> Optional[Driver]:
        """Return the active Driver with the given name, or None."""
        return self.driver_repo.get_by_name(name)

    def get_driver_by_license(self, license_number: str) -> Optional[Driver]:
        """Return the active Driver with the given licence number, or None."""
        return self.driver_repo.get_by_license_number(license_number)

    def get_all_drivers(self, active_only: bool = True) -> List[Driver]:
        """Return all drivers, optionally filtered to active only."""
        return self.driver_repo.get_all(active_only)

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Return (driver_id, name, embedding) for all active drivers."""
        return self.driver_repo.get_all_embeddings()

    def update_driver_status(self, driver_id: int, status: str) -> bool:
        """Update a driver's status field. Returns True on success."""
        return self.driver_repo.update_status(driver_id, status)

    def delete_driver(self, driver_id: int) -> bool:
        """Soft-delete a driver by setting status = 'inactive'."""
        return self.update_driver_status(driver_id, 'inactive')

    def driver_exists(self, name: str) -> bool:
        """Return True if an active driver with this name already exists."""
        return self.driver_repo.exists(name)

    # ==================================================================
    # Verification log operations  (delegates to VerificationRepository)
    # ==================================================================

    def log_verification(self, log: VerificationLog) -> int:
        """Persist a verification event and return its log_id."""
        return self.verification_repo.log_verification(log)

    def get_recent_logs(self, limit: int = 100) -> List[VerificationLog]:
        """Return the most recent verification logs (newest first)."""
        return self.verification_repo.get_recent(limit)

    def get_logs_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        """Return recent verification logs for a specific driver."""
        return self.verification_repo.get_by_driver(driver_id, limit)

    def get_unauthorized_attempts(self, limit: int = 50) -> List[VerificationLog]:
        """Return recent unauthorized access attempts."""
        return self.verification_repo.get_unauthorized(limit)

    def get_statistics(self) -> dict:
        """Return overall verification statistics plus total active driver count."""
        stats = self.verification_repo.get_statistics()
        with self._db() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM drivers WHERE status = 'active'")
            stats['total_drivers'] = cur.fetchone()['n']
        return stats

    def get_daily_statistics(self) -> dict:
        """Return today's verification counts."""
        return self.verification_repo.get_daily_statistics()

    # ==================================================================
    # Audit log operations  (delegates to AuditRepository)
    # ==================================================================

    def log_audit(
        self,
        action: str,
        user_email: str,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """Record an administrative action in the audit log."""
        return self.audit_repo.log_event(action, user_email, details, ip_address)

    def get_audit_logs(self, limit: int = 100, action_filter: str = None) -> List[SystemAuditLog]:
        """Return the most recent audit log entries, optionally filtered by action type."""
        return self.audit_repo.get_recent_logs(limit, action_filter=action_filter)

    def delete_audit_log(self, audit_id: int) -> bool:
        """Permanently delete a single audit log entry by ID."""
        return self.audit_repo.delete_by_id(audit_id)

    def clear_audit_logs(self) -> int:
        """Permanently delete all audit log entries. Returns number of rows deleted."""
        return self.audit_repo.clear_all()


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python -m database.db_manager)
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print("Connecting to PostgreSQL …")
    db = DatabaseManager()
    stats = db.get_statistics()
    print(f"  Active drivers    : {stats.get('total_drivers', 0)}")
    print(f"  Total verifications: {stats.get('total_verifications', 0)}")
    print(f"  Authorization rate : {stats.get('authorization_rate', 0):.1f}%")
