"""
Audit Repository
Handles all database operations for the system audit log (audit_logs table).
Uses a connection_factory pattern so the repository owns no long-lived connection.
"""

import psycopg2
import psycopg2.extras
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, List, Optional

from database.models import SystemAuditLog


class AuditRepository:
    """
    Repository for SystemAuditLog CRUD operations.

    All methods open a fresh connection for each call and close it immediately
    afterwards, keeping the connection lifetime as short as possible.
    """

    def __init__(self, connection_factory: Callable):
        """
        Args:
            connection_factory: Zero-argument callable that returns an open
                                psycopg2 connection.
        """
        self._connect = connection_factory

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @contextmanager
    def _db(self):
        """Context manager that opens a connection + RealDictCursor and
        commits (or rolls back) on exit."""
        conn = self._connect()
        cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        try:
            yield cursor
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            cursor.close()
            conn.close()

    @staticmethod
    def _row_to_audit_log(row: dict) -> SystemAuditLog:
        """Map a DB row dict to a SystemAuditLog dataclass instance."""
        ts = row['timestamp']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return SystemAuditLog(
            audit_id=row['audit_id'],
            timestamp=ts,
            action=row['action'],
            user_email=row['user_email'],
            details=row['details'],
            ip_address=row['ip_address'],
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log_event(
        self,
        action: str,
        user_email: str,
        details: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> bool:
        """
        Insert a new audit log entry.

        Returns:
            True on success, False if a database error occurred.
        """
        try:
            with self._db() as cur:
                cur.execute(
                    """
                    INSERT INTO audit_logs (timestamp, action, user_email, details, ip_address)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (datetime.now(), action, user_email, details, ip_address),
                )
            return True
        except Exception as exc:
            print(f"[AuditRepository] Error logging event '{action}': {exc}")
            return False

    def get_recent_logs(self, limit: int = 50) -> List[SystemAuditLog]:
        """
        Return the most recent audit log entries, newest first.

        Args:
            limit: Maximum number of rows to return.

        Returns:
            List of SystemAuditLog instances (empty list on error).
        """
        try:
            with self._db() as cur:
                cur.execute(
                    """
                    SELECT * FROM audit_logs
                    ORDER BY timestamp DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                return [self._row_to_audit_log(row) for row in cur.fetchall()]
        except Exception as exc:
            print(f"[AuditRepository] Error fetching audit logs: {exc}")
            return []
