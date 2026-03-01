"""
Verification Log Repository
Handles all database operations for the verification_logs table.

Design notes:
  - Uses the same _db() context-manager pattern as the other repositories.
  - All statistics queries run in a single connection to minimise round-trips.
"""

from contextlib import contextmanager
from datetime import datetime
from typing import Callable, List

import psycopg2
import psycopg2.extras

from database.models import VerificationLog


class VerificationRepository:
    """
    Repository for VerificationLog CRUD and stats operations.

    All SQL lives here; no SQL should appear in DatabaseManager or above.
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
        """Context manager: open connection + RealDictCursor, commit on exit."""
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
    def _to_log(row: dict) -> VerificationLog:
        """Map a DB row dict to a VerificationLog dataclass instance."""
        ts = row['timestamp']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return VerificationLog(
            log_id=row['log_id'],
            timestamp=ts,
            driver_id=row['driver_id'],
            driver_name=row['driver_name'],
            similarity_score=float(row['similarity_score'] or 0),
            authorized=bool(row['authorized']),
            processing_time_ms=float(row['processing_time_ms'] or 0),
            image_path=row['image_path'],
            liveness_passed=bool(row['liveness_passed']),
            system_id=row.get('system_id'),
            brightness=row.get('brightness'),
            location=row.get('location'),
            retry_count=row.get('retry_count', 0),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def log_verification(self, log: VerificationLog) -> int:
        """
        Insert a verification event row and return the generated log_id.

        Args:
            log: Populated VerificationLog dataclass (log_id will be ignored).

        Returns:
            The new log_id assigned by PostgreSQL.
        """
        with self._db() as cur:
            cur.execute(
                """
                INSERT INTO verification_logs
                    (driver_id, driver_name, similarity_score, authorized,
                     processing_time_ms, image_path, liveness_passed,
                     system_id, brightness, location, retry_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING log_id
                """,
                (
                    log.driver_id,
                    log.driver_name,
                    log.similarity_score,
                    log.authorized,
                    log.processing_time_ms,
                    log.image_path,
                    log.liveness_passed,
                    log.system_id,
                    log.brightness,
                    log.location,
                    log.retry_count,
                ),
            )
            return cur.fetchone()['log_id']

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_recent(self, limit: int = 100) -> List[VerificationLog]:
        """Return the most recent verification logs, newest first."""
        with self._db() as cur:
            cur.execute(
                "SELECT * FROM verification_logs ORDER BY timestamp DESC LIMIT %s",
                (limit,),
            )
            return [self._to_log(row) for row in cur.fetchall()]

    def get_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        """Return verification logs for a specific driver, newest first."""
        with self._db() as cur:
            cur.execute(
                """
                SELECT * FROM verification_logs
                WHERE driver_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (driver_id, limit),
            )
            return [self._to_log(row) for row in cur.fetchall()]

    def get_unauthorized(self, limit: int = 50) -> List[VerificationLog]:
        """Return the most recent unauthorized access attempts, newest first."""
        with self._db() as cur:
            cur.execute(
                """
                SELECT * FROM verification_logs
                WHERE authorized = FALSE
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            return [self._to_log(row) for row in cur.fetchall()]

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_statistics(self) -> dict:
        """
        Return aggregate statistics across all verification logs.

        Keys: total_verifications, authorized_count, unauthorized_count,
              avg_processing_time_ms, avg_authorized_similarity,
              authorization_rate (percentage, 0–100).
        """
        with self._db() as cur:
            cur.execute("SELECT COUNT(*) AS n FROM verification_logs")
            total = cur.fetchone()['n']

            cur.execute(
                "SELECT COUNT(*) AS n FROM verification_logs WHERE authorized = TRUE"
            )
            authorized = cur.fetchone()['n']

            cur.execute("SELECT AVG(processing_time_ms) AS v FROM verification_logs")
            avg_time = cur.fetchone()['v'] or 0

            cur.execute(
                "SELECT AVG(similarity_score) AS v FROM verification_logs WHERE authorized = TRUE"
            )
            avg_similarity = cur.fetchone()['v'] or 0

        unauthorized = total - authorized
        auth_rate = (authorized / total * 100) if total > 0 else 0

        return {
            'total_verifications':     total,
            'authorized_count':        authorized,
            'unauthorized_count':      unauthorized,
            'avg_processing_time_ms':  float(avg_time),
            'avg_authorized_similarity': float(avg_similarity),
            'authorization_rate':      auth_rate,
        }

    def get_daily_statistics(self) -> dict:
        """
        Return today's verification counts.

        Keys: date, total, authorized, unauthorized.
        """
        today = datetime.now().strftime('%Y-%m-%d')

        with self._db() as cur:
            cur.execute(
                "SELECT COUNT(*) AS n FROM verification_logs WHERE timestamp::date = %s",
                (today,),
            )
            total = cur.fetchone()['n']

            cur.execute(
                """
                SELECT COUNT(*) AS n FROM verification_logs
                WHERE authorized = TRUE AND timestamp::date = %s
                """,
                (today,),
            )
            authorized = cur.fetchone()['n']

        return {
            'date':         today,
            'total':        total,
            'authorized':   authorized,
            'unauthorized': total - authorized,
        }
