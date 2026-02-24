"""
Verification Log Repository
Handles PostgreSQL database operations for Verification Log entities.
"""
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import List
from database.models import VerificationLog


class VerificationRepository:
    """Repository for Verification Log entity operations"""

    def __init__(self, connection_factory):
        self._get_connection = connection_factory

    def _get_cursor(self, conn):
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _row_to_log(self, row: dict) -> VerificationLog:
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
            liveness_passed=bool(row['liveness_passed'])
        )

    def log_verification(self, log: VerificationLog) -> int:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("""
                INSERT INTO verification_logs
                (driver_id, driver_name, similarity_score, authorized,
                 processing_time_ms, image_path, liveness_passed)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING log_id
            """, (
                log.driver_id,
                log.driver_name,
                log.similarity_score,
                log.authorized,
                log.processing_time_ms,
                log.image_path,
                log.liveness_passed
            ))
            log_id = cursor.fetchone()['log_id']
            conn.commit()
            return log_id
        finally:
            cursor.close()
            conn.close()

    def get_recent(self, limit: int = 100) -> List[VerificationLog]:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("""
                SELECT * FROM verification_logs
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            return [self._row_to_log(row) for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

    def get_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("""
                SELECT * FROM verification_logs
                WHERE driver_id = %s
                ORDER BY timestamp DESC
                LIMIT %s
            """, (driver_id, limit))
            return [self._row_to_log(row) for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

    def get_unauthorized(self, limit: int = 50) -> List[VerificationLog]:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("""
                SELECT * FROM verification_logs
                WHERE authorized = FALSE
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            return [self._row_to_log(row) for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()

    def get_statistics(self) -> dict:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("SELECT COUNT(*) AS count FROM verification_logs")
            total_verifications = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) AS count FROM verification_logs WHERE authorized = TRUE")
            authorized_count = cursor.fetchone()['count']

            cursor.execute("SELECT COUNT(*) AS count FROM verification_logs WHERE authorized = FALSE")
            unauthorized_count = cursor.fetchone()['count']

            cursor.execute("SELECT AVG(processing_time_ms) AS avg_time FROM verification_logs")
            avg_processing_time = cursor.fetchone()['avg_time'] or 0

            cursor.execute("""
                SELECT AVG(similarity_score) AS avg_score
                FROM verification_logs
                WHERE authorized = TRUE
            """)
            avg_authorized_score = cursor.fetchone()['avg_score'] or 0

            return {
                'total_verifications': total_verifications,
                'authorized_count': authorized_count,
                'unauthorized_count': unauthorized_count,
                'avg_processing_time_ms': float(avg_processing_time),
                'avg_authorized_similarity': float(avg_authorized_score),
                'authorization_rate': (authorized_count / total_verifications * 100)
                                       if total_verifications > 0 else 0
            }
        finally:
            cursor.close()
            conn.close()

    def get_daily_statistics(self) -> dict:
        today = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            # PostgreSQL uses DATE() function
            cursor.execute(
                "SELECT COUNT(*) AS count FROM verification_logs WHERE timestamp::date = %s",
                (today,))
            daily_total = cursor.fetchone()['count']

            cursor.execute(
                "SELECT COUNT(*) AS count FROM verification_logs WHERE authorized = TRUE AND timestamp::date = %s",
                (today,))
            daily_authorized = cursor.fetchone()['count']

            cursor.execute(
                "SELECT COUNT(*) AS count FROM verification_logs WHERE authorized = FALSE AND timestamp::date = %s",
                (today,))
            daily_unauthorized = cursor.fetchone()['count']

            return {
                'date': today,
                'total': daily_total,
                'authorized': daily_authorized,
                'unauthorized': daily_unauthorized
            }
        finally:
            cursor.close()
            conn.close()
