"""
Driver Repository
Handles PostgreSQL database operations for Driver entities.
Uses psycopg2 with %s placeholders (not ? as in SQLite).
"""
import pickle
import psycopg2
import psycopg2.extras
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from database.models import Driver


class DriverRepository:
    """Repository for Driver entity operations"""

    def __init__(self, connection_factory):
        """
        Args:
            connection_factory: Callable that returns a psycopg2 connection.
        """
        self._get_connection = connection_factory

    def _get_cursor(self, conn):
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def _row_to_driver(self, row: dict) -> Driver:
        """Convert a dict row to a Driver object."""
        embedding = pickle.loads(bytes(row['biometric_embedding']))
        ts = row['enrollment_date']
        if isinstance(ts, str):
            ts = datetime.fromisoformat(ts)
        return Driver(
            driver_id=row['driver_id'],
            name=row['name'],
            license_number=row.get('license_number'),
            category=row.get('category', 'A'),
            biometric_embedding=embedding,
            enrollment_date=ts,
            email=row.get('email'),
            status=row['status']
        )

    def enroll(self, name: str, embedding: np.ndarray,
               email: str = None, license_number: str = None,
               category: str = 'A') -> int:
        """Enroll a new driver."""
        embedding_blob = psycopg2.Binary(pickle.dumps(embedding))
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("""
                INSERT INTO drivers (name, license_number, category, biometric_embedding, email)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING driver_id
            """, (name, license_number, category, embedding_blob, email))
            driver_id = cursor.fetchone()['driver_id']
            conn.commit()
            return driver_id
        finally:
            cursor.close()
            conn.close()

    def get_by_id(self, driver_id: int) -> Optional[Driver]:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("SELECT * FROM drivers WHERE driver_id = %s", (driver_id,))
            row = cursor.fetchone()
            return self._row_to_driver(row) if row else None
        finally:
            cursor.close()
            conn.close()

    def get_by_name(self, name: str) -> Optional[Driver]:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute(
                "SELECT * FROM drivers WHERE name = %s AND status = 'active'", (name,))
            row = cursor.fetchone()
            return self._row_to_driver(row) if row else None
        finally:
            cursor.close()
            conn.close()

    def get_all(self, active_only: bool = True) -> List[Driver]:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            if active_only:
                cursor.execute("""
                    SELECT * FROM drivers WHERE status = 'active'
                    ORDER BY enrollment_date DESC
                """)
            else:
                cursor.execute("SELECT * FROM drivers ORDER BY enrollment_date DESC")
            rows = cursor.fetchall()
            return [self._row_to_driver(row) for row in rows]
        finally:
            cursor.close()
            conn.close()

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get all active driver embeddings for verification."""
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute("""
                SELECT driver_id, name, biometric_embedding
                FROM drivers
                WHERE status = 'active'
            """)
            rows = cursor.fetchall()
            result = []
            for row in rows:
                embedding = pickle.loads(bytes(row['biometric_embedding']))
                result.append((row['driver_id'], row['name'], embedding))
            return result
        finally:
            cursor.close()
            conn.close()

    def update_status(self, driver_id: int, status: str) -> bool:
        conn = self._get_connection()
        cursor = self._get_cursor(conn)
        try:
            cursor.execute(
                "UPDATE drivers SET status = %s WHERE driver_id = %s",
                (status, driver_id))
            success = cursor.rowcount > 0
            conn.commit()
            return success
        finally:
            cursor.close()
            conn.close()

    def exists(self, name: str) -> bool:
        return self.get_by_name(name) is not None
