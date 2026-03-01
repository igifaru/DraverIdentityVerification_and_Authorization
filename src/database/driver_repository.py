"""
Driver Repository
Handles all database operations for the drivers table.

Design notes:
  - Uses psycopg2 with %s parameter placeholders (PostgreSQL style).
  - Biometric embeddings are serialised with pickle and stored as BYTEA.
  - Every public method opens a fresh connection and closes it on exit via
    the _db() context manager – no connection is held between calls.
"""

import pickle
from contextlib import contextmanager
from datetime import datetime
from typing import Callable, List, Optional, Tuple

import numpy as np
import psycopg2
import psycopg2.extras

from database.models import Driver


class DriverRepository:
    """
    Repository for Driver entity CRUD operations.

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
    def _to_driver(row: dict) -> Driver:
        """Map a DB row dict to a Driver dataclass instance."""
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
            status=row['status'],
            photo_path=row.get('photo_path'),
            dob=row.get('dob'),
            gender=row.get('gender'),
            expiry_date=row.get('expiry_date'),
            issue_place=row.get('issue_place'),
        )

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def enroll(
        self,
        name: str,
        embedding: np.ndarray,
        email: Optional[str] = None,
        license_number: Optional[str] = None,
        category: str = 'A',
        photo_path: Optional[str] = None,
        dob: Optional[str] = None,
        gender: Optional[str] = None,
        expiry_date: Optional[str] = None,
        issue_place: Optional[str] = None,
    ) -> int:
        """
        Insert a new driver row and return the generated driver_id.

        Args:
            name:           Legal full name.
            embedding:      FaceNet embedding as a NumPy array.
            email:          Optional contact email.
            license_number: Driving licence number.
            category:       Comma-separated category codes, e.g. "A,B,C".
            photo_path:     Optional path to the saved enrollment photo on disk.
            dob:            Date of birth (ISO YYYY-MM-DD).
            gender:         Sex or gender.
            expiry_date:    License expiry date (ISO YYYY-MM-DD).
            issue_place:    Place of license issue.

        Returns:
            The new driver_id assigned by PostgreSQL.
        """
        embedding_blob = psycopg2.Binary(pickle.dumps(embedding))
        with self._db() as cur:
            cur.execute(
                """
                INSERT INTO drivers (
                    name, license_number, category, biometric_embedding, 
                    email, photo_path, dob, gender, expiry_date, issue_place
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING driver_id
                """,
                (
                    name, license_number, category, embedding_blob, 
                    email, photo_path, dob, gender, expiry_date, issue_place
                ),
            )
            return cur.fetchone()['driver_id']

    def update_status(self, driver_id: int, status: str) -> bool:
        """
        Update the status field for a given driver.

        Returns:
            True if a row was actually updated, False otherwise.
        """
        with self._db() as cur:
            cur.execute(
                "UPDATE drivers SET status = %s WHERE driver_id = %s",
                (status, driver_id),
            )
            return cur.rowcount > 0

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_id(self, driver_id: int) -> Optional[Driver]:
        """Return the Driver with the given ID, or None."""
        with self._db() as cur:
            cur.execute("SELECT * FROM drivers WHERE driver_id = %s", (driver_id,))
            row = cur.fetchone()
        return self._to_driver(row) if row else None

    def get_by_name(self, name: str) -> Optional[Driver]:
        """Return the active Driver with the given name, or None."""
        with self._db() as cur:
            cur.execute(
                "SELECT * FROM drivers WHERE name = %s AND status = 'active'",
                (name,),
            )
            row = cur.fetchone()
        return self._to_driver(row) if row else None

    def get_by_license_number(self, license_number: str) -> Optional[Driver]:
        """Return the active Driver with the given licence number, or None."""
        with self._db() as cur:
            cur.execute(
                "SELECT * FROM drivers WHERE license_number = %s AND status = 'active'",
                (license_number,),
            )
            row = cur.fetchone()
        return self._to_driver(row) if row else None

    def get_all(self, active_only: bool = True) -> List[Driver]:
        """
        Return all drivers ordered by enrollment date (newest first).

        Args:
            active_only: When True, skip inactive/deleted drivers.
        """
        with self._db() as cur:
            if active_only:
                cur.execute(
                    "SELECT * FROM drivers WHERE status = 'active' ORDER BY enrollment_date DESC"
                )
            else:
                cur.execute("SELECT * FROM drivers ORDER BY enrollment_date DESC")
            rows = cur.fetchall()
        return [self._to_driver(row) for row in rows]

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Return (driver_id, name, embedding) tuples for all active drivers.

        Used by the verification engine to load embeddings into memory.
        """
        with self._db() as cur:
            cur.execute(
                """
                SELECT driver_id, name, biometric_embedding
                FROM drivers
                WHERE status = 'active'
                """
            )
            rows = cur.fetchall()

        return [
            (row['driver_id'], row['name'], pickle.loads(bytes(row['biometric_embedding'])))
            for row in rows
        ]

    def exists(self, name: str) -> bool:
        """Return True if an active driver with this name already exists."""
        return self.get_by_name(name) is not None
