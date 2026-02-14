"""
Driver Repository
Handles database operations for Driver entities
"""
import sqlite3
import pickle
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple
from database.models import Driver

class DriverRepository:
    """Repository for Driver entity operations"""
    
    def __init__(self, connection_factory):
        """
        Initialize repository
        
        Args:
            connection_factory: Callable that returns a database connection
        """
        self._get_connection = connection_factory

    def _row_to_driver(self, row: sqlite3.Row) -> Driver:
        """Convert database row to Driver object"""
        embedding = pickle.loads(row['biometric_embedding'])
        
        return Driver(
            driver_id=row['driver_id'],
            name=row['name'],
            id_number=row.get('id_number'),
            biometric_embedding=embedding,
            enrollment_date=datetime.fromisoformat(row['enrollment_date']),
            email=row['email'],
            status=row['status']
        )

    def enroll(self, name: str, embedding: np.ndarray, email: str = None, id_number: str = None) -> int:
        """Enroll a new driver"""
        embedding_blob = pickle.dumps(embedding)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO drivers (name, biometric_embedding, email, id_number)
                VALUES (?, ?, ?, ?)
            """, (name, embedding_blob, email, id_number))
            
            driver_id = cursor.lastrowid
            conn.commit()
            return driver_id
        finally:
            conn.close()

    def get_by_id(self, driver_id: int) -> Optional[Driver]:
        """Get driver by ID"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM drivers WHERE driver_id = ?", (driver_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_driver(row)
            return None
        finally:
            conn.close()

    def get_by_name(self, name: str) -> Optional[Driver]:
        """Get driver by name"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("SELECT * FROM drivers WHERE name = ? AND status = 'active'", (name,))
            row = cursor.fetchone()
            if row:
                return self._row_to_driver(row)
            return None
        finally:
            conn.close()

    def get_all(self, active_only: bool = True) -> List[Driver]:
        """Get all drivers"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
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
            conn.close()

    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get all driver embeddings for verification"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT driver_id, name, biometric_embedding 
                FROM drivers 
                WHERE status = 'active'
            """)
            
            rows = cursor.fetchall()
            embeddings = []
            for row in rows:
                embedding = pickle.loads(row['biometric_embedding'])
                embeddings.append((row['driver_id'], row['name'], embedding))
            
            return embeddings
        finally:
            conn.close()

    def update_status(self, driver_id: int, status: str) -> bool:
        """Update driver status"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("UPDATE drivers SET status = ? WHERE driver_id = ?", (status, driver_id))
            success = cursor.rowcount > 0
            conn.commit()
            return success
        finally:
            conn.close()
            
    def exists(self, name: str) -> bool:
        """Check if a driver with the given name exists"""
        return self.get_by_name(name) is not None
