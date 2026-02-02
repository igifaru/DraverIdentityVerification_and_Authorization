"""
Database Manager
Handles all SQLite database operations for driver enrollment and verification logging
"""

import sqlite3
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple
from database.models import Driver, VerificationLog
from utils.config import config


class DatabaseManager:
    """Manages SQLite database operations"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database manager
        
        Args:
            db_path: Path to SQLite database file (uses config if not provided)
        """
        self.db_path = db_path or config.database_path
        self._ensure_database_exists()
        self._create_tables()
    
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
        
        conn.commit()
        conn.close()
    
    # ==================== Driver Operations ====================
    
    def enroll_driver(self, name: str, embedding: np.ndarray, email: str = None, id_number: str = None) -> int:
        """
        Enroll a new driver in the database
        
        Args:
            name: Driver's name
            embedding: 128-dimensional FaceNet embedding
            email: Optional email address
            id_number: Optional identification number
            
        Returns:
            driver_id of the newly enrolled driver
        """
        # Serialize embedding
        embedding_blob = pickle.dumps(embedding)
        
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO drivers (name, biometric_embedding, email, id_number)
            VALUES (?, ?, ?, ?)
        """, (name, embedding_blob, email, id_number))
        
        driver_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return driver_id
    
    def get_driver(self, driver_id: int) -> Optional[Driver]:
        """
        Get driver by ID
        
        Args:
            driver_id: Driver ID
            
        Returns:
            Driver object or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM drivers WHERE driver_id = ?
        """, (driver_id,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_driver(row)
        return None
    
    def get_driver_by_name(self, name: str) -> Optional[Driver]:
        """
        Get driver by name
        
        Args:
            name: Driver's name
            
        Returns:
            Driver object or None if not found
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM drivers WHERE name = ? AND status = 'active'
        """, (name,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return self._row_to_driver(row)
        return None
    
    def get_all_drivers(self, active_only: bool = True) -> List[Driver]:
        """
        Get all drivers
        
        Args:
            active_only: If True, only return active drivers
            
        Returns:
            List of Driver objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        if active_only:
            cursor.execute("""
                SELECT * FROM drivers WHERE status = 'active'
                ORDER BY enrollment_date DESC
            """)
        else:
            cursor.execute("""
                SELECT * FROM drivers ORDER BY enrollment_date DESC
            """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_driver(row) for row in rows]
    
    def get_all_embeddings(self) -> List[Tuple[int, str, np.ndarray]]:
        """
        Get all driver embeddings for verification
        
        Returns:
            List of tuples (driver_id, name, embedding)
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT driver_id, name, biometric_embedding 
            FROM drivers 
            WHERE status = 'active'
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        embeddings = []
        for row in rows:
            embedding = pickle.loads(row['biometric_embedding'])
            embeddings.append((row['driver_id'], row['name'], embedding))
        
        return embeddings
    
    def update_driver_status(self, driver_id: int, status: str) -> bool:
        """
        Update driver status (active/inactive)
        
        Args:
            driver_id: Driver ID
            status: New status ('active' or 'inactive')
            
        Returns:
            True if successful, False otherwise
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE drivers SET status = ? WHERE driver_id = ?
        """, (status, driver_id))
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_driver(self, driver_id: int) -> bool:
        """
        Delete a driver (soft delete by setting status to inactive)
        
        Args:
            driver_id: Driver ID
            
        Returns:
            True if successful, False otherwise
        """
        return self.update_driver_status(driver_id, 'inactive')
    
    def driver_exists(self, name: str) -> bool:
        """
        Check if a driver with the given name exists
        
        Args:
            name: Driver's name
            
        Returns:
            True if driver exists, False otherwise
        """
        driver = self.get_driver_by_name(name)
        return driver is not None
    
    # ==================== Verification Log Operations ====================
    
    def log_verification(self, log: VerificationLog) -> int:
        """
        Log a verification attempt
        
        Args:
            log: VerificationLog object
            
        Returns:
            log_id of the created log entry
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO verification_logs 
            (driver_id, driver_name, similarity_score, authorized, 
             processing_time_ms, image_path, liveness_passed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            log.driver_id,
            log.driver_name,
            log.similarity_score,
            log.authorized,
            log.processing_time_ms,
            log.image_path,
            log.liveness_passed
        ))
        
        log_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return log_id
    
    def get_recent_logs(self, limit: int = 100) -> List[VerificationLog]:
        """
        Get recent verification logs
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of VerificationLog objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM verification_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_log(row) for row in rows]
    
    def get_logs_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        """
        Get verification logs for a specific driver
        
        Args:
            driver_id: Driver ID
            limit: Maximum number of logs to return
            
        Returns:
            List of VerificationLog objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM verification_logs 
            WHERE driver_id = ?
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (driver_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_log(row) for row in rows]
    
    def get_unauthorized_attempts(self, limit: int = 50) -> List[VerificationLog]:
        """
        Get recent unauthorized access attempts
        
        Args:
            limit: Maximum number of logs to return
            
        Returns:
            List of VerificationLog objects
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM verification_logs 
            WHERE authorized = 0
            ORDER BY timestamp DESC 
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [self._row_to_log(row) for row in rows]
    
    def get_statistics(self) -> dict:
        """
        Get verification statistics
        
        Returns:
            Dictionary with statistics
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Total drivers
        cursor.execute("SELECT COUNT(*) as count FROM drivers WHERE status = 'active'")
        total_drivers = cursor.fetchone()['count']
        
        # Total verifications
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs")
        total_verifications = cursor.fetchone()['count']
        
        # Authorized verifications
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE authorized = 1")
        authorized_count = cursor.fetchone()['count']
        
        # Unauthorized attempts
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE authorized = 0")
        unauthorized_count = cursor.fetchone()['count']
        
        # Average processing time
        cursor.execute("SELECT AVG(processing_time_ms) as avg_time FROM verification_logs")
        avg_processing_time = cursor.fetchone()['avg_time'] or 0
        
        # Average similarity score for authorized
        cursor.execute("""
            SELECT AVG(similarity_score) as avg_score 
            FROM verification_logs 
            WHERE authorized = 1
        """)
        avg_authorized_score = cursor.fetchone()['avg_score'] or 0
        
        conn.close()
        
        return {
            'total_drivers': total_drivers,
            'total_verifications': total_verifications,
            'authorized_count': authorized_count,
            'unauthorized_count': unauthorized_count,
            'avg_processing_time_ms': avg_processing_time,
            'avg_authorized_similarity': avg_authorized_score,
            'authorization_rate': (authorized_count / total_verifications * 100) if total_verifications > 0 else 0
        }

    def get_daily_statistics(self) -> dict:
        """
        Get verification statistics for the current day
        
        Returns:
            Dictionary with daily statistics
        """
        today = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Daily total
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE date(timestamp) = ?", (today,))
        daily_total = cursor.fetchone()['count']
        
        # Daily authorized
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE authorized = 1 AND date(timestamp) = ?", (today,))
        daily_authorized = cursor.fetchone()['count']
        
        # Daily unauthorized
        cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE authorized = 0 AND date(timestamp) = ?", (today,))
        daily_unauthorized = cursor.fetchone()['count']
        
        conn.close()
        
        return {
            'date': today,
            'total': daily_total,
            'authorized': daily_authorized,
            'unauthorized': daily_unauthorized
        }
    
    # ==================== Helper Methods ====================
    
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
    
    def _row_to_log(self, row: sqlite3.Row) -> VerificationLog:
        """Convert database row to VerificationLog object"""
        return VerificationLog(
            log_id=row['log_id'],
            timestamp=datetime.fromisoformat(row['timestamp']),
            driver_id=row['driver_id'],
            driver_name=row['driver_name'],
            similarity_score=row['similarity_score'],
            authorized=bool(row['authorized']),
            processing_time_ms=row['processing_time_ms'],
            image_path=row['image_path'],
            liveness_passed=bool(row['liveness_passed'])
        )


if __name__ == "__main__":
    # Test database operations
    print("Testing database operations...")
    
    db = DatabaseManager()
    print(f"Database initialized at: {db.db_path}")
    
    # Test statistics
    stats = db.get_statistics()
    print(f"\nDatabase Statistics:")
    print(f"  Total drivers: {stats['total_drivers']}")
    print(f"  Total verifications: {stats['total_verifications']}")
    print(f"  Authorization rate: {stats['authorization_rate']:.2f}%")
