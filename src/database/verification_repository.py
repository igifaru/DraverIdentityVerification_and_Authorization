"""
Verification Log Repository
Handles database operations for Verification Log entities
"""
import sqlite3
from datetime import datetime
from typing import List, Optional
from database.models import VerificationLog

class VerificationRepository:
    """Repository for Verification Log entity operations"""
    
    def __init__(self, connection_factory):
        """
        Initialize repository
        
        Args:
            connection_factory: Callable that returns a database connection
        """
        self._get_connection = connection_factory

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

    def log_verification(self, log: VerificationLog) -> int:
        """Log a verification attempt"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
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
            return log_id
        finally:
            conn.close()

    def get_recent(self, limit: int = 100) -> List[VerificationLog]:
        """Get recent verification logs"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM verification_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_log(row) for row in rows]
        finally:
            conn.close()

    def get_by_driver(self, driver_id: int, limit: int = 50) -> List[VerificationLog]:
        """Get verification logs for a specific driver"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM verification_logs 
                WHERE driver_id = ?
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (driver_id, limit))
            
            rows = cursor.fetchall()
            return [self._row_to_log(row) for row in rows]
        finally:
            conn.close()

    def get_unauthorized(self, limit: int = 50) -> List[VerificationLog]:
        """Get recent unauthorized access attempts"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                SELECT * FROM verification_logs 
                WHERE authorized = 0
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (limit,))
            
            rows = cursor.fetchall()
            return [self._row_to_log(row) for row in rows]
        finally:
            conn.close()

    def get_statistics(self) -> dict:
        """Get general verification statistics"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
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
            
            return {
                'total_verifications': total_verifications,
                'authorized_count': authorized_count,
                'unauthorized_count': unauthorized_count,
                'avg_processing_time_ms': avg_processing_time,
                'avg_authorized_similarity': avg_authorized_score,
                'authorization_rate': (authorized_count / total_verifications * 100) if total_verifications > 0 else 0
            }
        finally:
            conn.close()

    def get_daily_statistics(self) -> dict:
        """Get verification statistics for the current day"""
        today = datetime.now().strftime('%Y-%m-%d')
        conn = self._get_connection()
        cursor = conn.cursor()
        
        try:
            # Daily total
            cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE date(timestamp) = ?", (today,))
            daily_total = cursor.fetchone()['count']
            
            # Daily authorized
            cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE authorized = 1 AND date(timestamp) = ?", (today,))
            daily_authorized = cursor.fetchone()['count']
            
            # Daily unauthorized
            cursor.execute("SELECT COUNT(*) as count FROM verification_logs WHERE authorized = 0 AND date(timestamp) = ?", (today,))
            daily_unauthorized = cursor.fetchone()['count']
            
            return {
                'date': today,
                'total': daily_total,
                'authorized': daily_authorized,
                'unauthorized': daily_unauthorized
            }
        finally:
            conn.close()
