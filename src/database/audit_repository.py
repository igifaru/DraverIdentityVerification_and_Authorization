"""
Audit Repository
Handles database operations for system audit logs
"""
import sqlite3
from typing import List, Optional
from datetime import datetime
from database.models import SystemAuditLog

class AuditRepository:
    def __init__(self, connection_factory):
        self.connection_factory = connection_factory

    def log_event(self, action: str, user_email: str, details: Optional[str] = None, ip_address: Optional[str] = None):
        """Insert a new audit log entry"""
        try:
            conn = self.connection_factory()
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO audit_logs (timestamp, action, user_email, details, ip_address)
                VALUES (?, ?, ?, ?, ?)
            ''', (datetime.now().isoformat(), action, user_email, details, ip_address))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error logging audit event: {e}")
            return False

    def get_recent_logs(self, limit: int = 50) -> List[SystemAuditLog]:
        """Fetch recent audit logs"""
        logs = []
        try:
            conn = self.connection_factory()
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM audit_logs 
                ORDER BY timestamp DESC 
                LIMIT ?
            ''', (limit,))
            
            for row in cursor.fetchall():
                logs.append(SystemAuditLog(
                    audit_id=row['audit_id'],
                    timestamp=datetime.fromisoformat(row['timestamp']),
                    action=row['action'],
                    user_email=row['user_email'],
                    details=row['details'],
                    ip_address=row['ip_address']
                ))
            conn.close()
        except Exception as e:
            print(f"Error fetching audit logs: {e}")
        return logs
