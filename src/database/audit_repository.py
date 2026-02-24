"""
Audit Repository
Handles PostgreSQL database operations for system audit logs.
"""
import psycopg2
import psycopg2.extras
from datetime import datetime
from typing import List, Optional
from database.models import SystemAuditLog


class AuditRepository:
    def __init__(self, connection_factory):
        self.connection_factory = connection_factory

    def _get_cursor(self, conn):
        return conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    def log_event(self, action: str, user_email: str,
                  details: Optional[str] = None, ip_address: Optional[str] = None):
        """Insert a new audit log entry"""
        try:
            conn = self.connection_factory()
            cursor = self._get_cursor(conn)
            cursor.execute("""
                INSERT INTO audit_logs (timestamp, action, user_email, details, ip_address)
                VALUES (%s, %s, %s, %s, %s)
            """, (datetime.now(), action, user_email, details, ip_address))
            conn.commit()
            cursor.close()
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
            cursor = self._get_cursor(conn)
            cursor.execute("""
                SELECT * FROM audit_logs
                ORDER BY timestamp DESC
                LIMIT %s
            """, (limit,))
            for row in cursor.fetchall():
                ts = row['timestamp']
                if isinstance(ts, str):
                    ts = datetime.fromisoformat(ts)
                logs.append(SystemAuditLog(
                    audit_id=row['audit_id'],
                    timestamp=ts,
                    action=row['action'],
                    user_email=row['user_email'],
                    details=row['details'],
                    ip_address=row['ip_address']
                ))
            cursor.close()
            conn.close()
        except Exception as e:
            print(f"Error fetching audit logs: {e}")
        return logs
