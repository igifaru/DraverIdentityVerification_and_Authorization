
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from database.db_manager import DatabaseManager

db = DatabaseManager()
logs = db.get_recent_logs(limit=10)
print(f"Total recent logs: {len(logs)}")
for log in logs:
    print(f"ID: {log.log_id}, Auth: {log.authorized}, Name: {log.driver_name}, Score: {log.similarity_score}")
