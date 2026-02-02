import sqlite3
import os

db_path = 'data/verification.db'

if os.path.exists(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if id_number column exists
    cursor.execute("PRAGMA table_info(drivers)")
    columns = [col[1] for col in cursor.fetchall()]
    
    if 'id_number' not in columns and columns:
        print("Adding id_number column to drivers table...")
        cursor.execute("ALTER TABLE drivers ADD COLUMN id_number TEXT")
        conn.commit()
        print("Column added successfully.")
    else:
        print("id_number column already exists or table does not exist yet.")
        
    conn.close()
else:
    print(f"Database file {db_path} not found.")
