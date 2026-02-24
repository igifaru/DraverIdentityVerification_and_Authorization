#!/usr/bin/env python3
"""
migrate_to_postgres.py
Migrates existing data from the SQLite database to PostgreSQL.

Usage:
    python migrate_to_postgres.py

Requires:
    - SQLite DB at data/database/drivers.db
    - PostgreSQL running and credentials set in config/config.yaml (or env vars)
    - psycopg2 installed:  pip install psycopg2-binary
"""

import sqlite3
import pickle
import sys
import os

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import psycopg2
import psycopg2.extras
from utils.config import config

SQLITE_PATH = os.path.join(os.path.dirname(__file__), 'data', 'database', 'drivers.db')


def get_pg_conn():
    return psycopg2.connect(config.database_url)


def get_sqlite_conn():
    conn = sqlite3.connect(SQLITE_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def migrate_drivers(sqlite_cur, pg_conn):
    pg_cur = pg_conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)

    sqlite_cur.execute("SELECT * FROM drivers")
    rows = sqlite_cur.fetchall()
    migrated = 0

    for row in rows:
        # Check for new column 'category' (may not exist in old SQLite schema)
        try:
            category = row['category']
        except IndexError:
            category = 'A'

        try:
            license_number = row['license_number']
        except IndexError:
            license_number = None

        blob = bytes(row['biometric_embedding'])
        pg_cur.execute("""
            INSERT INTO drivers
                (driver_id, name, license_number, category, biometric_embedding,
                 enrollment_date, email, status)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (driver_id) DO NOTHING
        """, (
            row['driver_id'],
            row['name'],
            license_number,
            category,
            psycopg2.Binary(blob),
            row['enrollment_date'],
            row['email'],
            row['status'],
        ))
        migrated += 1

    # Reset sequence so next INSERT gets correct ID
    pg_cur.execute("""
        SELECT setval('drivers_driver_id_seq', COALESCE(MAX(driver_id), 1))
        FROM drivers
    """)

    pg_conn.commit()
    pg_cur.close()
    print(f"  ✅ Drivers migrated: {migrated}")


def migrate_verification_logs(sqlite_cur, pg_conn):
    pg_cur = pg_conn.cursor()

    sqlite_cur.execute("SELECT * FROM verification_logs")
    rows = sqlite_cur.fetchall()
    migrated = 0

    for row in rows:
        pg_cur.execute("""
            INSERT INTO verification_logs
                (log_id, timestamp, driver_id, driver_name, similarity_score,
                 authorized, processing_time_ms, image_path, liveness_passed)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (log_id) DO NOTHING
        """, (
            row['log_id'],
            row['timestamp'],
            row['driver_id'],
            row['driver_name'],
            row['similarity_score'],
            bool(row['authorized']),
            row['processing_time_ms'],
            row['image_path'],
            bool(row['liveness_passed']),
        ))
        migrated += 1

    pg_cur.execute("""
        SELECT setval('verification_logs_log_id_seq', COALESCE(MAX(log_id), 1))
        FROM verification_logs
    """)

    pg_conn.commit()
    pg_cur.close()
    print(f"  ✅ Verification logs migrated: {migrated}")


def migrate_audit_logs(sqlite_cur, pg_conn):
    pg_cur = pg_conn.cursor()

    # audit_logs may not exist in older SQLite schemas
    try:
        sqlite_cur.execute("SELECT * FROM audit_logs")
        rows = sqlite_cur.fetchall()
    except Exception:
        print("  ⚠️  No audit_logs table found in SQLite – skipping.")
        return

    migrated = 0
    for row in rows:
        pg_cur.execute("""
            INSERT INTO audit_logs
                (audit_id, timestamp, action, user_email, details, ip_address)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (audit_id) DO NOTHING
        """, (
            row['audit_id'],
            row['timestamp'],
            row['action'],
            row['user_email'],
            row['details'],
            row['ip_address'],
        ))
        migrated += 1

    pg_cur.execute("""
        SELECT setval('audit_logs_audit_id_seq', COALESCE(MAX(audit_id), 1))
        FROM audit_logs
    """)

    pg_conn.commit()
    pg_cur.close()
    print(f"  ✅ Audit logs migrated: {migrated}")


def main():
    if not os.path.exists(SQLITE_PATH):
        print(f"❌ SQLite database not found at: {SQLITE_PATH}")
        sys.exit(1)

    print(f"📦 Source SQLite: {SQLITE_PATH}")
    print(f"🐘 Target PostgreSQL: {config.database_url}\n")

    # First run _create_tables via DatabaseManager to ensure schema exists
    print("Creating PostgreSQL schema...")
    from database.db_manager import DatabaseManager
    DatabaseManager()  # this triggers _create_tables()
    print("  ✅ Schema ready\n")

    sqlite_conn = get_sqlite_conn()
    sqlite_cur = sqlite_conn.cursor()

    pg_conn = get_pg_conn()

    print("Migrating data...")
    migrate_drivers(sqlite_cur, pg_conn)
    migrate_verification_logs(sqlite_cur, pg_conn)
    migrate_audit_logs(sqlite_cur, pg_conn)

    sqlite_conn.close()
    pg_conn.close()

    print("\n🎉 Migration complete! Your PostgreSQL database is ready.")
    print("   You can now run: python run.py")


if __name__ == "__main__":
    main()
