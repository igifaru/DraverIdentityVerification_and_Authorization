"""
System Entry Point
==================
Starts both Flask applications in separate processes:

  Admin Dashboard  →  http://localhost:5000
  Driver Terminal  →  http://localhost:5001/driver

Usage:
    python run.py               # start both (default)
    python run.py --admin-only  # start only the admin dashboard
    python run.py --driver-only # start only the driver terminal

Environment variables:
    ADMIN_PORT      (default 5000)
    DRIVER_PORT     (default 5001)
    DRIVER_APP_URL  (default http://localhost:5001)  — used by admin to proxy
"""

import argparse
import multiprocessing
import os
import sys
import time

# ── Path setup ──────────────────────────────────────────────────────────────
project_root = os.path.dirname(os.path.abspath(__file__))
src_path     = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# ── Dependency check ─────────────────────────────────────────────────────────
try:
    import cv2
    import deepface
    from flask import Flask
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Activate your virtual environment and install requirements.txt")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Process targets
# ---------------------------------------------------------------------------

def run_admin(port: int):
    """Entry point for the admin dashboard process."""
    from src.dashboard.admin_app import create_admin_app
    print(f"\n[admin] Starting Admin Dashboard on http://0.0.0.0:{port}")
    app = create_admin_app()
    print(f"[admin] Login: {app.config['ADMIN_EMAIL']} / {app.config['ADMIN_PASSWORD']}")
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


def run_driver(port: int):
    """Entry point for the driver terminal process."""
    from src.dashboard.driver_app import create_driver_app
    print(f"\n[driver] Starting Driver Terminal on http://0.0.0.0:{port}/driver")
    app = create_driver_app()
    app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Draver Verification System')
    parser.add_argument('--admin-only',  action='store_true', help='Run only the admin dashboard')
    parser.add_argument('--driver-only', action='store_true', help='Run only the driver terminal')
    args = parser.parse_args()

    admin_port  = int(os.environ.get('ADMIN_PORT',  5000))
    driver_port = int(os.environ.get('DRIVER_PORT', 5001))

    for d in ('data/logs', 'data/alerts'):
        os.makedirs(os.path.join(project_root, d), exist_ok=True)

    print("=" * 60)
    print("  DRAVER — IDENTITY VERIFICATION & AUTHORIZATION SYSTEM")
    print("=" * 60)

    if args.admin_only:
        print(f"  Mode: Admin only  →  http://localhost:{admin_port}")
        print("=" * 60)
        run_admin(admin_port)

    elif args.driver_only:
        print(f"  Mode: Driver terminal only  →  http://localhost:{driver_port}/driver")
        print("=" * 60)
        run_driver(driver_port)

    else:
        print(f"  Admin Dashboard  →  http://localhost:{admin_port}")
        print(f"  Driver Terminal  →  http://localhost:{driver_port}/driver")
        print("=" * 60)

        driver_proc = multiprocessing.Process(
            target=run_driver, args=(driver_port,),
            name='driver-app', daemon=True,
        )
        driver_proc.start()

        time.sleep(2)

        admin_proc = multiprocessing.Process(
            target=run_admin, args=(admin_port,),
            name='admin-app', daemon=True,
        )
        admin_proc.start()

        print("\n[run] Both apps running. Press Ctrl+C to stop.\n")

        try:
            driver_proc.join()
            admin_proc.join()
        except KeyboardInterrupt:
            print("\n[run] Shutting down...")
            driver_proc.terminate()
            admin_proc.terminate()
            driver_proc.join(timeout=5)
            admin_proc.join(timeout=5)
            print("[run] All processes stopped.")