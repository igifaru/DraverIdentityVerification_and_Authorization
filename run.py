"""
System Entry Point
Run this script to start the Driver Verification System
"""
import os
import sys

# Add src directory to path so imports work correctly
project_root = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(project_root, 'src')
sys.path.append(src_path)

# Verify environment
try:
    import cv2
    import deepface
    from flask import Flask
except ImportError as e:
    print(f"ERROR: Missing dependency: {e}")
    print("Please verify you have activated the virtual environment and installed requirements.txt")
    sys.exit(1)

# Import and run app
from dashboard.app import create_app

if __name__ == "__main__":
    print(f"Starting Driver Verification System from: {project_root}")
    
    # Check for config
    config_path = os.path.join(project_root, 'config', 'config.yaml')
    if not os.path.exists(config_path):
        print(f"WARNING: Config file not found at {config_path}")

    # Ensure data directories exist
    os.makedirs(os.path.join(project_root, 'data', 'logs'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'database'), exist_ok=True)
    os.makedirs(os.path.join(project_root, 'data', 'alerts'), exist_ok=True)

    # Run application
    # Note: app.run is called here. app.py's __name__ == '__main__' block won't execute when imported.
    # We recreate the run logic here.
    
    flask_app = create_app()
    
    print(f"Admin Login: {flask_app.config['ADMIN_EMAIL']} / {flask_app.config['ADMIN_PASSWORD']}")
    flask_app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
