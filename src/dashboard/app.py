"""
Flask Dashboard Application
Web-based interface for monitoring driver verification system
"""

from flask import Flask, render_template, Response, jsonify, request, redirect, url_for, session
from flask_cors import CORS
import threading
import time
import os
import secrets

# Import verification engine
from verification.verification_engine import VerificationEngine
from utils.config import config

# Import Blueprints
from dashboard.routes.auth_routes import auth_bp
from dashboard.routes.main_routes import main_bp
from dashboard.routes.api_routes import api_bp

# Global verification engine instance
# In a production app, this might be handled differently (e.g., factory pattern)
engine = None

def create_app():
    """Create and configure Flask application"""
    app = Flask(__name__, 
                template_folder='templates',
                static_folder='static')
    
    # Configuration
    app.secret_key = secrets.token_hex(16)
    app.config['ADMIN_EMAIL'] = "admin@gmail.com"
    app.config['ADMIN_PASSWORD'] = "admin123"
    
    # Initialize Engine
    global engine
    if engine is None:
        try:
            engine = VerificationEngine()
            
            # Start verification in background thread
            # verification_thread = threading.Thread(
            #     target=engine.run_continuous_verification,
            #     kwargs={'show_preview': False},
            #     daemon=True
            # )
            # verification_thread.start()
            print("INFO: Verification engine initialized but NOT started (Waiting for manual trigger)")
            
        except Exception as e:
            print(f"Error initializing verification engine: {e}")
            engine = None

    # Store engine in app config for blueprints to access
    app.config['VERIFICATION_ENGINE'] = engine
    
    # Enable CORS
    CORS(app)
    
    # Register Blueprints
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')
    
    # Error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('404.html'), 404
        
    @app.errorhandler(500)
    def internal_server_error(e):
        return render_template('500.html'), 500
        
    return app

if __name__ == '__main__':
    app = create_app()
    
    # Ensure templates directory exists (it should, but just in case)
    if not os.path.exists('templates'):
        print("WARNING: 'templates' directory not found in current path.")
    
    print(f"Starting dashboard on http://localhost:5000")
    print(f"Admin Login: {app.config['ADMIN_EMAIL']} / {app.config['ADMIN_PASSWORD']}")
    
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
