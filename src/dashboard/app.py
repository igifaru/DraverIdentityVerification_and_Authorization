"""
Flask Application Factory
Creates and configures the Flask app used by run.py.

Blueprint layout:
  auth_bp  – /login, /logout
  main_bp  – /, /driver, /video_feed
  api_bp   – /api/…  (JSON endpoints)
"""

import os
import secrets

from flask import Flask, render_template
from flask_cors import CORS

from dashboard.routes.api_routes import api_bp
from dashboard.routes.auth_routes import auth_bp
from dashboard.routes.main_routes import main_bp
from utils.config import config
from verification.verification_engine import VerificationEngine


def create_app() -> Flask:
    """
    Application factory.

    Initialises the VerificationEngine, registers all blueprints, and
    attaches error handlers.  The engine is stored in app.config so that
    every request handler can reach it via current_app.config['VERIFICATION_ENGINE'].

    Returns:
        A fully configured Flask application instance.
    """
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )

    # ------------------------------------------------------------------ #
    # Security configuration                                               #
    # ------------------------------------------------------------------ #
    app.secret_key = secrets.token_hex(32)

    # In a production deployment these should come from environment vars /
    # a secrets manager, not be hardcoded here.
    app.config['ADMIN_EMAIL']    = os.getenv('ADMIN_EMAIL', 'admin@gmail.com')
    app.config['ADMIN_PASSWORD'] = os.getenv('ADMIN_PASSWORD', 'admin123')

    # ------------------------------------------------------------------ #
    # Verification engine                                                  #
    # ------------------------------------------------------------------ #
    engine = _init_engine()
    app.config['VERIFICATION_ENGINE'] = engine

    # ------------------------------------------------------------------ #
    # Extensions & blueprints                                              #
    # ------------------------------------------------------------------ #
    CORS(app)

    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)
    app.register_blueprint(api_bp, url_prefix='/api')

    # ------------------------------------------------------------------ #
    # Error handlers                                                       #
    # ------------------------------------------------------------------ #
    @app.errorhandler(404)
    def not_found(e):
        return render_template('404.html'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('500.html'), 500

    return app


def _init_engine() -> VerificationEngine | None:
    """
    Instantiate the VerificationEngine.

    The engine is created but NOT started; the dashboard operator must press
    "Start System" to begin continuous verification.  This keeps the camera
    free during the login / enrollment workflow.

    Returns:
        Initialised VerificationEngine, or None if initialisation fails.
    """
    try:
        engine = VerificationEngine()
        print("[app] VerificationEngine ready — waiting for manual start.")
        return engine
    except Exception as exc:
        print(f"[app] VerificationEngine init failed: {exc}")
        return None


# ---------------------------------------------------------------------------
# Direct execution — useful for development only
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    app = create_app()
    print(f"[app] Dashboard at http://localhost:5000")
    print(f"[app] Admin: {app.config['ADMIN_EMAIL']} / {app.config['ADMIN_PASSWORD']}")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
