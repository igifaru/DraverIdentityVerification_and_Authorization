"""
Driver Application Factory  (port 5001)
"""

import secrets

from flask import Flask, render_template
from flask_cors import CORS

from dashboard.routes.driver_api_routes import driver_api_bp
from dashboard.routes.driver_main_routes import driver_main_bp
from utils.config import config
from verification.verification_engine import VerificationEngine


def create_driver_app() -> Flask:
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )

    app.secret_key = secrets.token_hex(32)

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    engine = _init_engine()
    app.config['VERIFICATION_ENGINE'] = engine

    app.register_blueprint(driver_main_bp)
    app.register_blueprint(driver_api_bp, url_prefix='/api')

    @app.errorhandler(404)
    def not_found(e):
        return render_template('error.html', code=404,
            title='Page Not Found',
            message="Endpoint not found.",
            link_label='Driver Terminal'), 404

    return app


def _init_engine() -> VerificationEngine | None:
    import threading
    try:
        engine = VerificationEngine()
        thread = threading.Thread(
            target=engine.run_continuous_verification,
            kwargs={'show_preview': False},
            daemon=True,
        )
        thread.start()
        print("[driver_app] VerificationEngine started in background thread.")
        return engine
    except Exception as exc:
        print(f"[driver_app] VerificationEngine init failed: {exc}")
        return None


if __name__ == '__main__':
    app = create_driver_app()
    print(f"[driver] Terminal → http://localhost:5001/driver")
    app.run(host='0.0.0.0', port=5001, debug=True, use_reloader=False)