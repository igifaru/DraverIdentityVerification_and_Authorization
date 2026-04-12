"""
Admin Application Factory  (port 5000)
"""

import hashlib
import os

from flask import Flask, render_template
from flask_cors import CORS

from dashboard.routes.admin_api_routes import admin_api_bp
from dashboard.routes.auth_routes import auth_bp
from dashboard.routes.admin_main_routes import admin_main_bp
from utils.config import config


def create_admin_app() -> Flask:
    app = Flask(
        __name__,
        template_folder='templates',
        static_folder='static',
    )

    # ── Stable secret key ────────────────────────────────────────────────
    # Derived from admin credentials so it survives restarts without needing
    # a separate config field. Change admin_password in config.yaml to rotate.
    _seed = f"draver-admin-{config.admin_email}-{config.admin_password}"
    app.secret_key = hashlib.sha256(_seed.encode()).hexdigest()

    app.config['ADMIN_EMAIL']         = config.admin_email
    app.config['ADMIN_PASSWORD']      = config.admin_password
    app.config['VERIFICATION_ENGINE'] = None
    app.config['DRIVER_APP_URL']      = os.environ.get('DRIVER_APP_URL', 'http://localhost:5001')

    CORS(app, resources={r"/api/*": {"origins": "*"}})

    app.register_blueprint(auth_bp)
    app.register_blueprint(admin_main_bp)
    app.register_blueprint(admin_api_bp, url_prefix='/api')

    @app.errorhandler(404)
    def not_found(e):
        return render_template('error.html', code=404,
            title='Page Not Found',
            message="The page you're looking for doesn't exist.",
            link_label='Return to Dashboard'), 404

    @app.errorhandler(500)
    def server_error(e):
        return render_template('error.html', code=500,
            title='System Unavailable',
            message='A critical error occurred.',
            link_label='Try Again'), 500

    return app


if __name__ == '__main__':
    app = create_admin_app()
    print(f"[admin] Dashboard → http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)