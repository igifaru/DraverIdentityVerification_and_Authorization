"""
Authentication Routes
Handles admin login and logout.

The login credentials are stored in app.config (set from environment variables
in app.py).  This is a prototype-grade implementation — a production deployment
should replace it with hashed passwords and a proper user store.
"""

from functools import wraps

from flask import (
    Blueprint, current_app, flash, redirect,
    render_template, request, session, url_for,
)

auth_bp = Blueprint('auth', __name__)


# ---------------------------------------------------------------------------
# Decorator
# ---------------------------------------------------------------------------

def login_required(f):
    """Route decorator that redirects unauthenticated requests to /login."""
    @wraps(f)
    def _wrapper(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return _wrapper


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Render the login form (GET) or process submitted credentials (POST)."""
    engine = current_app.config['VERIFICATION_ENGINE']

    if request.method == 'POST':
        email    = request.form.get('email', '').strip()
        password = request.form.get('password', '')

        credentials_ok = (
            email    == current_app.config['ADMIN_EMAIL'] and
            password == current_app.config['ADMIN_PASSWORD']
        )

        if credentials_ok:
            session['logged_in']  = True
            session['user_email'] = email

            if engine:
                engine.db.log_audit(
                    "LOGIN", email,
                    "Admin logged in successfully",
                    request.remote_addr,
                )

            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.index'))

        # Failed attempt
        if engine:
            engine.db.log_audit(
                "LOGIN_FAILURE", email or "UNKNOWN",
                "Failed login attempt",
                request.remote_addr,
            )
        flash('Invalid credentials – please try again.')

    return render_template('login.html', error=None)


@auth_bp.route('/logout')
def logout():
    """Clear the session and redirect to the login page."""
    engine = current_app.config['VERIFICATION_ENGINE']
    email  = session.get('user_email', 'UNKNOWN')

    if engine and 'logged_in' in session:
        engine.db.log_audit(
            "LOGOUT", email,
            "Admin logged out",
            request.remote_addr,
        )

    session.clear()
    return redirect(url_for('auth.login'))
