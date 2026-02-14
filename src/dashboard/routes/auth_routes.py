"""
Authentication Routes
Handles user login and logout
"""
from flask import Blueprint, render_template, request, redirect, url_for, session, flash, current_app
from functools import wraps

auth_bp = Blueprint('auth', __name__)

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'logged_in' not in session:
            return redirect(url_for('auth.login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    """Login route"""
    engine = current_app.config['VERIFICATION_ENGINE']
    
    if request.method == 'POST':
        email = request.form.get('email')
        password = request.form.get('password')
        
        # Simple hardcoded check for prototype
        if email == current_app.config['ADMIN_EMAIL'] and password == current_app.config['ADMIN_PASSWORD']:
            session['logged_in'] = True
            session['user_email'] = email
            
            # Log audit event
            if engine:
                engine.db.log_audit("LOGIN", email, "User logged in successfully", request.remote_addr)
            
            next_page = request.args.get('next')
            return redirect(next_page or url_for('main.index'))
        else:
            # Log failed login attempt
            if engine:
                engine.db.log_audit("LOGIN_FAILURE", email or "UNKNOWN", "Failed login attempt", request.remote_addr)
            flash('Invalid credentials')
            
    return render_template('login.html', error=None)

@auth_bp.route('/logout')
def logout():
    """Logout route"""
    engine = current_app.config['VERIFICATION_ENGINE']
    email = session.get('user_email', 'UNKNOWN')
    
    # Log audit event
    if engine and 'logged_in' in session:
        engine.db.log_audit("LOGOUT", email, "User logged out", request.remote_addr)
        
    session.clear()
    return redirect(url_for('auth.login'))
