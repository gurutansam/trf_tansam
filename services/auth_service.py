from functools import wraps
from flask import session, redirect, url_for, request
from werkzeug.security import generate_password_hash, check_password_hash

def hash_password(password: str) -> str:
    """Generate secure password hash."""
    return generate_password_hash(password)

def verify_password(stored_password: str, provided_password: str) -> bool:
    """
    Verify password against stored hash with fallback to plain comparison
    for existing legacy accounts.
    """
    if not stored_password:
        return False
    # If already hashed
    if stored_password.startswith(("scrypt:", "pbkdf2:")):
        return check_password_hash(stored_password, provided_password)
    # Plain text fallback
    return stored_password == provided_password

def login_required(f):
    """Decorator to require user session authentication for protected routes."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get("user"):
            return redirect("/login")
        return f(*args, **kwargs)
    return decorated_function
