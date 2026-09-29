from flask import Blueprint, render_template, request, redirect, session
from database import get_db
from services.auth_service import verify_password, hash_password

auth_bp = Blueprint("auth", __name__)

@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()
        conn.close()

        if user:
            if verify_password(user["password"], password):
                session["user"] = user["email"]
                session["name"] = user["name"]
                return redirect("/site")
            else:
                return render_template("Login.html", error="Wrong password")
        else:
            return render_template("Login.html", error="Account does not exist")

    return render_template("Login.html")


@auth_bp.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not all([name, email, password]):
            return render_template("register.html", error="All fields are required")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        existing_user = cur.fetchone()

        if existing_user:
            conn.close()
            return render_template("register.html", error="Email already exists")

        hashed = hash_password(password)
        cur.execute(
            "INSERT INTO users(name, email, password) VALUES(?, ?, ?)",
            (name, email, hashed)
        )
        conn.commit()
        conn.close()

        return redirect("/login")

    return render_template("register.html")


@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        new_password = request.form.get("password", "")

        if not email or not new_password:
            return render_template("forgot_password.html", error="Please provide email and new password")

        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cur.fetchone()

        if user:
            hashed = hash_password(new_password)
            cur.execute("UPDATE users SET password=? WHERE email=?", (hashed, email))
            conn.commit()
            conn.close()
            return redirect("/login")
        else:
            conn.close()
            return render_template("forgot_password.html", error="Email not found")

    return render_template("forgot_password.html")
