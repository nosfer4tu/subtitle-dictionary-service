from dotenv import load_dotenv
from pathlib import Path
import os
import secrets
import hashlib
import base64
from flask import Blueprint,request, render_template, redirect, url_for, session
from database.connection import get_db
from psycopg2.extras import RealDictCursor

auth_blueprint = Blueprint("auth", __name__)

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
#load from current directory as fallback
load_dotenv()

HASH_ALGORITHM = os.environ.get("HASH_ALGORITHM")

def hash_password(password, salt=None, iterations=310000):
    if salt is None:
        salt = secrets.token_hex(16)
    assert salt and isinstance(salt, str) and "$" not in salt
    assert isinstance(password, str)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations
    )
    b64_hash = base64.b64encode(pw_hash).decode("ascii").strip()
    return "{}${}${}${}".format(HASH_ALGORITHM, iterations, salt, b64_hash)

def verify_password(password, password_hash):
    if (password_hash or "").count("$") != 3:
        return False
    algorithm, iterations, salt, _ = password_hash.split("$", 3)
    iterations = int(iterations)
    assert algorithm == HASH_ALGORITHM
    compare_hash = hash_password(password, salt, iterations)
    return secrets.compare_digest(password_hash, compare_hash)

@auth_blueprint.route("/login", methods=["POST"])
def login():
    login_input = request.form.get("login")
    if not login_input:
        return render_template("login.html", error_user=True, form=request.form)

    password = request.form.get("password")
    if not password:
        return render_template("login.html", error_password=True, form=request.form)

    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE username = %s or email = %s", 
                (login_input, login_input)
            )
            row = cursor.fetchone()

            verified = row is not None and verify_password(
                password, row["password_hash"]
            )

            if verified:
                session["user_id"] = row["id"]
                return redirect(url_for("index"))
            else:
                return render_template("login.html", error_login=True)

@auth_blueprint.route("/register", methods=["POST"])
def register():
    username = request.form.get("username")
    email = request.form.get("email")
    if not username or len(username) < 3:
        return render_template("register.html", error_user=True, form=request.form)

    password = request.form.get("password")
    if not password:
        return render_template("register.html", error_password=True, form=request.form)

    password_confirmation = request.form.get("password_confirmation")
    if password != password_confirmation:
        return render_template("register.html", error_confirm=True, form=request.form)

    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM users WHERE username = %s", (username,)
            )
            res = cursor.fetchall()
            if len(res) != 0:
                return render_template(
                    "register.html", error_unique=True, form=request.form
                )

            password_hash = hash_password(password)
            cursor.execute(
                "INSERT INTO users (username, email, password_hash) VALUES (%s, %s, %s)",
                (username, email, password_hash),
            )
        conn.commit()

    return redirect(url_for("login_form"))
@auth_blueprint.route("/logout")
def logout():
    session.pop("user_id", None)
    return redirect(url_for("index"))

@auth_blueprint.route("/login", methods=["GET"])
def login_form():
    return render_template("login.html")

@auth_blueprint.route("/register", methods=["GET"])
def register_form():
    return render_template("register.html")