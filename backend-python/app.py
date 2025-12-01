from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import requests
import base64
import secrets
import hashlib
import psycopg2
import psycopg2.extras
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env file
# Try to find .env in the project root (one level up from backend-python)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
# Also try loading from current directory as fallback
load_dotenv()

app = Flask(__name__, template_folder='../frontend')
app.secret_key = os.environ.get("SECRET_KEY")

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_SEARCH_URL = os.environ.get("TMDB_SEARCH_URL")
TMDB_IMAGE_BASE_URL = os.environ.get("TMDB_IMAGE_BASE_URL")
HASH_ALGORITHM = os.environ.get("HASH_ALGORITHM")
DATABASE_URL = os.environ.get("DATABASE_URL")

def get_db():
    if not DATABASE_URL:
        print("DATABASE_URL not set", flush=True)
        raise ValueError("DATABASE_URL environment variable is not set")
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    except Exception as e:
        print("Database connection error:", e, flush=True)
        raise
    return conn

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

# @app.route("/login", methods=["POST"])
# def login():
#     login_input = request.form.get("login")
#     if not login_input:
#         return render_template("login.html", error_user=True, form=request.form)

#     password = request.form.get("password")
#     if not password:
#         return render_template("login.html", error_password=True, form=request.form)

#     with get_db() as conn:
#         with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#             cursor.execute(
#                 "SELECT * FROM users WHERE username = %s or email = %s", 
#                 (login_input, login_input)
#             )
#             row = cursor.fetchone()

#             verified = row is not None and verify_password(
#                 password, row["password_hash"]
#             )

#             if verified:
#                 session["user_id"] = row["id"]
#                 return redirect(url_for("index"))
#             else:
#                 return render_template("login.html", error_login=True)

# @app.route("/register", methods=["POST"])
# def register():
#     username = request.form.get("username")
#     email = request.form.get("email")
#     if not username or len(username) < 3:
#         return render_template("register.html", error_user=True, form=request.form)

#     password = request.form.get("password")
#     if not password:
#         return render_template("register.html", error_password=True, form=request.form)

#     password_confirmation = request.form.get("password_confirmation")
#     if password != password_confirmation:
#         return render_template("register.html", error_confirm=True, form=request.form)

#     with get_db() as conn:
#         with conn.cursor(cursor_factory=RealDictCursor) as cursor:
#             cursor.execute(
#                 "SELECT * FROM users WHERE username = %s", (username,)
#             )
#             res = cursor.fetchall()
#             if len(res) != 0:
#                 return render_template(
#                     "register.html", error_unique=True, form=request.form
#                 )

#             password_hash = hash_password(password)
#             cursor.execute(
#                 "INSERT INTO users (username, password_hash) VALUES (%s, %s)",
#                 (username, password_hash),
#             )
#         conn.commit()

#     return redirect(url_for("login_form"))
# @app.route("/logout")
# def logout():
#     session.pop("user_id", None)
#     return redirect(url_for("index"))

# @app.route("/login", methods=["GET"])
# def login_form():
#     return render_template("login.html")

# @app.route("/register", methods=["GET"])
# def register_form():
#     return render_template("register.html")

def search_movies(query):
    if not query:
        return []
    
    if not TMDB_API_KEY or not TMDB_SEARCH_URL or not TMDB_IMAGE_BASE_URL:
        print("Error: TMDB environment variables not set", flush=True)
        return []
    
    
    params = {
        "api_key": TMDB_API_KEY,
        "query": query,
        "language": "en-US",
    }
    res = requests.get(TMDB_SEARCH_URL, params=params)
    res.raise_for_status()  # Raise an exception for bad status codes
    data = res.json()
    
    
    
    movies = []
    for item in data.get("results", []):
        poster_path = item.get("poster_path")
        title = item.get("title")
        movie_id = item.get("id")
        if poster_path and title and movie_id:
            movies.append({
                "id": movie_id,
                "title": title,
                "api_url": f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            })
    return movies

def fetch_indian_movies():
    if not TMDB_API_KEY or not TMDB_SEARCH_URL or not TMDB_IMAGE_BASE_URL:
        print("Error: TMDB environment variables not set", flush=True)
        return []
    url = "http://api.themoviedb.org/3/discover/movie"
    
    params = {
        "api_key": TMDB_API_KEY,
        "with_origin_country": "IN",
        "sort_by": "popularity.desc",
        "language": "en-US",
        "page": 1,
    }
    res = requests.get(url, params=params)
    res.raise_for_status()
    data = res.json()
    
    movies = []
    for item in data.get("results", []):
        poster_path = item.get("poster_path")
        title = item.get("title")
        movie_id = item.get("id")
        if poster_path and title and movie_id:
            movies.append({
                "id": movie_id,
                "title": title,
                "api_url": f"{TMDB_IMAGE_BASE_URL}{poster_path}"
            })
    return movies

@app.route('/')
def index():
    # if "user_id" not in session:
    #     return redirect(url_for("login"))
    
    
        query = request.args.get("query")
        movies = search_movies(query) if query else []
        indian_movies = fetch_indian_movies()
        return render_template("index.html", movies=movies, query=query or "", indian_movies=indian_movies)

# @app.route('/movie/<int:movie_id>/review', methods=['POST'])
# def add_review(movie_id):
#     if 'user_id' not in session:
#         return redirect(url_for('login'))

#     comment = request.form.get('comment')

#     with get_db() as db:
#         with db.cursor() as cur:
#             cur.execute('INSERT INTO reviews (movie_id, user_id, comment) VALUES (%s, %s, %s)',
#                         (movie_id, session['user_id'], comment))
#             db.commit()

#     return redirect(url_for('movie_detail', movie_id=movie_id))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

