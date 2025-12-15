import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from pathlib import Path


env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
#load from current directory as fallback
load_dotenv()
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