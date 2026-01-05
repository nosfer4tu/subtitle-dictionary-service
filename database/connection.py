import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from pathlib import Path
from contextlib import contextmanager


env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
#load from current directory as fallback
load_dotenv()
DATABASE_URL = os.environ.get("DATABASE_URL")

@contextmanager
def get_db():
    if not DATABASE_URL:
        print("DATABASE_URL not set", flush=True)
        raise ValueError("DATABASE_URL environment variable is not set")
    conn = None
    try:
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        yield conn
        # Only commit if no exception occurred and transaction wasn't already committed
        # Note: Explicit commits in code are fine, this is a safety net
        try:
            if not conn.closed:
                conn.commit()
        except Exception:
            pass  # Connection might already be closed or committed
    except Exception as e:
        if conn and not conn.closed:
            try:
                conn.rollback()
            except Exception:
                pass  # Ignore rollback errors
        print("Database connection error:", e, flush=True)
        raise
    finally:
        if conn and not conn.closed:
            conn.close()