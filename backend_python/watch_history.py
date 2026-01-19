from database.connection import get_db
from psycopg2.extras import RealDictCursor
from datetime import datetime


def create_watch_history_table():
    """
    Create the watch_history table if it doesn't exist.
    Run this once to set up the database schema.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS watch_history (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        movie_id INTEGER NOT NULL,
                        youtube_video_id VARCHAR(255) NOT NULL,
                        movie_title VARCHAR(500),
                        watched_at TIMESTAMP DEFAULT NOW(),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_watch_history_user_id ON watch_history(user_id);
                    CREATE INDEX IF NOT EXISTS idx_watch_history_movie_id ON watch_history(movie_id);
                    CREATE INDEX IF NOT EXISTS idx_watch_history_watched_at ON watch_history(watched_at DESC);
                """)
                conn.commit()
                print("Watch history table created/verified successfully", flush=True)
    except Exception as e:
        print(f"Error creating watch_history table: {e}", flush=True)
        raise


def add_watch_history(user_id: int, movie_id: int, youtube_video_id: str, movie_title: str = None):
    """
    Add or update watch history entry for a user.
    If the user has already watched this movie, update the watched_at timestamp.
    
    Args:
        user_id: User ID
        movie_id: TMDB movie ID
        youtube_video_id: YouTube video ID
        movie_title: Movie title (optional)
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                # Check if entry already exists
                cursor.execute(
                    """
                    SELECT id FROM watch_history 
                    WHERE user_id = %s AND movie_id = %s AND youtube_video_id = %s
                    """,
                    (user_id, movie_id, youtube_video_id)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entry with new timestamp
                    cursor.execute(
                        """
                        UPDATE watch_history 
                        SET watched_at = NOW(), movie_title = COALESCE(%s, movie_title)
                        WHERE id = %s
                        """,
                        (movie_title, existing[0])
                    )
                else:
                    # Insert new entry
                    cursor.execute(
                        """
                        INSERT INTO watch_history (user_id, movie_id, youtube_video_id, movie_title, watched_at)
                        VALUES (%s, %s, %s, %s, NOW())
                        """,
                        (user_id, movie_id, youtube_video_id, movie_title)
                    )
                
                conn.commit()
                return True
    except Exception as e:
        print(f"Error adding watch history: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return False


def get_watch_history(user_id: int, limit: int = 50):
    """
    Get watch history for a user, ordered by most recent first.
    
    Args:
        user_id: User ID
        limit: Maximum number of entries to return (default: 50)
    
    Returns:
        List of watch history entries as dictionaries
    """
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, movie_id, youtube_video_id, movie_title, watched_at
                    FROM watch_history
                    WHERE user_id = %s
                    ORDER BY watched_at DESC
                    LIMIT %s
                """, (user_id, limit))
                
                return cursor.fetchall()
    except Exception as e:
        print(f"Error getting watch history: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return []


def get_watch_history_count(user_id: int):
    """
    Get total count of watch history entries for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        Count of watch history entries
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM watch_history WHERE user_id = %s",
                    (user_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else 0
    except Exception as e:
        print(f"Error getting watch history count: {e}", flush=True)
        return 0


def delete_watch_history_entry(user_id: int, entry_id: int):
    """
    Delete a specific watch history entry.
    Only allows deletion if the entry belongs to the user.
    
    Args:
        user_id: User ID
        entry_id: Watch history entry ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                # Verify ownership
                cursor.execute(
                    "SELECT id FROM watch_history WHERE id = %s AND user_id = %s",
                    (entry_id, user_id)
                )
                if not cursor.fetchone():
                    return False
                
                # Delete entry
                cursor.execute(
                    "DELETE FROM watch_history WHERE id = %s AND user_id = %s",
                    (entry_id, user_id)
                )
                conn.commit()
                return True
    except Exception as e:
        print(f"Error deleting watch history entry: {e}", flush=True)
        return False


def clear_watch_history(user_id: int):
    """
    Clear all watch history for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM watch_history WHERE user_id = %s",
                    (user_id,)
                )
                conn.commit()
                return True
    except Exception as e:
        print(f"Error clearing watch history: {e}", flush=True)
        return False
