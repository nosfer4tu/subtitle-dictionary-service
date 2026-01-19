from database.connection import get_db
from psycopg2.extras import RealDictCursor
from datetime import datetime


def create_watch_later_table():
    """
    Create the watch_later table if it doesn't exist.
    Run this once to set up the database schema.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS watch_later (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER NOT NULL,
                        movie_id INTEGER NOT NULL,
                        movie_title VARCHAR(500),
                        added_at TIMESTAMP DEFAULT NOW(),
                        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE,
                        UNIQUE(user_id, movie_id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_watch_later_user_id ON watch_later(user_id);
                    CREATE INDEX IF NOT EXISTS idx_watch_later_movie_id ON watch_later(movie_id);
                    CREATE INDEX IF NOT EXISTS idx_watch_later_added_at ON watch_later(added_at DESC);
                """)
                conn.commit()
                print("Watch later table created/verified successfully", flush=True)
    except Exception as e:
        print(f"Error creating watch_later table: {e}", flush=True)
        raise


def add_to_watch_later(user_id: int, movie_id: int, movie_title: str = None):
    """
    Add a movie to user's watch later list.
    If the movie is already in the list, update the added_at timestamp.
    
    Args:
        user_id: User ID
        movie_id: TMDB movie ID
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
                    SELECT id FROM watch_later 
                    WHERE user_id = %s AND movie_id = %s
                    """,
                    (user_id, movie_id)
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing entry with new timestamp
                    cursor.execute(
                        """
                        UPDATE watch_later 
                        SET added_at = NOW(), movie_title = COALESCE(%s, movie_title)
                        WHERE id = %s
                        """,
                        (movie_title, existing[0])
                    )
                else:
                    # Insert new entry
                    cursor.execute(
                        """
                        INSERT INTO watch_later (user_id, movie_id, movie_title, added_at)
                        VALUES (%s, %s, %s, NOW())
                        """,
                        (user_id, movie_id, movie_title)
                    )
                
                conn.commit()
                return True
    except Exception as e:
        print(f"Error adding to watch later: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return False


def remove_from_watch_later(user_id: int, movie_id: int):
    """
    Remove a movie from user's watch later list.
    
    Args:
        user_id: User ID
        movie_id: TMDB movie ID
    
    Returns:
        True if removed, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    DELETE FROM watch_later 
                    WHERE user_id = %s AND movie_id = %s
                    """,
                    (user_id, movie_id)
                )
                conn.commit()
                return cursor.rowcount > 0
    except Exception as e:
        print(f"Error removing from watch later: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return False


def is_in_watch_later(user_id: int, movie_id: int):
    """
    Check if a movie is in user's watch later list.
    
    Args:
        user_id: User ID
        movie_id: TMDB movie ID
    
    Returns:
        True if in watch later, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT id FROM watch_later 
                    WHERE user_id = %s AND movie_id = %s
                    """,
                    (user_id, movie_id)
                )
                return cursor.fetchone() is not None
    except Exception as e:
        print(f"Error checking watch later: {e}", flush=True)
        return False


def get_watch_later(user_id: int, limit: int = 100):
    """
    Get watch later list for a user, ordered by most recently added first.
    
    Args:
        user_id: User ID
        limit: Maximum number of entries to return (default: 100)
    
    Returns:
        List of watch later entries as dictionaries
    """
    try:
        with get_db() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        id, movie_id, movie_title, added_at
                    FROM watch_later
                    WHERE user_id = %s
                    ORDER BY added_at DESC
                    LIMIT %s
                """, (user_id, limit))
                
                return cursor.fetchall()
    except Exception as e:
        print(f"Error getting watch later: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return []


def get_watch_later_count(user_id: int):
    """
    Get total count of watch later entries for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        Count of watch later entries
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "SELECT COUNT(*) FROM watch_later WHERE user_id = %s",
                    (user_id,)
                )
                result = cursor.fetchone()
                return result[0] if result else 0
    except Exception as e:
        print(f"Error getting watch later count: {e}", flush=True)
        return 0


def delete_watch_later_entry(user_id: int, entry_id: int):
    """
    Delete a specific watch later entry.
    Only allows deletion if the entry belongs to the user.
    
    Args:
        user_id: User ID
        entry_id: Watch later entry ID
    
    Returns:
        True if deleted, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                # Verify ownership
                cursor.execute(
                    "SELECT id FROM watch_later WHERE id = %s AND user_id = %s",
                    (entry_id, user_id)
                )
                if not cursor.fetchone():
                    return False
                
                # Delete entry
                cursor.execute(
                    "DELETE FROM watch_later WHERE id = %s AND user_id = %s",
                    (entry_id, user_id)
                )
                conn.commit()
                return True
    except Exception as e:
        print(f"Error deleting watch later entry: {e}", flush=True)
        return False


def clear_watch_later(user_id: int):
    """
    Clear all watch later entries for a user.
    
    Args:
        user_id: User ID
    
    Returns:
        True if successful, False otherwise
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute(
                    "DELETE FROM watch_later WHERE user_id = %s",
                    (user_id,)
                )
                conn.commit()
                return True
    except Exception as e:
        print(f"Error clearing watch later: {e}", flush=True)
        return False
