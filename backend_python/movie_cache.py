from database.connection import get_db
import json
import os
from pathlib import Path
from datetime import datetime

# Create a persistent cache directory for video files
CACHE_DIR = Path(__file__).parent.parent / 'video_cache'
CACHE_DIR.mkdir(exist_ok=True)

def get_cached_movie(movie_id: int = None, video_id: str = None):
    """
    Retrieve cached movie processing data.
    Tries both movie_id and video_id for better matching.
    
    Args:
        movie_id: TMDB movie ID
        video_id: YouTube video ID
    
    Returns:
        Dictionary with cached data or None if not found
    """
    if not movie_id and not video_id:
        return None
    
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                # Try movie_id first, then video_id, then both
                if movie_id and video_id:
                    cursor.execute(
                        """
                        SELECT 
                            movie_id, youtube_video_id, source_transcription, 
                            japanese_translation, detected_terms, language, 
                            segments, pipeline_version, created_at, updated_at
                        FROM movie_cache
                        WHERE movie_id = %s AND youtube_video_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (movie_id, video_id)
                    )
                elif movie_id:
                    cursor.execute(
                        """
                        SELECT 
                            movie_id, youtube_video_id, source_transcription, 
                            japanese_translation, detected_terms, language, 
                            segments, pipeline_version, created_at, updated_at
                        FROM movie_cache
                        WHERE movie_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (movie_id,)
                    )
                elif video_id:
                    cursor.execute(
                        """
                        SELECT 
                            movie_id, youtube_video_id, source_transcription, 
                            japanese_translation, detected_terms, language, 
                            segments, pipeline_version, created_at, updated_at
                        FROM movie_cache
                        WHERE youtube_video_id = %s
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """,
                        (video_id,)
                    )
                
                row = cursor.fetchone()
                if row:
                    cached_video_id = row[1]
                    # Check if cached video file exists
                    cached_video_path = get_cached_video_path(cached_video_id)
                    
                    # Parse detected_terms (JSONB is already parsed by psycopg2, but might be string in some cases)
                    detected_terms = None
                    if row[4]:  # detected_terms column
                        try:
                            if isinstance(row[4], str):
                                detected_terms = json.loads(row[4])
                            else:
                                detected_terms = row[4]  # Already a dict/list from JSONB
                            terms_count = len(detected_terms.get('terms', [])) if isinstance(detected_terms, dict) else 0
                            print(f"Retrieved {terms_count} cultural terms from cache", flush=True)
                        except Exception as e:
                            print(f"Warning: Could not parse cached detected_terms: {e}", flush=True)
                    
                    # Parse segments (JSONB is already parsed by psycopg2, but might be string in some cases)
                    segments = None
                    if row[6]:  # segments column
                        try:
                            if isinstance(row[6], str):
                                segments = json.loads(row[6])
                            else:
                                segments = row[6]  # Already a list/dict from JSONB
                        except Exception as e:
                            print(f"Warning: Could not parse cached segments: {e}", flush=True)
                    
                    return {
                        'movie_id': row[0],
                        'youtube_video_id': row[1],
                        'source_transcription': row[2],
                        'japanese_translation': row[3],
                        'detected_terms': detected_terms,
                        'language': row[5],
                        'segments': segments,
                        'pipeline_version': row[7] if row[7] is not None else 1,  # Default to 1 for old cache entries
                        'created_at': row[8],
                        'updated_at': row[9],
                        'cached_video_path': cached_video_path if os.path.exists(cached_video_path) else None
                    }
                return None
    except Exception as e:
        print(f"Error retrieving cached movie: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)
        return None

def get_cached_video_path(video_id: str) -> str:
    """Get path to cached processed video file"""
    return str(CACHE_DIR / f"{video_id}_processed.mp4")

def get_cached_raw_video_path(video_id: str) -> str:
    """Get path to cached raw downloaded video file"""
    return str(CACHE_DIR / f"{video_id}_raw.mp4")

def save_cached_movie(
    movie_id: int,
    video_id: str,
    source_transcription: str,
    japanese_translation: str,
    detected_terms: dict = None,
    language: str = None,
    segments: list = None,
    processed_video_path: str = None,
    pipeline_version: int = 1
):
    """
    Save processed movie data to cache.
    Also saves the processed video file to persistent cache.
    
    Args:
        movie_id: TMDB movie ID
        video_id: YouTube video ID
        source_transcription: Source language transcription
        japanese_translation: Japanese translation
        detected_terms: Dictionary of detected cultural terms
        language: Language code (hi, kn, ta)
        segments: List of transcription segments with timestamps
        processed_video_path: Path to processed video file (will be copied to cache)
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                # Check if entry exists
                cursor.execute(
                    "SELECT id FROM movie_cache WHERE movie_id = %s AND youtube_video_id = %s",
                    (movie_id, video_id)
                )
                existing = cursor.fetchone()
                
                detected_terms_json = json.dumps(detected_terms) if detected_terms else None
                segments_json = json.dumps(segments) if segments else None
                
                if existing:
                    # Update existing entry
                    cursor.execute(
                        """
                        UPDATE movie_cache
                        SET source_transcription = %s,
                            japanese_translation = %s,
                            detected_terms = %s,
                            language = %s,
                            segments = %s,
                            pipeline_version = %s,
                            updated_at = NOW()
                        WHERE movie_id = %s AND youtube_video_id = %s
                        """,
                        (
                            source_transcription,
                            japanese_translation,
                            detected_terms_json,
                            language,
                            segments_json,
                            pipeline_version,
                            movie_id,
                            video_id
                        )
                    )
                    print(f"Updated cached movie: movie_id={movie_id}, video_id={video_id}, pipeline_version={pipeline_version}", flush=True)
                else:
                    # Insert new entry
                    cursor.execute(
                        """
                        INSERT INTO movie_cache 
                        (movie_id, youtube_video_id, source_transcription, japanese_translation, 
                         detected_terms, language, segments, pipeline_version, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        """,
                        (
                            movie_id,
                            video_id,
                            source_transcription,
                            japanese_translation,
                            detected_terms_json,
                            language,
                            segments_json,
                            pipeline_version
                        )
                    )
                    print(f"Cached new movie: movie_id={movie_id}, video_id={video_id}, pipeline_version={pipeline_version}", flush=True)
                
                conn.commit()
                
                # Copy processed video to persistent cache if provided
                if processed_video_path and os.path.exists(processed_video_path):
                    cached_video_path = get_cached_video_path(video_id)
                    try:
                        import shutil
                        shutil.copy2(processed_video_path, cached_video_path)
                        print(f"Cached processed video to: {cached_video_path}", flush=True)
                    except Exception as e:
                        print(f"Warning: Could not cache video file: {e}", flush=True)
                        
    except Exception as e:
        print(f"Error saving cached movie: {e}", flush=True)
        import traceback
        print(f"Traceback: {traceback.format_exc()}", flush=True)

def save_cached_raw_video(video_id: str, video_path: str):
    """
    Save raw downloaded video to cache to avoid re-downloading.
    
    Args:
        video_id: YouTube video ID
        video_path: Path to downloaded video file
    """
    if not os.path.exists(video_path):
        return None
    
    cached_path = get_cached_raw_video_path(video_id)
    try:
        import shutil
        shutil.copy2(video_path, cached_path)
        print(f"Cached raw video to: {cached_path}", flush=True)
        return cached_path
    except Exception as e:
        print(f"Warning: Could not cache raw video: {e}", flush=True)
        return None

def get_cached_raw_video(video_id: str) -> str:
    """
    Get cached raw video if it exists.
    
    Args:
        video_id: YouTube video ID
    
    Returns:
        Path to cached video or None if not found
    """
    cached_path = get_cached_raw_video_path(video_id)
    if os.path.exists(cached_path) and os.path.getsize(cached_path) > 0:
        print(f"Found cached raw video: {cached_path}", flush=True)
        return cached_path
    return None

def create_movie_cache_table():
    """
    Create the movie_cache table if it doesn't exist.
    Run this once to set up the database schema.
    """
    try:
        with get_db() as conn:
            with conn.cursor() as cursor:
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS movie_cache (
                        id SERIAL PRIMARY KEY,
                        movie_id INTEGER NOT NULL,
                        youtube_video_id VARCHAR(255) NOT NULL,
                        source_transcription TEXT,
                        japanese_translation TEXT,
                        detected_terms JSONB,
                        language VARCHAR(10),
                        segments JSONB,
                        pipeline_version INTEGER DEFAULT 1,
                        created_at TIMESTAMP DEFAULT NOW(),
                        updated_at TIMESTAMP DEFAULT NOW(),
                        UNIQUE(movie_id, youtube_video_id)
                    );
                    
                    CREATE INDEX IF NOT EXISTS idx_movie_cache_movie_id ON movie_cache(movie_id);
                    CREATE INDEX IF NOT EXISTS idx_movie_cache_video_id ON movie_cache(youtube_video_id);
                """)
                
                # Add pipeline_version column if it doesn't exist (for existing tables)
                try:
                    cursor.execute("""
                        ALTER TABLE movie_cache 
                        ADD COLUMN IF NOT EXISTS pipeline_version INTEGER DEFAULT 1
                    """)
                except Exception as e:
                    # Column might already exist, ignore
                    pass
                conn.commit()
                print("Movie cache table created/verified successfully", flush=True)
    except Exception as e:
        print(f"Error creating movie_cache table: {e}", flush=True)
        raise
