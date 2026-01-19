from flask import Flask, request, jsonify, render_template, redirect, url_for, session, abort
import requests
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path
from backend_python.auth import auth_blueprint, create_users_table
from backend_python.tmdb import search_movies, fetch_indian_movies, get_movie_trailer, get_movie_details
from backend_python.youtube_upload import download_video, extract_video_id
from backend_python.subtitle_processor import process_trailer_video
from backend_python.dictionary import add_dictionary_entry, update_dictionary_transcription, save_words_to_dictionary
from database.connection import get_db
from backend_python.movie_cache import (
    get_cached_movie, save_cached_movie, 
    get_cached_raw_video, save_cached_raw_video,
    get_cached_video_path,
    create_movie_cache_table  # Add this import
)
from backend_python.watch_history import (
    create_watch_history_table,
    add_watch_history,
    get_watch_history,
    delete_watch_history_entry,
    clear_watch_history
)
from backend_python.watch_later import (
    create_watch_later_table,
    add_to_watch_later,
    remove_from_watch_later,
    is_in_watch_later,
    get_watch_later,
    delete_watch_later_entry,
    clear_watch_later
)
import tempfile
import shutil  # Add shutil import for file operations

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

app = Flask(__name__, template_folder='./frontend')
app.secret_key = os.environ.get("SECRET_KEY")
app.register_blueprint(auth_blueprint)

# Pipeline version for cache invalidation
# Increment this when the translation pipeline logic changes
# Version 3: Removed source text from subtitles, display detected terms on middle-right in vertical line
PIPELINE_VERSION = 15

# Initialize database tables on startup
try:
    create_users_table()
except Exception as e:
    print(f"Warning: Could not initialize users table: {e}", flush=True)

try:
    create_movie_cache_table()
except Exception as e:
    print(f"Warning: Could not initialize movie cache table: {e}", flush=True)

try:
    create_watch_history_table()
except Exception as e:
    print(f"Warning: Could not initialize watch history table: {e}", flush=True)

try:
    create_watch_later_table()
except Exception as e:
    print(f"Warning: Could not initialize watch later table: {e}", flush=True)

@app.route('/')
def index():
    """Top page with project explanation and auth buttons"""
    user_id = session.get('user_id')
    return render_template("top.html", user_id=user_id)

@app.route('/app')
def app_index():
    """Movie search page - requires authentication"""
    if "user_id" not in session:
        return redirect(url_for("auth.login_form"))
    
    # Ignore indian_page parameter (only page 1 is allowed)
    if request.args.get("indian_page") and request.args.get("indian_page") != "1":
        # Redirect to remove the indian_page parameter if it's not 1
        query = request.args.get("query")
        if query:
            return redirect(url_for("app_index", query=query))
        else:
            return redirect(url_for("app_index"))
    
    query = request.args.get("query")
    movies = search_movies(query) if query else []
    
    # Only show page 1 (40 entries limit)
    indian_movies_data = fetch_indian_movies(page=1, movies_per_page=40)
    
    return render_template(
        "index.html", 
        movies=movies, 
        query=query or "", 
        indian_movies=indian_movies_data["movies"],
        indian_current_page=indian_movies_data["current_page"],
        indian_total_pages=indian_movies_data["total_pages"],
        indian_has_next=indian_movies_data["has_next"]
    )

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    """Movie detail page that automatically uploads trailer"""
    if "user_id" not in session:
        return redirect(url_for("auth.login_form"))
    
    user_id = session.get('user_id')
    movie = get_movie_details(movie_id)
    if not movie:
        return render_template("error.html", message="Movie not found or language not supported"), 404
    
    trailer_url = get_movie_trailer(movie_id)
    in_watch_later = is_in_watch_later(user_id, movie_id)
    return render_template("movie_detail.html", movie=movie, trailer_url=trailer_url, in_watch_later=in_watch_later)

@app.route('/api/video/<path:filename>')
def serve_video(filename):
    """Serve processed video files"""
    # Security: Only allow files from temp directory or video cache
    temp_dir = tempfile.gettempdir()
    video_cache_dir = os.path.join(os.path.dirname(__file__), 'video_cache')
    
    # Try temp directory first
    video_path = os.path.join(temp_dir, filename)
    if os.path.exists(video_path) and video_path.startswith(temp_dir):
        from flask import send_file
        return send_file(video_path, mimetype='video/mp4')
    
    # Try video cache directory
    video_path = os.path.join(video_cache_dir, filename)
    if os.path.exists(video_path) and video_path.startswith(video_cache_dir):
        from flask import send_file
        return send_file(video_path, mimetype='video/mp4')
    
    return jsonify({'error': 'Video not found'}), 404

@app.route('/api/upload-trailer', methods=['POST'])
def upload_trailer():
    """Handle trailer upload with transcription, translation, and subtitles"""
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        movie_title = data.get('movie_title', 'Unknown Movie')
        user_id = session.get('user_id', 1)
        
        if not movie_id:
            return jsonify({'error': 'Movie ID is required'}), 400
        
        # Get movie details to detect language
        movie = get_movie_details(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found or language not supported'}), 404
        
        language = movie.get('original_language')
        
        # Get trailer URL from TMDB
        trailer_url = get_movie_trailer(movie_id)
        if not trailer_url:
            return jsonify({'error': 'No trailer found for this movie'}), 404
        
        # Extract YouTube video ID
        try:
            video_id = extract_video_id(trailer_url)
        except ValueError:
            return jsonify({'error': 'Invalid trailer URL'}), 400
        
        # Check cache first - try both movie_id and video_id
        print(f"Checking cache for movie_id={movie_id}, video_id={video_id}...", flush=True)
        cached_data = get_cached_movie(movie_id=movie_id, video_id=video_id)
        
        if cached_data:
            # Check pipeline version for cache invalidation
            cached_version = cached_data.get('pipeline_version', 1)  # Default to 1 for old cache entries
            
            if cached_version == PIPELINE_VERSION:
                # Cache hit (valid) - version matches
                print(f"✓ CACHE HIT (valid): Found cached data for movie_id={movie_id}, video_id={video_id}, pipeline_version={cached_version}", flush=True)
                
                # Log what we retrieved from cache
                terms_count = 0
                if cached_data.get('detected_terms') and isinstance(cached_data['detected_terms'], dict):
                    terms_count = len(cached_data['detected_terms'].get('terms', []))
                print(f"✓ Retrieved from cache: transcription ({len(cached_data.get('source_transcription', '') or '')} chars), translation ({len(cached_data.get('japanese_translation', '') or '')} chars), {terms_count} cultural terms", flush=True)
                
                # Check if we have a cached processed video
                cached_video_path = cached_data.get('cached_video_path')
                output_video = None
                
                if cached_video_path and os.path.exists(cached_video_path):
                    # Copy cached video to temp location for serving
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        output_video = tmp_file.name
                    shutil.copy2(cached_video_path, output_video)
                    print(f"Using cached processed video: {cached_video_path}", flush=True)
                
                # Save to dictionary and words if not already saved (for cached data)
                try:
                    # Check if dictionary exists for this user and video_id
                    with get_db() as conn:
                        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                            cursor.execute(
                                "SELECT id FROM dictionaries WHERE user_id = %s AND youtube_video_id = %s",
                                (user_id, video_id)
                            )
                            existing_dict = cursor.fetchone()
                            
                            if not existing_dict:
                                # Create dictionary entry if it doesn't exist
                                dict_id = add_dictionary_entry(user_id, video_id, movie_title)
                                update_dictionary_transcription(
                                    dict_id,
                                    cached_data['source_transcription'],
                                    cached_data['japanese_translation']
                                )
                                # Save words from detected_terms
                                if cached_data.get('detected_terms'):
                                    save_words_to_dictionary(dict_id, cached_data['detected_terms'])
                            else:
                                # Dictionary exists, just ensure words are saved
                                if cached_data.get('detected_terms'):
                                    save_words_to_dictionary(existing_dict['id'], cached_data['detected_terms'])
                except Exception as db_error:
                    print(f"Database error (non-critical): {db_error}", flush=True)
                
                # Record watch history
                try:
                    add_watch_history(user_id, movie_id, video_id, movie_title)
                except Exception as watch_error:
                    print(f"Error recording watch history (non-critical): {watch_error}", flush=True)
                
                return jsonify({
                    'success': True,
                    'trailer_url': trailer_url,
                    'video_id': video_id,
                    'movie_title': movie_title,
                    'source_transcription': cached_data['source_transcription'],
                    'japanese_translation': cached_data['japanese_translation'],
                    'detected_terms': cached_data['detected_terms'],  # Cultural terms from cache
                    'language': cached_data['language'],
                    'output_video': output_video.split('/')[-1] if output_video else None,  # Just filename for serving
                    'cached': True,
                    'message': 'Trailer data retrieved from cache!'
                })
            else:
                # Cache hit (stale) - version mismatch, ignore cache and run pipeline
                print(f"⚠️  CACHE HIT (stale, ignored): Found cached data but pipeline_version mismatch (cached={cached_version}, current={PIPELINE_VERSION})", flush=True)
                print(f"   Ignoring stale cache and running full translation pipeline...", flush=True)
                cached_data = None  # Treat as cache miss to run pipeline
        
        if not cached_data:
            # Cache miss or stale - process the video
            if cached_data is None:
                print(f"✗ CACHE MISS: No cache found for movie_id={movie_id}, processing...", flush=True)
            # (If cached_data was set to None above, we already logged the stale cache message)
        
        # Check for cached raw video first (avoid re-downloading)
        cached_raw_video = get_cached_raw_video(video_id)
        actual_video_path = None
        video_path = None  # Initialize to avoid UnboundLocalError
        
        if cached_raw_video:
            print(f"✓ Using cached raw video, skipping download", flush=True)
            actual_video_path = cached_raw_video
        else:
            # Need to download video
            print(f"Downloading video (this may take a while)...", flush=True)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
                video_path = tmp_video.name
            
            try:
                downloaded_file = download_video(trailer_url, video_path)
            except Exception as download_error:
                raise Exception(f"Failed to download video: {str(download_error)}")
            
            # Verify the downloaded file
            if not os.path.exists(downloaded_file):
                raise Exception(f"Downloaded video file not found: {downloaded_file}")
            
            file_size = os.path.getsize(downloaded_file)
            if file_size == 0:
                raise Exception(f"Downloaded video file is empty (0 bytes): {downloaded_file}")
            
            # Cache the raw video for future use
            save_cached_raw_video(video_id, downloaded_file)
            
            # Remux/repair the video file
            try:
                from backend_python.subtitle_processor import verify_and_repair_video
                with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_repaired:
                    repaired_path = tmp_repaired.name
                actual_video_path = verify_and_repair_video(downloaded_file, repaired_path)
            except Exception as repair_error:
                print(f"Video repair failed, using original: {repair_error}", flush=True)
                actual_video_path = downloaded_file
        
        # Process video with subtitles
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            output_path = tmp_output.name
        
        try:
            result = process_trailer_video(actual_video_path, output_path, language=language)
        except ValueError as ve:
            error_msg = str(ve)
            if "OPENAI_API_KEY" in error_msg:
                return jsonify({
                    'error': 'OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file or environment variables.'
                }), 500
            # Check if it's a generic content detection error
            if "generic/demo content" in error_msg.lower() or "generic content" in error_msg.lower():
                return jsonify({
                    'error': error_msg,
                    'error_type': 'generic_content'
                }), 400
            raise
        
        # Save to cache (including video file)
        try:
            terms_count = 0
            if result.get('detected_terms') and isinstance(result.get('detected_terms'), dict):
                terms_count = len(result['detected_terms'].get('terms', []))
            
            save_cached_movie(
                movie_id=movie_id,
                video_id=video_id,
                source_transcription=result['source_transcription'],
                japanese_translation=result['japanese_translation'],
                detected_terms=result.get('detected_terms'),  # Cultural terms saved here
                language=result.get('language'),
                segments=result.get('segments'),
                processed_video_path=result['output_video'],
                pipeline_version=PIPELINE_VERSION
            )
            print(f"✓ Saved processed movie to cache: movie_id={movie_id} (including {terms_count} cultural terms)", flush=True)
        except Exception as cache_error:
            print(f"Error saving to cache (non-critical): {cache_error}", flush=True)
        
        # Save to dictionary (existing functionality)
        try:
            # Check if dictionary already exists for this user and video_id
            with get_db() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute(
                        "SELECT id FROM dictionaries WHERE user_id = %s AND youtube_video_id = %s",
                        (user_id, video_id)
                    )
                    existing_dict = cursor.fetchone()
                    
                    if existing_dict:
                        dict_id = existing_dict['id']
                        update_dictionary_transcription(
                            dict_id,
                            result['source_transcription'],
                            result['japanese_translation']
                        )
                    else:
                        dict_id = add_dictionary_entry(user_id, video_id, movie_title)
                        update_dictionary_transcription(
                            dict_id,
                            result['source_transcription'],
                            result['japanese_translation']
                        )
            
            # Save detected terms as words
            if result.get('detected_terms'):
                save_words_to_dictionary(dict_id, result['detected_terms'])
        except Exception as db_error:
            print(f"Database error (non-critical): {db_error}", flush=True)
        
        # Clean up temporary files (but keep cached ones)
        if actual_video_path and actual_video_path != cached_raw_video:
            if os.path.exists(actual_video_path) and actual_video_path != result.get('output_video'):
                try:
                    os.remove(actual_video_path)
                except:
                    pass
        
        # Clean up video_path if it was created
        if video_path and os.path.exists(video_path) and video_path != result.get('output_video'):
            try:
                os.remove(video_path)
            except:
                pass
        
        # Record watch history
        try:
            add_watch_history(user_id, movie_id, video_id, movie_title)
        except Exception as watch_error:
            print(f"Error recording watch history (non-critical): {watch_error}", flush=True)
        
        return jsonify({
            'success': True,
            'trailer_url': trailer_url,
            'video_id': video_id,
            'movie_title': movie_title,
            'source_transcription': result['source_transcription'],
            'japanese_translation': result['japanese_translation'],
            'detected_terms': result.get('detected_terms'),
            'output_video': result['output_video'].split('/')[-1] if result.get('output_video') else None,
            'language': result.get('language'),
            'cached': False,
            'message': 'Trailer processed successfully with subtitles!'
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Processing error: {error_trace}", flush=True)
        return jsonify({'error': f'Failed to process trailer: {str(e)}'}), 500

@app.route('/dictionary/<int:dictionary_id>')
def dictionary_detail(dictionary_id):
    """Display dictionary with words - requires authentication and ownership"""
    if "user_id" not in session:
        return redirect(url_for("auth.login_form"))
    
    user_id = session.get('user_id')
    show_favorites_only = request.args.get('favorites_only') == '1'
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Verify dictionary ownership
            cursor.execute(
                "SELECT id, user_id, youtube_video_id, title, transcription, translation FROM dictionaries WHERE id = %s",
                (dictionary_id,)
            )
            dictionary = cursor.fetchone()
            
            if not dictionary:
                abort(404)
            
            if dictionary['user_id'] != user_id:
                abort(403)
            
            # Get words for this dictionary
            if show_favorites_only:
                cursor.execute("""
                    SELECT w.id, w.word, w.pronunciation_japanese, w.meaning_japanese, w.why_important
                    FROM words w
                    INNER JOIN favorites f ON w.id = f.word_id
                    WHERE w.dictionary_id = %s AND f.user_id = %s
                    ORDER BY w.word
                """, (dictionary_id, user_id))
            else:
                cursor.execute("""
                    SELECT w.id, w.word, w.pronunciation_japanese, w.meaning_japanese, w.why_important
                    FROM words w
                    WHERE w.dictionary_id = %s
                    ORDER BY w.word
                """, (dictionary_id,))
            
            words = cursor.fetchall()
            
            # Get favorited word IDs for this user
            if words:
                word_ids = [w['id'] for w in words]
                placeholders = ','.join(['%s'] * len(word_ids))
                cursor.execute(
                    f"SELECT word_id FROM favorites WHERE user_id = %s AND word_id IN ({placeholders})",
                    [user_id] + word_ids
                )
                favorited_ids = {row['word_id'] for row in cursor.fetchall()}
            else:
                favorited_ids = set()
            
            # Mark which words are favorited
            for word in words:
                word['is_favorited'] = word['id'] in favorited_ids
    
    return render_template("dictionary.html", dictionary=dictionary, words=words, show_favorites_only=show_favorites_only)

@app.route('/api/favorite', methods=['POST'])
def add_favorite():
    """Add word to favorites - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    data = request.get_json()
    word_id = data.get('word_id')
    
    if not word_id:
        return jsonify({'error': 'word_id is required'}), 400
    
    # Verify word exists and user has access to its dictionary
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            # Get word's dictionary
            cursor.execute(
                "SELECT dictionary_id FROM words WHERE id = %s",
                (word_id,)
            )
            word = cursor.fetchone()
            
            if not word:
                return jsonify({'error': 'Word not found'}), 404
            
            # Verify dictionary ownership
            cursor.execute(
                "SELECT user_id FROM dictionaries WHERE id = %s",
                (word['dictionary_id'],)
            )
            dictionary = cursor.fetchone()
            
            if not dictionary or dictionary['user_id'] != user_id:
                return jsonify({'error': 'Unauthorized'}), 403
            
            # Check if already favorited
            cursor.execute(
                "SELECT word_id FROM favorites WHERE user_id = %s AND word_id = %s",
                (user_id, word_id)
            )
            existing = cursor.fetchone()
            
            if existing:
                return jsonify({'success': True, 'message': 'Already in favorites'})
            
            # Insert favorite
            try:
                cursor.execute(
                    "INSERT INTO favorites (user_id, word_id) VALUES (%s, %s)",
                    (user_id, word_id)
                )
                conn.commit()
                return jsonify({'success': True, 'message': 'Added to favorites'})
            except Exception as e:
                # If unique constraint violation, it's already favorited
                conn.rollback()
                return jsonify({'success': True, 'message': 'Already in favorites'})

@app.route('/api/unfavorite', methods=['POST'])
def remove_favorite():
    """Remove word from favorites - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    data = request.get_json()
    word_id = data.get('word_id')
    
    if not word_id:
        return jsonify({'error': 'word_id is required'}), 400
    
    with get_db() as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                "DELETE FROM favorites WHERE user_id = %s AND word_id = %s",
                (user_id, word_id)
            )
            conn.commit()
            return jsonify({'success': True, 'message': 'Removed from favorites'})

@app.route('/dictionaries')
def dictionaries_list():
    """List all dictionaries for current user - requires authentication"""
    if "user_id" not in session:
        return redirect(url_for("auth.login_form"))
    
    user_id = session.get('user_id')
    
    with get_db() as conn:
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("""
                SELECT d.id, d.youtube_video_id, d.title, d.created_at,
                       COUNT(w.id) as word_count
                FROM dictionaries d
                LEFT JOIN words w ON d.id = w.dictionary_id
                WHERE d.user_id = %s
                GROUP BY d.id, d.youtube_video_id, d.title, d.created_at
                ORDER BY d.created_at DESC
            """, (user_id,))
            dictionaries = cursor.fetchall()
    
    return render_template("dictionaries.html", dictionaries=dictionaries)

@app.route('/watch-history')
def watch_history_page():
    """Display watch history for current user - requires authentication"""
    if "user_id" not in session:
        return redirect(url_for("auth.login_form"))
    
    user_id = session.get('user_id')
    history = get_watch_history(user_id, limit=100)
    
    # Enrich history with movie details from TMDB
    from backend_python.tmdb import get_movie_details
    enriched_history = []
    for entry in history:
        movie_details = get_movie_details(entry['movie_id'])
        entry_dict = dict(entry)
        if movie_details:
            entry_dict['poster_path'] = movie_details.get('poster_path')
            entry_dict['release_date'] = movie_details.get('release_date')
            entry_dict['vote_average'] = movie_details.get('vote_average')
        enriched_history.append(entry_dict)
    
    return render_template("watch_history.html", history=enriched_history)

@app.route('/api/watch-history', methods=['GET'])
def get_watch_history_api():
    """Get watch history as JSON - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    limit = request.args.get('limit', 50, type=int)
    history = get_watch_history(user_id, limit=limit)
    
    return jsonify({'success': True, 'history': history})

@app.route('/api/watch-history/<int:entry_id>', methods=['DELETE'])
def delete_watch_history_entry_api(entry_id):
    """Delete a watch history entry - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    success = delete_watch_history_entry(user_id, entry_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Watch history entry deleted'})
    else:
        return jsonify({'error': 'Entry not found or unauthorized'}), 404

@app.route('/api/watch-history/clear', methods=['POST'])
def clear_watch_history_api():
    """Clear all watch history - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    success = clear_watch_history(user_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Watch history cleared'})
    else:
        return jsonify({'error': 'Failed to clear watch history'}), 500

@app.route('/watch-later')
def watch_later_page():
    """Display watch later list for current user - requires authentication"""
    if "user_id" not in session:
        return redirect(url_for("auth.login_form"))
    
    user_id = session.get('user_id')
    watch_later_list = get_watch_later(user_id, limit=200)
    
    # Enrich with movie details from TMDB
    from backend_python.tmdb import get_movie_details
    enriched_list = []
    for entry in watch_later_list:
        movie_details = get_movie_details(entry['movie_id'])
        entry_dict = dict(entry)
        if movie_details:
            entry_dict['poster_path'] = movie_details.get('poster_path')
            entry_dict['release_date'] = movie_details.get('release_date')
            entry_dict['vote_average'] = movie_details.get('vote_average')
            entry_dict['overview'] = movie_details.get('overview')
        enriched_list.append(entry_dict)
    
    return render_template("watch_later.html", watch_later=enriched_list)

@app.route('/api/watch-later', methods=['GET'])
def get_watch_later_api():
    """Get watch later list as JSON - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    limit = request.args.get('limit', 100, type=int)
    watch_later_list = get_watch_later(user_id, limit=limit)
    
    return jsonify({'success': True, 'watch_later': watch_later_list})

@app.route('/api/watch-later/check/<int:movie_id>', methods=['GET'])
def check_watch_later_api(movie_id):
    """Check if movie is in watch later - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    in_list = is_in_watch_later(user_id, movie_id)
    
    return jsonify({'success': True, 'in_watch_later': in_list})

@app.route('/api/watch-later', methods=['POST'])
def add_watch_later_api():
    """Add movie to watch later - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    data = request.get_json()
    movie_id = data.get('movie_id')
    movie_title = data.get('movie_title')
    
    if not movie_id:
        return jsonify({'error': 'movie_id is required'}), 400
    
    success = add_to_watch_later(user_id, movie_id, movie_title)
    
    if success:
        return jsonify({'success': True, 'message': 'Added to watch later'})
    else:
        return jsonify({'error': 'Failed to add to watch later'}), 500

@app.route('/api/watch-later/<int:movie_id>', methods=['DELETE'])
def remove_watch_later_api(movie_id):
    """Remove movie from watch later - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    success = remove_from_watch_later(user_id, movie_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Removed from watch later'})
    else:
        return jsonify({'error': 'Movie not in watch later'}), 404

@app.route('/api/watch-later/entry/<int:entry_id>', methods=['DELETE'])
def delete_watch_later_entry_api(entry_id):
    """Delete a watch later entry - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    success = delete_watch_later_entry(user_id, entry_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Watch later entry deleted'})
    else:
        return jsonify({'error': 'Entry not found or unauthorized'}), 404

@app.route('/api/watch-later/clear', methods=['POST'])
def clear_watch_later_api():
    """Clear all watch later entries - requires authentication"""
    if "user_id" not in session:
        return jsonify({'error': 'Authentication required'}), 401
    
    user_id = session.get('user_id')
    success = clear_watch_later(user_id)
    
    if success:
        return jsonify({'success': True, 'message': 'Watch later cleared'})
    else:
        return jsonify({'error': 'Failed to clear watch later'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

