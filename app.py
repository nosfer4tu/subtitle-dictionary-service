from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import requests
from psycopg2.extras import RealDictCursor
import os
from contextlib import contextmanager
from dotenv import load_dotenv
from pathlib import Path
from backend_python.auth import auth_blueprint
from backend_python.tmdb import search_movies, fetch_indian_movies, get_movie_trailer, get_movie_details
from backend_python.youtube_upload import download_video, extract_video_id
from backend_python.subtitle_processor import process_trailer_video
from backend_python.dictionary import add_dictionary_entry, update_dictionary_transcription
from backend_python.movie_cache import (
    get_cached_movie, save_cached_movie, 
    get_cached_raw_video, save_cached_raw_video,
    get_cached_video_path,
    create_movie_cache_table  # Add this import
)
import tempfile
import shutil  # Add shutil import for file operations

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

app = Flask(__name__, template_folder='./frontend')
app.secret_key = os.environ.get("SECRET_KEY")
app.register_blueprint(auth_blueprint)

# Initialize movie cache table on startup
try:
    create_movie_cache_table()
except Exception as e:
    print(f"Warning: Could not initialize movie cache table: {e}", flush=True)

@app.route('/')
def index():
    # if "user_id" not in session:
    #     return redirect(url_for("login"))
    
    
        query = request.args.get("query")
        movies = search_movies(query) if query else []
        indian_movies = fetch_indian_movies()
        return render_template("index.html", movies=movies, query=query or "", indian_movies=indian_movies)

@app.route('/movie/<int:movie_id>')
def movie_detail(movie_id):
    """Movie detail page that automatically uploads trailer"""
    movie = get_movie_details(movie_id)
    if not movie:
        return render_template("error.html", message="Movie not found or language not supported"), 404
    
    trailer_url = get_movie_trailer(movie_id)
    return render_template("movie_detail.html", movie=movie, trailer_url=trailer_url)

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
            print(f"✓ Found cached data for movie_id={movie_id}, video_id={video_id}", flush=True)
            
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
        
        # Not in cache, process the video
        print(f"✗ No cache found for movie_id={movie_id}, processing...", flush=True)
        
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
            if "OPENAI_API_KEY" in str(ve):
                return jsonify({
                    'error': 'OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file or environment variables.'
                }), 500
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
                processed_video_path=result['output_video']
            )
            print(f"✓ Saved processed movie to cache: movie_id={movie_id} (including {terms_count} cultural terms)", flush=True)
        except Exception as cache_error:
            print(f"Error saving to cache (non-critical): {cache_error}", flush=True)
        
        # Save to dictionary (existing functionality)
        try:
            dict_id = add_dictionary_entry(user_id, video_id, movie_title)
            update_dictionary_transcription(
                dict_id,
                result['source_transcription'],
                result['japanese_translation']
            )
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

