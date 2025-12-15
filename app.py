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
import tempfile

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
#load from current directory as fallback
load_dotenv()

app = Flask(__name__, template_folder='./frontend')
app.secret_key = os.environ.get("SECRET_KEY")
app.register_blueprint(auth_blueprint)

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
        return render_template("error.html", message="Movie not found"), 404
    
    trailer_url = get_movie_trailer(movie_id)
    return render_template("movie_detail.html", movie=movie, trailer_url=trailer_url)

@app.route('/api/video/<path:filename>')
def serve_video(filename):
    """Serve processed video files"""
    # Security: Only allow files from temp directory
    import tempfile
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, filename)
    
    if os.path.exists(video_path) and video_path.startswith(temp_dir):
        from flask import send_file
        return send_file(video_path, mimetype='video/mp4')
    else:
        return jsonify({'error': 'Video not found'}), 404

@app.route('/api/upload-trailer', methods=['POST'])
def upload_trailer():
    """Handle trailer upload with transcription, translation, and subtitles"""
    try:
        data = request.get_json()
        movie_id = data.get('movie_id')
        movie_title = data.get('movie_title', 'Unknown Movie')
        user_id = session.get('user_id', 1)  # Default to 1 if not logged in
        
        if not movie_id:
            return jsonify({'error': 'Movie ID is required'}), 400
        
        # Get movie details to detect language
        movie = get_movie_details(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found or language not supported'}), 404
        
        # Get language from movie details
        language = movie.get('original_language')  # This gets the language code (hi, kn, ta, etc.)
        
        # Get trailer URL from TMDB
        trailer_url = get_movie_trailer(movie_id)
        
        if not trailer_url:
            return jsonify({'error': 'No trailer found for this movie'}), 404
        
        # Extract YouTube video ID
        try:
            video_id = extract_video_id(trailer_url)
        except ValueError:
            return jsonify({'error': 'Invalid trailer URL'}), 400
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_video:
            video_path = tmp_video.name
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_output:
            output_path = tmp_output.name
        
        try:
            # Step 1: Download the video
            try:
                downloaded_file = download_video(trailer_url, video_path)
            except Exception as download_error:
                raise Exception(f"Failed to download video: {str(download_error)}")
            
            # Verify the downloaded file exists and is valid
            if not os.path.exists(downloaded_file):
                raise Exception(f"Downloaded video file not found: {downloaded_file}. Please check if yt-dlp downloaded successfully.")
            
            # Check file size
            file_size = os.path.getsize(downloaded_file)
            if file_size == 0:
                raise Exception(f"Downloaded video file is empty (0 bytes): {downloaded_file}")
            
            # Remux/repair the video file to ensure it's a valid MP4
            # This fixes issues with incomplete downloads or corrupted moov atoms
            try:
                from backend_python.subtitle_processor import verify_and_repair_video
                actual_video_path = verify_and_repair_video(downloaded_file, video_path)
            except Exception as repair_error:
                # If repair fails, try using original file
                print(f"Video repair failed, using original: {repair_error}", flush=True)
                actual_video_path = downloaded_file
            
            # Step 2: Process video with subtitles (pass language parameter)
            try:
                result = process_trailer_video(actual_video_path, output_path, language=language)
            except ValueError as ve:
                # Handle missing API key error with helpful message
                if "OPENAI_API_KEY" in str(ve):
                    return jsonify({
                        'error': 'OpenAI API key is not configured. Please set OPENAI_API_KEY in your .env file or environment variables.'
                    }), 500
                raise
            
            # Step 3: Save to dictionary
            try:
                dict_id = add_dictionary_entry(user_id, video_id, movie_title)
                update_dictionary_transcription(
                    dict_id,
                    result['source_transcription'],
                    result['japanese_translation']
                )
            except Exception as db_error:
                print(f"Database error (non-critical): {db_error}", flush=True)
            
            # IMPORTANT: Don't delete the output video - it's needed for display
            # Only delete the temporary downloaded video, not the processed one
            if os.path.exists(video_path) and video_path != result['output_video']:
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
                'output_video': result['output_video'],
                'language': result.get('language'),
                'message': 'Trailer processed successfully with subtitles!'
            })
        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            print(f"Processing error: {error_trace}", flush=True)
            return jsonify({'error': f'Failed to process trailer: {str(e)}'}), 500
        finally:
            # Only clean up the temporary downloaded video, not the processed output
            # The processed video should be served or saved elsewhere
            if os.path.exists(video_path) and video_path != output_path:
                try:
                    os.remove(video_path)
                except:
                    pass
                
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Server error: {error_trace}", flush=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000)

