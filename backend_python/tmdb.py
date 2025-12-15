from dotenv import load_dotenv
from pathlib import Path
import requests
import os

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
#load from current directory as fallback
load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_SEARCH_URL = os.environ.get("TMDB_SEARCH_URL")
TMDB_IMAGE_BASE_URL = os.environ.get("TMDB_IMAGE_BASE_URL")

# Languages supported by Whisper API
SUPPORTED_LANGUAGES = {'hi', 'kn', 'ta'}  # Hindi, Kannada, Tamil

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
        # Filter out Malayalam and only include Whisper-supported languages
        original_language = item.get("original_language", "")
        if original_language not in SUPPORTED_LANGUAGES:
            continue  # Skip this movie if language is not supported
        
        poster_path = item.get("poster_path")
        title = item.get("title")
        movie_id = item.get("id")
        if poster_path and title and movie_id:
            movies.append({
                "id": movie_id,
                "title": title,
                "api_url": f"{TMDB_IMAGE_BASE_URL}{poster_path}",
                "original_language": original_language  # Include for reference
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
        # Filter out Malayalam and only include Whisper-supported languages
        original_language = item.get("original_language", "")
        if original_language not in SUPPORTED_LANGUAGES:
            continue  # Skip this movie if language is not supported
        
        poster_path = item.get("poster_path")
        title = item.get("title")
        movie_id = item.get("id")
        if poster_path and title and movie_id:
            movies.append({
                "id": movie_id,
                "title": title,
                "api_url": f"{TMDB_IMAGE_BASE_URL}{poster_path}",
                "original_language": original_language  # Include for reference
            })
    return movies

def get_movie_details(movie_id):
    """Fetch detailed information about a movie from TMDB"""
    if not TMDB_API_KEY:
        print("Error: TMDB_API_KEY not set", flush=True)
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        
        # Check if language is supported before returning
        original_language = data.get("original_language", "")
        if original_language not in SUPPORTED_LANGUAGES:
            return None  # Return None for unsupported languages (e.g., Malayalam)
        
        return {
            "id": data.get("id"),
            "title": data.get("title"),
            "overview": data.get("overview"),
            "release_date": data.get("release_date"),
            "poster_path": f"{TMDB_IMAGE_BASE_URL}{data.get('poster_path')}" if data.get("poster_path") else None,
            "backdrop_path": f"{TMDB_IMAGE_BASE_URL}{data.get('backdrop_path')}" if data.get("backdrop_path") else None,
            "vote_average": data.get("vote_average"),
            "genres": [genre.get("name") for genre in data.get("genres", [])],
            "original_language": original_language,
        }
    except Exception as e:
        print(f"Error fetching movie details: {e}", flush=True)
        return None

def get_movie_trailer(movie_id):
    """Fetch trailer URL for a movie from TMDB"""
    if not TMDB_API_KEY:
        print("Error: TMDB_API_KEY not set", flush=True)
        return None
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        
        # Look for YouTube trailer
        for video in data.get("results", []):
            if video.get("type") == "Trailer" and video.get("site") == "YouTube":
                return f"https://www.youtube.com/watch?v={video.get('key')}"
        
        # If no trailer, return first YouTube video
        for video in data.get("results", []):
            if video.get("site") == "YouTube":
                return f"https://www.youtube.com/watch?v={video.get('key')}"
        
        return None
    except Exception as e:
        print(f"Error fetching trailer: {e}", flush=True)
        return None