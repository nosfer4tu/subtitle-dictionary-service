from dotenv import load_dotenv
from pathlib import Path
import requests
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
#load from current directory as fallback
load_dotenv()
TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_SEARCH_URL = os.environ.get("TMDB_SEARCH_URL")
TMDB_IMAGE_BASE_URL = os.environ.get("TMDB_IMAGE_BASE_URL")

# Languages supported by Whisper API
SUPPORTED_LANGUAGES = {'hi', 'kn', 'ta'}  # Hindi, Kannada, Tamil

def has_trailer(movie_id):
    """
    Check if a movie has a trailer by querying TMDB API directly.
    No caching - checks every time.
    """
    if not TMDB_API_KEY:
        return False
    
    url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos"
    params = {
        "api_key": TMDB_API_KEY,
        "language": "en-US",
    }
    try:
        res = requests.get(url, params=params)
        res.raise_for_status()
        data = res.json()
        
        # Check for YouTube trailer or any YouTube video
        for video in data.get("results", []):
            if video.get("site") == "YouTube":
                return True
        
        return False
    except Exception as e:
        print(f"Error checking trailer for movie {movie_id}: {e}", flush=True)
        return False

def check_trailers_parallel(movie_ids, max_workers=10):
    """
    Check trailers for multiple movies in parallel.
    
    Args:
        movie_ids: List of movie IDs to check
        max_workers: Maximum number of concurrent requests (default: 10)
    
    Returns:
        Dictionary mapping movie_id to boolean (True if has trailer, False otherwise)
    """
    results = {}
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all trailer checks
        future_to_movie_id = {
            executor.submit(has_trailer, movie_id): movie_id 
            for movie_id in movie_ids
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_movie_id):
            movie_id = future_to_movie_id[future]
            try:
                results[movie_id] = future.result()
            except Exception as e:
                print(f"Error checking trailer for movie {movie_id}: {e}", flush=True)
                results[movie_id] = False
    
    return results

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
    
    
    
    # First, collect candidate movies (after language filter)
    candidate_movies = []
    movie_ids_to_check = []
    
    for item in data.get("results", []):
        # Filter out Malayalam and only include Whisper-supported languages
        original_language = item.get("original_language", "")
        if original_language not in SUPPORTED_LANGUAGES:
            continue  # Skip this movie if language is not supported
        
        poster_path = item.get("poster_path")
        title = item.get("title")
        movie_id = item.get("id")
        
        if poster_path and title and movie_id:
            candidate_movies.append({
                "id": movie_id,
                "title": title,
                "api_url": f"{TMDB_IMAGE_BASE_URL}{poster_path}",
                "original_language": original_language
            })
            movie_ids_to_check.append(movie_id)
    
    # Check trailers in parallel
    if movie_ids_to_check:
        trailer_results = check_trailers_parallel(movie_ids_to_check, max_workers=10)
        
        # Only include movies that have trailers
        movies = []
        for movie in candidate_movies:
            if trailer_results.get(movie["id"], False):
                movies.append(movie)
        
        return movies
    
    return []

def fetch_indian_movies(page=1, movies_per_page=40):
    """
    Fetch Indian movies from TMDB, filtering by supported languages and trailers.
    Checks trailers directly from API (no caching).
    Limited to 40 movies.
    
    Args:
        page: Logical page number (1-indexed) for filtered results (default: 1)
        movies_per_page: Number of movies to show per page (default: 40)
    
    Returns:
        dict with 'movies' list, 'current_page', 'total_pages', and 'has_next'
    """
    if not TMDB_API_KEY or not TMDB_SEARCH_URL or not TMDB_IMAGE_BASE_URL:
        print("Error: TMDB environment variables not set", flush=True)
        return {"movies": [], "current_page": 1, "total_pages": 1, "has_next": False}
    
    url = "http://api.themoviedb.org/3/discover/movie"
    MAX_TMDB_PAGES = 10  # Limit to 10 TMDB pages maximum
    MAX_MOVIES = 40  # Limit to 40 movies total
    
    # Only allow page 1
    if page > 1:
        return {"movies": [], "current_page": 1, "total_pages": 1, "has_next": False}
    
    movies_collected = []
    candidate_movies = []  # Store all candidate movies (after language filter)
    current_tmdb_page = 1
    total_tmdb_pages = 1
    
    # First pass: Collect candidate movies (after language filter) from multiple TMDB pages
    # Fetch more candidates than needed to account for movies without trailers
    while len(movies_collected) < MAX_MOVIES and current_tmdb_page <= MAX_TMDB_PAGES:
        params = {
            "api_key": TMDB_API_KEY,
            "with_origin_country": "IN",
            "sort_by": "popularity.desc",
            "language": "en-US",
            "page": current_tmdb_page,
        }
        
        try:
            res = requests.get(url, params=params)
            res.raise_for_status()
            data = res.json()
            
            # Get total pages from first response
            if current_tmdb_page == 1:
                total_tmdb_pages = min(data.get("total_pages", 1), MAX_TMDB_PAGES)
            
            # Collect candidate movies (after language filter)
            for item in data.get("results", []):
                # Filter out unsupported languages
                original_language = item.get("original_language", "")
                if original_language not in SUPPORTED_LANGUAGES:
                    continue
                
                poster_path = item.get("poster_path")
                title = item.get("title")
                movie_id = item.get("id")
                
                if poster_path and title and movie_id:
                    candidate_movies.append({
                        "id": movie_id,
                        "title": title,
                        "api_url": f"{TMDB_IMAGE_BASE_URL}{poster_path}",
                        "original_language": original_language
                    })
            
            current_tmdb_page += 1
            
            # Stop if we've reached the max TMDB pages
            if current_tmdb_page > total_tmdb_pages:
                break
                
        except Exception as e:
            print(f"Error fetching TMDB page {current_tmdb_page}: {e}", flush=True)
            break
    
    # Second pass: Check trailers in parallel (no cache)
    if candidate_movies:
        movie_ids_to_check = [movie["id"] for movie in candidate_movies]
        
        # Check trailers in parallel (up to 10 concurrent requests)
        trailer_results = check_trailers_parallel(movie_ids_to_check, max_workers=10)
        
        # Collect movies that have trailers
        for movie in candidate_movies:
            movie_id = movie["id"]
            if trailer_results.get(movie_id, False):
                movies_collected.append(movie)
            
            # Stop if we have enough movies
            if len(movies_collected) >= MAX_MOVIES:
                break
    
    # Limit to MAX_MOVIES
    page_movies = movies_collected[:MAX_MOVIES]
    
    return {
        "movies": page_movies,
        "current_page": 1,
        "total_pages": 1,
        "has_next": False
    }

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
    """
    Fetch trailer URL for a movie from TMDB.
    Prioritizes official trailers and tries multiple options to avoid generic/demo content.
    """
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
        
        videos = data.get("results", [])
        if not videos:
            return None
        
        # Strategy 1: Prefer official trailers (type="Trailer" and official=True)
        official_trailers = [
            v for v in videos 
            if v.get("type") == "Trailer" 
            and v.get("site") == "YouTube"
            and v.get("official", False)
        ]
        if official_trailers:
            # Sort by published_at (newer first) and pick the first one
            official_trailers.sort(key=lambda x: x.get("published_at", ""), reverse=True)
            trailer = official_trailers[0]
            print(f"Selected official trailer: {trailer.get('name', 'Unknown')} (published: {trailer.get('published_at', 'Unknown')})", flush=True)
            return f"https://www.youtube.com/watch?v={trailer.get('key')}"
        
        # Strategy 2: Any trailer (type="Trailer")
        trailers = [
            v for v in videos 
            if v.get("type") == "Trailer" 
            and v.get("site") == "YouTube"
        ]
        if trailers:
            # Sort by published_at (newer first) and pick the first one
            trailers.sort(key=lambda x: x.get("published_at", ""), reverse=True)
            trailer = trailers[0]
            print(f"Selected trailer: {trailer.get('name', 'Unknown')} (published: {trailer.get('published_at', 'Unknown')})", flush=True)
            return f"https://www.youtube.com/watch?v={trailer.get('key')}"
        
        # Strategy 3: Teaser trailers (type="Teaser")
        teasers = [
            v for v in videos 
            if v.get("type") == "Teaser" 
            and v.get("site") == "YouTube"
        ]
        if teasers:
            teasers.sort(key=lambda x: x.get("published_at", ""), reverse=True)
            teaser = teasers[0]
            print(f"Selected teaser: {teaser.get('name', 'Unknown')} (published: {teaser.get('published_at', 'Unknown')})", flush=True)
            return f"https://www.youtube.com/watch?v={teaser.get('key')}"
        
        # Strategy 4: Any YouTube video (last resort)
        youtube_videos = [
            v for v in videos 
            if v.get("site") == "YouTube"
        ]
        if youtube_videos:
            youtube_videos.sort(key=lambda x: x.get("published_at", ""), reverse=True)
            video = youtube_videos[0]
            print(f"Selected YouTube video (fallback): {video.get('name', 'Unknown')} (type: {video.get('type', 'Unknown')})", flush=True)
            return f"https://www.youtube.com/watch?v={video.get('key')}"
        
        return None
    except Exception as e:
        print(f"Error fetching trailer: {e}", flush=True)
        return None