"""
Script to initialize the movie_cache table.
Run this once: python -m backend_python.init_movie_cache
"""
from backend_python.movie_cache import create_movie_cache_table

if __name__ == '__main__':
    print("Creating movie_cache table...")
    try:
        create_movie_cache_table()
        print("✓ Movie cache table created successfully!")
    except Exception as e:
        print(f"✗ Error: {e}")
