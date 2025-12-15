# subtitle-dictionary-service

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Variables

Copy the example environment file and fill in your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

- **OPENAI_API_KEY**: Get from https://platform.openai.com/api-keys
- **TMDB_API_KEY**: Get from https://www.themoviedb.org/settings/api
- **DATABASE_URL**: Your PostgreSQL connection string
- **SECRET_KEY**: A random string for Flask sessions (generate with: `python -c "import secrets; print(secrets.token_hex(32))"`)

### 3. Run the Application

```bash
python app.py
# or
flask run
```

The application will be available at `http://localhost:8000`

## Features

- Search movies using TMDB API
- Download movie trailers from YouTube
- Transcribe audio using OpenAI Whisper
- Translate Hindi to Japanese
- Generate subtitles and burn them into videos
