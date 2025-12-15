from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

_client = None

def get_client():
    """Lazy initialization of OpenAI client"""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            env_file = Path(__file__).parent.parent / '.env'
            error_msg = (
                "OPENAI_API_KEY environment variable is not set.\n"
                f"Please set it in your environment or create a .env file at: {env_file}\n"
                "Add this line to your .env file: OPENAI_API_KEY=your_api_key_here"
            )
            raise ValueError(error_msg)
        _client = OpenAI(api_key=api_key)
    return _client

def transcribe_audio(file_path: str, language: str = None) -> str:
    """
    Transcribe audio and return text.
    
    Args:
        file_path: Path to audio file
        language: Language code (hi, kn, ta). If None, auto-detect.
    """
    client = get_client()
    with open(file_path, "rb") as f:
        params = {
            "file": f,
            "model": "whisper-1",
        }
        if language:
            params["language"] = language
        
        res = client.audio.transcriptions.create(**params)
    return res.text

def transcribe_audio_with_timestamps(file_path: str, language: str = None) -> dict:
    """
    Transcribe audio with word-level timestamps.
    
    Args:
        file_path: Path to audio file
        language: Language code (hi, kn, ta). If None, auto-detect.
    """
    client = get_client()
    with open(file_path, "rb") as f:
        params = {
            "file": f,
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"]
        }
        if language:
            params["language"] = language
        
        res = client.audio.transcriptions.create(**params)
    return res