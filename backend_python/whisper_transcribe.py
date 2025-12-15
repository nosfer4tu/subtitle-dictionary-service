from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import re

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

def clean_transcription(text: str, language: str = None) -> str:
    """
    Clean and fix common transcription errors in Indian language text.
    
    Args:
        text: Transcribed text
        language: Language code (hi, kn, ta)
    """
    if not text:
        return text
    
    import unicodedata
    
    # Step 1: Remove specific problematic characters that are transcription errors
    # These are often isolated characters that don't form valid words
    problematic_chars = [
        '\u0B97',  # ஗ - common transcription error
        '\u0B9F',  # ட - sometimes appears as error
        '\u0B9E',  # ஞ
        '\u0BA3',  # ண
        '\u0BA4',  # த
        '\u0BA8',  # ந
        '\u0BAA',  # ப
        '\u0BAE',  # ம
        '\u0BAF',  # ய
        '\u0BB0',  # ர
        '\u0BB2',  # ல
        '\u0BB5',  # வ
        '\u0BB4',  # ழ
        '\u0BB3',  # ள
        '\u0BB1',  # ற
        '\u0BA9',  # ன
        '\u0B99',  # ங
        '\u0B9C',  # ஜ
        '\u0B9D',  # ஝
        '\u0B9B',  # ஛
        '\u0B9A',  # ச
    ]
    
    # Actually, don't remove all these - they might be valid. Instead, remove isolated ones
    # Remove isolated single characters that are likely errors (not part of words)
    import re
    
    # Remove isolated Tamil characters surrounded by spaces or punctuation
    # Split into separate patterns to avoid variable-width lookbehind issue
    # Pattern 1: Start of string + Tamil char + space or end
    text = re.sub(r'^[\u0B80-\u0BFF](?=\s|$|[^\u0B80-\u0BFF])', '', text, flags=re.MULTILINE)
    # Pattern 2: Space + Tamil char + space or end or non-Tamil
    text = re.sub(r'(?<=\s)[\u0B80-\u0BFF](?=\s|$|[^\u0B80-\u0BFF])', '', text)
    
    # Remove the specific problematic character ஗ when it appears isolated
    text = re.sub(r'\s*஗\s*', ' ', text)
    
    # Step 2: Remove control characters and non-printable characters
    cleaned_chars = []
    for char in text:
        category = unicodedata.category(char)
        # Keep letters, numbers, punctuation, spaces, and common symbols
        if category[0] in 'LNP' or category == 'Zs' or char in '\n\t':
            try:
                # Normalize the character
                normalized = unicodedata.normalize('NFKC', char)
                normalized.encode('utf-8')
                cleaned_chars.append(normalized)
            except:
                continue
        elif category[0] == 'C':
            # Control characters - replace with space (except newline/tab)
            if char not in '\n\t':
                cleaned_chars.append(' ')
        else:
            cleaned_chars.append(char)
    
    text = ''.join(cleaned_chars)
    
    # Step 3: Fix common transcription error patterns
    fixes = {
        '  ': ' ',  # Multiple spaces
        '\n\n\n': '\n',  # Multiple newlines
    }
    
    for old, new in fixes.items():
        text = text.replace(old, new)
    
    # Step 4: Remove sequences of corrupted characters
    # Remove patterns like "஗" appearing in the middle of words
    # This is a heuristic: if we see a character that's not followed by a valid continuation, it might be an error
    # But be conservative - only remove obvious errors
    
    # Clean up extra spaces
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n\s*\n', '\n', text)
    text = text.strip()
    
    return text

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
            "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",  # Updated prompt
        }
        if language:
            params["language"] = language
        
        res = client.audio.transcriptions.create(**params)
        text = res.text
    
    # Clean the transcription
    cleaned_text = clean_transcription(text, language)
    return cleaned_text

# Script detection - character ranges (separate from language)
SCRIPT_RANGES = {
    'Tamil': (0x0B80, 0x0BFF),  # Tamil script range
    'Kannada': (0x0C80, 0x0CFF),  # Kannada script range
    'Devanagari': (0x0900, 0x097F),  # Devanagari (Hindi) script range
}

def detect_script_from_text(text: str) -> str:
    """
    Detect the script (Unicode range) of text.
    This is SEPARATE from language detection.
    Returns script name: 'Tamil', 'Kannada', or 'Devanagari'
    """
    if not text or not text.strip():
        return 'Unknown'
    
    script_counts = {'Tamil': 0, 'Kannada': 0, 'Devanagari': 0}
    
    for char in text:
        code_point = ord(char)
        for script_name, (start, end) in SCRIPT_RANGES.items():
            if start <= code_point <= end:
                script_counts[script_name] += 1
                break
    
    detected_script = max(script_counts.items(), key=lambda x: x[1])[0]
    if script_counts[detected_script] > 5:
        return detected_script
    return 'Unknown'

def transcribe_audio_with_timestamps(file_path: str, language: str = None) -> dict:
    """
    Transcribe audio with word-level timestamps.
    Uses Whisper's automatic language detection per segment.
    
    Args:
        file_path: Path to audio file
        language: Language code (hi, kn, ta). If None, auto-detect per segment.
    
    Returns:
        Transcription result with segments containing detected language
    """
    client = get_client()
    with open(file_path, "rb") as f:
        params = {
            "file": f,
            "model": "whisper-1",
            "response_format": "verbose_json",
            "timestamp_granularities": ["segment"],
            "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
        }
        # DO NOT pass language parameter - let Whisper auto-detect per segment
        # This allows handling code-mixing (multiple languages in one trailer)
        
        res = client.audio.transcriptions.create(**params)
        
        # Map Whisper language codes to our codes
        LANGUAGE_CODE_MAP = {
            'hi': 'hi',  # Hindi
            'kn': 'kn',  # Kannada
            'ta': 'ta',  # Tamil
            'te': 'kn',  # Telugu (map to Kannada if needed)
            'mr': 'hi',  # Marathi (map to Hindi if needed)
        }
        
        # Process segments and add detected language
        if hasattr(res, 'segments') and res.segments:
            for segment in res.segments:
                # Get detected language from Whisper
                detected_lang = None
                if hasattr(segment, 'language'):
                    detected_lang = segment.language
                elif isinstance(segment, dict) and 'language' in segment:
                    detected_lang = segment['language']
                
                # Map to our language codes
                if detected_lang:
                    detected_lang = LANGUAGE_CODE_MAP.get(detected_lang, detected_lang)
                    # Store in segment
                    if isinstance(segment, dict):
                        segment['detected_language'] = detected_lang
                    else:
                        # For object format, we'll need to handle differently
                        segment.detected_language = detected_lang
                
                # Clean transcription
                if hasattr(segment, 'text'):
                    segment.text = clean_transcription(segment.text, detected_lang)
                elif isinstance(segment, dict) and 'text' in segment:
                    segment['text'] = clean_transcription(segment['text'], detected_lang)
        
        # Clean overall text
        if hasattr(res, 'text'):
            res.text = clean_transcription(res.text, None)
        
        return res