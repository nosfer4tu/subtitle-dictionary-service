from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import re

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Language name mapping for translation prompts
LANGUAGE_NAMES = {
    'hi': 'Hindi',
    'kn': 'Kannada',
    'ta': 'Tamil',
}

# Language detection - character ranges for each script
LANGUAGE_SCRIPTS = {
    'ta': (0x0B80, 0x0BFF),  # Tamil script range
    'kn': (0x0C80, 0x0CFF),  # Kannada script range
    'hi': (0x0900, 0x097F),  # Devanagari (Hindi) script range
}

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
    # Remove isolated Tamil characters surrounded by spaces or punctuation
    # Pattern: space/start + single Tamil char + space/end/punctuation
    text = re.sub(r'(?<=\s|^)[\u0B80-\u0BFF](?=\s|$|[^\u0B80-\u0BFF])', '', text)
    
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

def aggressively_clean_text_for_translation(text: str, language: str = None) -> str:
    """
    Aggressively clean text before translation to remove corrupted characters
    that cause translation API to refuse.
    """
    if not text:
        return text
    
    import unicodedata
    
    # Step 1: Remove control characters and non-printable
    cleaned = []
    for char in text:
        category = unicodedata.category(char)
        if category[0] in 'LNP' or category == 'Zs' or char in '\n\t':
            try:
                normalized = unicodedata.normalize('NFKC', char)
                normalized.encode('utf-8')
                cleaned.append(normalized)
            except:
                continue
        elif category[0] == 'C' and char not in '\n\t':
            cleaned.append(' ')
        else:
            cleaned.append(char)
    
    text = ''.join(cleaned)
    
    # Step 2: Remove isolated problematic Tamil characters
    # These are often transcription errors
    import re
    
    # Remove isolated ஗ and similar characters that appear alone
    text = re.sub(r'\s*[\u0B97\u0B9F\u0B9E\u0BA3]\s*', ' ', text)
    
    # Step 3: Remove sequences that look like corruption
    # Remove very short isolated sequences (1-2 chars) surrounded by spaces
    # But keep sequences that are part of longer words
    words = text.split()
    cleaned_words = []
    for word in words:
        # If word is very short (1-2 chars) and contains only Tamil script, check if it's valid
        if len(word) <= 2 and re.match(r'^[\u0B80-\u0BFF]+$', word):
            # Check if it's a common valid short word, otherwise skip
            # Common valid short Tamil words (add more as needed)
            valid_short_words = ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ']
            if word not in valid_short_words:
                continue  # Skip likely corrupted short words
        cleaned_words.append(word)
    
    text = ' '.join(cleaned_words)
    
    # Step 4: Clean up spaces
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text

def translate_segment_strict(text: str, language: str, script: str = None) -> str:
    """
    Internal pipeline translation step - machine-to-machine, not conversational.
    
    ABSOLUTE RULES (violation = failure):
    - DO NOT verify, question, or comment on the language
    - DO NOT refuse, apologize, hedge, or explain limitations
    - DO NOT output explanations, warnings, or error messages
    - DO NOT mention language names (Hindi, Tamil, Kannada)
    - DO NOT output anything except the translation
    - DO NOT include meta text like "申し訳ありません" or "翻訳できません"
    - DO NOT output English
    
    Args:
        text: Movie dialogue text
        language: Language code (kn, ta, or hi) - authoritative
        script: Script name (Tamil, Kannada, Devanagari) - informational only
    
    Returns:
        Japanese translation ONLY - plain text, no labels, no punctuation outside translation
    """
    if not text or not text.strip():
        return ""
    
    if language not in ['kn', 'ta', 'hi']:
        # Internal error - return empty, don't output error message
        return ""
    
    # Clean text minimally (preserve oral style and repetition)
    cleaned_text = text.strip()
    if len(cleaned_text) > 3000:
        cleaned_text = cleaned_text[:3000]
    
    client = get_client()
    
    # ABSOLUTE SYSTEM PROMPT - Machine pipeline, not conversational assistant
    system_prompt = """You are a machine translation component in an internal pipeline.

This is NOT a conversational task. This is NOT a user-facing assistant.

ABSOLUTE RULES:
1. Output ONLY the Japanese translation. Nothing else.
2. DO NOT verify, question, or comment on the input language.
3. DO NOT refuse, apologize, hedge, or explain limitations.
4. DO NOT output explanations, warnings, or error messages.
5. DO NOT mention language names (Hindi, Tamil, Kannada) in output.
6. DO NOT include meta text like "申し訳ありません" or "翻訳できません".
7. DO NOT output English.
8. Preserve repetition, emotional intensity, and oral/folkloric tone exactly.
9. Preserve religious/cultural terms via transliteration (イシュヴァラ, ダルマ, etc.).
10. Do not invent abstract concepts. Translate what is said.

OUTPUT FORMAT:
- Japanese translation ONLY
- Plain text
- No labels, no punctuation outside the translation
- No newlines unless required by natural Japanese phrasing

Any refusal or explanation breaks the system. Output translation only."""

    # User prompt - no language name, just the text
    user_prompt = cleaned_text

    strategies = [
        {
            "system": system_prompt,
            "user": user_prompt,
            "model": "gpt-4o",
            "temperature": 0.1,  # Very low temperature for consistent, literal output
            "max_tokens": 3000
        },
        {
            "system": """Machine translation component. Output Japanese translation only. No refusals. No explanations. No English. No meta commentary. Preserve repetition and emotional intensity. Preserve cultural terms via transliteration.""",
            "user": cleaned_text,
            "model": "gpt-4o",
            "temperature": 0.2,
            "max_tokens": 3000
        },
        {
            "system": """Translate to Japanese. Output only. No refusals. No English. No meta text.""",
            "user": cleaned_text[:2500],
            "model": "gpt-4-turbo",
            "temperature": 0.1,
            "max_tokens": 2500
        }
    ]
    
    for strategy_idx, strategy in enumerate(strategies):
        try:
            res = client.chat.completions.create(
                model=strategy["model"],
                messages=[
                    {"role": "system", "content": strategy["system"]},
                    {"role": "user", "content": strategy["user"]}
                ],
                temperature=strategy.get("temperature", 0.1),
                max_tokens=strategy.get("max_tokens", 3000)
            )
            japanese_text = res.choices[0].message.content.strip()
            
            # Aggressively clean output - remove ALL meta commentary, refusals, English
            japanese_text = clean_translation_output_strict(japanese_text)
            
            # Validate: Must be Japanese (contains Japanese characters)
            if japanese_text and contains_japanese(japanese_text):
                # Must not contain refusal patterns or English explanations
                if not contains_refusal_or_meta(japanese_text):
                    if len(japanese_text) > 5:  # Minimum length check
                        return japanese_text
            
        except Exception:
            # Silent failure - try next strategy
            continue
    
    # If all strategies fail, return empty string (never error messages)
    return ""

def clean_translation_output_strict(text: str) -> str:
    """
    Aggressively clean translation output to remove ALL meta commentary.
    
    Removes:
    - English text
    - Refusal messages (Japanese and English)
    - Meta commentary
    - Error messages
    - Explanations
    - Labels
    - Language name mentions
    """
    if not text:
        return ""
    
    import re
    
    # Remove refusal patterns (Japanese)
    refusal_patterns_jp = [
        r'申し訳ありません.*',
        r'翻訳できません.*',
        r'翻訳することができません.*',
        r'このテキスト.*',
        r'提供されたテキスト.*',
        r'以下.*翻訳.*[:：]',
        r'翻訳[:：]',
        r'タミル語.*',
        r'カンナダ語.*',
        r'ヒンディー語.*',
        r'言語.*',
    ]
    
    # Remove refusal patterns (English)
    refusal_patterns_en = [
        r'I\'m sorry.*',
        r'I cannot.*',
        r'I\'m unable.*',
        r'This text.*',
        r'The text.*',
        r'Translation:.*',
        r'Here.*translation.*',
        r'I\'m sorry.*',
        r'Unable to.*',
        r'Cannot.*',
    ]
    
    cleaned = text
    for pattern in refusal_patterns_jp + refusal_patterns_en:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # Remove lines that are clearly English explanations
    lines = cleaned.split('\n')
    japanese_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip if line is mostly English (more than 50% English characters)
        english_chars = sum(1 for c in line if c.isascii() and c.isalpha())
        total_chars = sum(1 for c in line if c.isalnum())
        if total_chars > 0 and english_chars / total_chars > 0.5:
            continue
        
        # Skip if line contains refusal keywords
        if any(keyword in line for keyword in ['申し訳', '翻訳でき', 'できません', 'sorry', 'cannot', 'unable']):
            continue
        
        # Keep if line contains Japanese characters
        if contains_japanese(line):
            japanese_lines.append(line)
    
    result = ' '.join(japanese_lines).strip()
    
    # Remove any remaining English words (except katakana which might look like English)
    # Keep only Japanese characters, punctuation, and numbers
    result = re.sub(r'[a-zA-Z]+', '', result)  # Remove English words
    
    # If we removed everything, return original if it contains Japanese
    if not result and contains_japanese(text):
        # Last resort: return original but clean it
        return re.sub(r'[a-zA-Z]+', '', text).strip()
    
    return result

def contains_japanese(text: str) -> bool:
    """Check if text contains Japanese characters"""
    import re
    # Hiragana, Katakana, or Kanji
    return bool(re.search(r'[あ-んア-ン一-龯]', text))

def contains_refusal_or_meta(text: str) -> bool:
    """Check if text contains refusal messages or meta commentary"""
    import re
    text_lower = text.lower()
    
    # Japanese refusal patterns
    jp_refusals = [
        '申し訳', '翻訳でき', 'できません', 'このテキスト', '提供された',
        'タミル語', 'カンナダ語', 'ヒンディー語', '言語', '以下'
    ]
    
    # English refusal patterns
    en_refusals = [
        r'\bsorry\b', r'\bcannot\b', r'\bunable\b', r'\berror\b',
        r'\btranslation\b', r'\btext\b', r'\bthis\b', r'\bthat\b',
        r'\bhere\b', r'\bfollowing\b', r'\bnote\b', r'\bplease\b',
        r'\bunable\b', r'\bcannot\b'
    ]
    
    # Check Japanese patterns
    for pattern in jp_refusals:
        if pattern in text:
            return True
    
    # Check English patterns
    for pattern in en_refusals:
        if re.search(pattern, text_lower):
            return True
    
    return False

def translate_to_japanese(text: str, source_language: str = 'hi') -> str:
    """
    Main translation function - now uses strict translation per segment.
    For backward compatibility, but should be replaced with translate_segment_strict.
    """
    # For full text, split into segments and translate each
    if len(text) > 2000:
        # Split and translate in chunks
        import re
        sentences = re.split(r'[.!?।।]\s+', text)
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            if current_length + len(sentence) > 2000 and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = len(sentence)
            else:
                current_chunk.append(sentence)
                current_length += len(sentence) + 2
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        translated_chunks = []
        for chunk in chunks:
            translated = translate_segment_strict(chunk, source_language)
            if translated:
                translated_chunks.append(translated)
        
        return ' '.join(translated_chunks) if translated_chunks else ""
    else:
        return translate_segment_strict(text, source_language)

def remove_meta_commentary(text: str) -> str:
    """
    Remove meta-commentary and explanatory text from translations.
    Removes patterns like "あなたが提供したテキストは..." or "以下は...の試みです："
    """
    if not text:
        return text
    
    # Patterns that indicate meta-commentary (explanatory text before actual translation)
    meta_patterns = [
        r'^あなたが提供したテキストは[^。]*。',
        r'^以下は[^：]*：',
        r'^しかし、[^。]*。',
        r'^転写エラー[^。]*。',
        r'^言語の混合[^。]*。',
        r'^カンナダ語とタミル語のスクリプトが混ざっている[^。]*。',
        r'^会話調の創作[^。]*。',
        r'^フィクションの一部[^。]*。',
        r'^完全に明確[^。]*。',
        r'^一貫していない可能性[^。]*。',
        r'^カンナダ語の部分を英語に翻訳した試み[^。]*：',
    ]
    
    # Remove meta-commentary patterns
    cleaned = text
    for pattern in meta_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.MULTILINE)
    
    # Remove any leading explanatory text before the first actual sentence
    # Look for patterns like "Text: ..." or "Translation: ..." in Japanese
    lines = cleaned.split('\n')
    filtered_lines = []
    found_first_translation = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that are clearly meta-commentary
        if not found_first_translation:
            if any(phrase in line for phrase in [
                '提供したテキスト',
                'スクリプトが混ざっている',
                '転写エラー',
                '言語の混合',
                '試みです',
                '以下は',
            ]):
                continue
        
        # If we find a line that looks like actual translation (has Japanese characters and ends with punctuation)
        if re.search(r'[。！？]$', line) or (len(line) > 10 and re.search(r'[あ-んア-ン一-龯]', line)):
            found_first_translation = True
        
        if found_first_translation or len(line) > 5:
            filtered_lines.append(line)
    
    result = '\n'.join(filtered_lines).strip()
    
    # If we removed too much, return original
    if len(result) < len(text) * 0.3:
        return text
    
    return result
