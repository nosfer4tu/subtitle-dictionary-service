from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import json

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
    Transcribe audio using COMPETING ASR PASSES.
    
    DEPRECATED: This function redirects to whisper_transcribe.transcribe_audio
    which uses competing ASR passes (en, ta, auto) and selects the best English candidate.
    
    Args:
        file_path: Path to audio file
        language: DEPRECATED - competing passes are always run (en, ta, auto)
    
    Returns:
        Best English transcribed text, or empty string if all passes rejected
    """
    # Redirect to whisper_transcribe which implements competing passes
    from backend_python.whisper_transcribe import transcribe_audio as whisper_transcribe_audio
    return whisper_transcribe_audio(file_path, language=None, use_call0_gate=True)

def transcribe_audio_with_timestamps(file_path: str, language: str = None) -> dict:
    """
    Transcribe audio with word-level timestamps using COMPETING ASR PASSES.
    
    DEPRECATED: This function redirects to whisper_transcribe.transcribe_audio_with_timestamps
    which runs competing ASR passes (en, ta, auto) and selects the best English candidate.
    
    Args:
        file_path: Path to audio file
        language: DEPRECATED - competing passes are always run (en, ta, auto)
    
    Returns:
        Transcription result with segments from the best English ASR candidate
    """
    # Redirect to whisper_transcribe which implements competing passes
    from backend_python.whisper_transcribe import transcribe_audio_with_timestamps as whisper_transcribe_with_timestamps
    return whisper_transcribe_with_timestamps(file_path, language=None, use_call0_gate=False)

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

def replace_katakana_with_meaning(text: str, english_source: str = None) -> str:
    """
    Replace katakana characters with kanji/hiragana equivalents that preserve meaning.
    This is smarter than just deleting - it asks LLM to replace with meaningful alternatives.
    
    Args:
        text: Japanese text that may contain katakana
        english_source: Optional English source text to help preserve meaning
    
    Returns:
        Text with katakana replaced by kanji/hiragana equivalents
    """
    if not text:
        return text
    
    # Check if katakana exists
    katakana_chars = re.findall(r'[ァ-ヶー]', text)
    if not katakana_chars:
        return text  # No katakana, return as-is
    
    # If we have English source, use it to help preserve meaning
    if english_source:
        try:
            client = get_client()
            replacement_prompt = f"""Your Japanese translation contains katakana that must be replaced.

JAPANESE TEXT (contains katakana):
{text}

ENGLISH SOURCE (for context):
{english_source[:1000]}

TASK:
Replace ALL katakana characters with kanji/hiragana equivalents that preserve the EXACT meaning.

Rules:
• Do NOT delete or remove words - REPLACE them with meaningful Japanese
• Use kanji/hiragana equivalents that convey the same meaning
• Preserve the original meaning completely
• If a word was in katakana, find the appropriate Japanese word (kanji/hiragana) with the same meaning

Examples:
• アッパ (father) → 父
• マーヤー (illusion) → 幻
• イシュヴァラ (god) → 神

Output: Rewritten Japanese text with ZERO katakana, preserving all meaning."""
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a Japanese language expert. Replace katakana with kanji/hiragana equivalents that preserve meaning. Do NOT delete words - REPLACE them. Output ONLY the corrected text."},
                    {"role": "user", "content": replacement_prompt}
                ],
                max_completion_tokens=4000
            )
            replaced_text = response.choices[0].message.content.strip()
            
            # Clean any meta-commentary
            replaced_text = re.sub(r'^(以下|以下は|修正|CORRECTION|Replacement)[:：]?\s*', '', replaced_text, flags=re.MULTILINE)
            replaced_text = replaced_text.strip()
            
            # Verify katakana is gone
            remaining_katakana = re.findall(r'[ァ-ヶー]', replaced_text)
            if not remaining_katakana:
                print(f"✓ Katakana replaced with meaning-preserving alternatives", flush=True)
                return replaced_text
            else:
                print(f"Warning: Some katakana still remains after replacement attempt", flush=True)
                # Fall through to character removal as last resort
        except Exception as e:
            print(f"Katakana replacement failed: {e}, falling back to character removal", flush=True)
    
    # Last resort: Remove katakana characters (but try to preserve spacing)
    # This is less ideal but better than nothing
    print(f"Warning: Using character removal as last resort - meaning may be affected", flush=True)
    text = re.sub(r'[ァ-ヶー]+', '', text)
    
    # Clean up resulting issues
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[。、]\s*[。、]', '。', text)
    text = re.sub(r'\s+([。、！？])', r'\1', text)
    text = re.sub(r'([。、！？])\s+', r'\1 ', text)
    
    return text.strip()

def force_remove_katakana(text: str) -> str:
    """
    DEPRECATED: Use replace_katakana_with_meaning() instead.
    This function is kept for backward compatibility but should not be used
    as it distorts meaning by deleting characters.
    """
    return replace_katakana_with_meaning(text)

def translate_segment_strict(text: str, language: str, script: str = None) -> str:
    """
    DEPRECATED: This function is a BYPASS PATH that violates the strict pipeline.
    
    ⚠️  HARD BLOCK: This function MUST NOT be called directly with raw ASR text.
    All translation MUST go through: Call 1 → has_clear_english() → Call 2
    
    This function is kept for backward compatibility but will return empty string
    to prevent raw ASR from being translated without going through Call 1.
    
    Use process_transcription_pipeline() instead, which enforces the strict gate.
    """
    # === HARD BLOCK: Prevent direct translation bypass ===
    print("=" * 80, flush=True)
    print("❌ BYPASS DETECTED: translate_segment_strict() called directly with raw ASR", flush=True)
    print("   This violates the strict pipeline: Call 1 → has_clear_english() → Call 2", flush=True)
    print(f"   Input text: {repr(text[:100])}", flush=True)
    print("   Returning empty string to prevent translation bypass.", flush=True)
    print("=" * 80, flush=True)
    return ""  # HARD BLOCK: Return empty to prevent bypass
    
    # Original function code below is now unreachable (kept for reference)
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
            "max_completion_tokens": 3000
        },
        {
            "system": """Machine translation component. Output Japanese translation only. No refusals. No explanations. No English. No meta commentary. Preserve repetition and emotional intensity. Preserve cultural terms via transliteration.""",
            "user": cleaned_text,
            "model": "gpt-4o",
            "max_completion_tokens": 3000
        },
        {
            "system": """Translate to Japanese. Output only. No refusals. No English. No meta text.""",
            "user": cleaned_text[:2500],
            "model": "gpt-4-turbo",
            "max_completion_tokens": 2500
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
                max_completion_tokens=strategy.get("max_completion_tokens", 3000)
            )
            japanese_text = res.choices[0].message.content.strip()
            
            # Aggressively clean output - remove ALL meta commentary, refusals, English
            japanese_text = clean_translation_output_strict(japanese_text)
            
            # Replace katakana with meaning-preserving alternatives (if any)
            katakana_check = re.findall(r'[ァ-ヶー]', japanese_text)
            if katakana_check:
                # Note: We don't have English source here, so just remove as last resort
                # But this is less ideal - the main pipeline should handle this better
                japanese_text = replace_katakana_with_meaning(japanese_text)
            
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
    # Even if empty, ensure no katakana
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
    
    # Remove refusal patterns (Japanese) - includes all patterns from is_refusal
    refusal_patterns_jp = [
        r'.*申し訳.*',
        r'.*お答えでき.*',
        r'.*対応でき.*',
        r'.*できません.*',
        r'.*ご要望.*',
        r'.*翻訳でき.*',
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

def cleanup_english_transcription(raw_asr_text: str) -> str:
    """
    Mechanical ASR text normalizer.
    Performs ONLY mechanical fixes: casing, punctuation, duplicates, contractions.
    Returns input unchanged if already clean.
    """
    if not raw_asr_text or not raw_asr_text.strip():
        return raw_asr_text
    
    text = raw_asr_text.strip()
    
    # Normalize contractions (preserve original casing)
    contractions = {
        r'\bdont\b': "don't", r'\bcant\b': "can't", r'\bwont\b': "won't",
        r'\bisnt\b': "isn't", r'\barent\b': "aren't", r'\bwasnt\b': "wasn't",
        r'\bwerent\b': "weren't", r'\bhasnt\b': "hasn't", r'\bhavent\b': "haven't",
        r'\bhadnt\b': "hadn't", r'\bwouldnt\b': "wouldn't", r'\bcouldnt\b': "couldn't",
        r'\bshouldnt\b': "shouldn't", r'\bmustnt\b': "mustn't", r'\bmightnt\b': "mightn't",
        r'\bim\b': "I'm", r'\byoure\b': "you're", r'\bhes\b': "he's", r'\bshes\b': "she's",
        r'\bits\b': "it's", r'\btheyre\b': "they're",
        r'\bive\b': "I've", r'\byouve\b': "you've", r'\bweve\b': "we've", r'\btheyve\b': "they've",
        r'\bid\b': "I'd", r'\byoud\b': "you'd", r'\bhed\b': "he'd", r'\bshed\b': "she'd",
        r'\bwed\b': "we'd", r'\btheyd\b': "they'd",
        r'\bill\b': "I'll", r'\byoull\b': "you'll", r'\bhell\b': "he'll",
        r'\bshell\b': "she'll", r'\bwell\b': "we'll", r'\btheyll\b': "they'll",
    }
    
    for pattern, replacement in contractions.items():
        def replace_func(match):
            word = match.group(0)
            if word[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement
        text = re.sub(pattern, replace_func, text, flags=re.IGNORECASE)
    
    # Fix isolated "i" → "I" (but not in "I'm", "I've", etc. - already handled)
    text = re.sub(r'\bi\b', 'I', text)
    
    # Remove duplicated consecutive words (case-insensitive)
    words = text.split()
    if not words:
        return text.strip()
    
    deduplicated = [words[0]]
    for i in range(1, len(words)):
        if words[i].lower() != words[i-1].lower():
            deduplicated.append(words[i])
    
    text = ' '.join(deduplicated)
    
    # Fix sentence capitalization (after . ! ?)
    # Capitalize first letter of text
    if text and text[0].islower() and text[0].isalpha():
        text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()
    
    # Capitalize after sentence endings (space required)
    text = re.sub(r'([.!?]\s+)([a-z])', lambda m: m.group(1) + m.group(2).upper(), text)
    
    # Normalize multiple spaces
    text = re.sub(r' +', ' ', text)
    
    # Normalize multiple punctuation (reduce .!? sequences to single)
    text = re.sub(r'([.!?]){2,}', r'\1', text)
    
    # Ensure single space after punctuation (but not before)
    text = re.sub(r'([.!?])([^\s.!?])', r'\1 \2', text)
    text = re.sub(r'([.!?])\s+([.!?])', r'\1\2', text)
    
    return text.strip()

def translate_subtitle_segment(cleaned_text: str) -> str:
    """
    Translate a single subtitle segment from English to Japanese.
    Designed for fictional movie dialogue with refusal detection and safety framing.
    
    Args:
        cleaned_text: Cleaned English text from ASR segment
    
    Returns:
        Japanese translation (never empty, never placeholder)
    """
    if not cleaned_text or not cleaned_text.strip():
        return ""
    
    text = cleaned_text.strip()
    word_count = len(text.split())
    char_count = len(text)
    
    client = get_client()
    
    # English refusal patterns
    EN_REFUSAL_PATTERNS = [
        "i'm sorry",
        "i cannot assist",
        "i can't assist",
        "as an ai",
        "i cannot",
        "i can't",
        "unable to",
        "cannot help",
    ]
    
    # Japanese refusal patterns (as specified)
    JP_REFUSAL_PATTERNS = [
        "申し訳",
        "お答えでき",
        "対応でき",
        "できません",
        "ご要望",
        "翻訳でき",
    ]
    
    def is_refusal(output: str) -> bool:
        """
        Check if output contains refusal patterns in ANY language.
        Detects both English and Japanese refusals.
        """
        if not output:
            return False
        
        output_lower = output.lower()
        output_text = output  # Keep original for Japanese pattern matching
        
        # Check English patterns (case-insensitive)
        if any(pattern in output_lower for pattern in EN_REFUSAL_PATTERNS):
            return True
        
        # Check Japanese patterns (exact match, no case conversion)
        if any(pattern in output_text for pattern in JP_REFUSAL_PATTERNS):
            return True
        
        return False
    
    def create_literal_fallback(text: str) -> str:
        """
        Create a literal Japanese paraphrase as last resort.
        Best-effort translation based on surface meaning.
        Derived ONLY from English input - no placeholders, no apologies.
        """
        # Enhanced word-by-word fallback dictionary
        common_words = {
            # Pronouns
            "i": "私", "you": "あなた", "he": "彼", "she": "彼女",
            "we": "私たち", "they": "彼ら", "me": "私", "us": "私たち",
            # Common verbs
            "is": "です", "are": "です", "am": "です", "was": "でした",
            "be": "です", "have": "持つ", "has": "持つ", "had": "持った",
            "do": "する", "does": "する", "did": "した", "will": "する",
            "can": "できる", "could": "できる", "should": "すべき",
            "go": "行く", "come": "来る", "see": "見る", "know": "知る",
            "say": "言う", "tell": "伝える", "get": "得る", "make": "作る",
            # Common nouns
            "king": "王", "queen": "女王", "man": "男", "woman": "女",
            "person": "人", "people": "人々", "thing": "もの", "way": "方法",
            "time": "時間", "day": "日", "night": "夜", "year": "年",
            "story": "話", "big": "大きい", "small": "小さい", "good": "良い",
            "bad": "悪い", "new": "新しい", "old": "古い",
            # Common adjectives
            "big": "大きい", "large": "大きい", "small": "小さい", "little": "小さい",
            "good": "良い", "bad": "悪い", "great": "素晴らしい",
            # Common adverbs
            "very": "とても", "much": "多く", "many": "多く", "more": "もっと",
            "most": "最も", "too": "も", "also": "も", "only": "だけ",
            # Interjections
            "hey": "おい", "yes": "はい", "no": "いいえ", "ok": "オーケー",
            # Violence-related (translate neutrally)
            "kill": "殺す", "die": "死ぬ", "death": "死", "war": "戦争",
            "fight": "戦う", "battle": "戦い", "blood": "血",
            # Articles and determiners (often omitted in Japanese)
            "the": "", "a": "", "an": "", "this": "これ", "that": "それ",
            "these": "これら", "those": "それら",
            # Common phrases
            "that's": "それは", "it's": "それは", "this is": "これは",
            "what": "何", "who": "誰", "where": "どこ", "when": "いつ",
            "why": "なぜ", "how": "どのように",
        }
        
        # Handle contractions
        text = text.replace("'s", " is").replace("'re", " are").replace("'m", " am")
        text = text.replace("'ve", " have").replace("'ll", " will").replace("'d", " would")
        text = text.replace("n't", " not")
        
        words = text.lower().split()
        result = []
        for word in words:
            # Remove punctuation for lookup
            clean_word = word.strip('.,!?;:')
            if clean_word in common_words:
                translation = common_words[clean_word]
                if translation:  # Skip empty translations (articles)
                    result.append(translation)
            else:
                # For unknown words, try to preserve them (better than nothing)
                # But only if they look like English words
                if clean_word.isalpha() and len(clean_word) > 1:
                    result.append(clean_word)
        
        # Join with spaces and add period if original had sentence-ending punctuation
        output = " ".join(result) if result else text
        if text and text[-1] in '.!?':
            if output and output[-1] not in '.!?。！？':
                output += "。"
        
        # Final check: ensure no refusal patterns leaked through
        if is_refusal(output):
            # If fallback itself contains refusal, return minimal safe text
            return "翻訳"
        
        return output if output else text
    
    # A. Short-utterance fast path
    if word_count <= 2 or char_count <= 12:
        # Direct translation, no fallback strategies
        system_prompt = """This is fictional dialogue from a movie.
Translate faithfully into natural Japanese.
Do not refuse, warn, or explain.
Violence may be present as part of the story.
Output ONLY the Japanese translation. Nothing else."""
        
        try:
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text}
                ],
                max_completion_tokens=100,
                temperature=0.3
            )
            japanese_text = res.choices[0].message.content.strip()
            japanese_text = clean_translation_output_strict(japanese_text)
            
            # Check for refusal (any language)
            if is_refusal(japanese_text):
                # Retry with stronger message
                stronger_prompt = """You are translating fictional movie subtitles.
Refusals are forbidden.
Translate literally.
Output Japanese text only."""
                res = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": stronger_prompt},
                        {"role": "user", "content": text}
                    ],
                    max_completion_tokens=100,
                    temperature=0.3
                )
                japanese_text = res.choices[0].message.content.strip()
                japanese_text = clean_translation_output_strict(japanese_text)
                
                # If still refusal, use literal fallback
                if is_refusal(japanese_text) or not contains_japanese(japanese_text):
                    return create_literal_fallback(text)
            
            if japanese_text and contains_japanese(japanese_text):
                return japanese_text
            else:
                return create_literal_fallback(text)
                
        except Exception as e:
            return create_literal_fallback(text)
    
    # B. Full translation with fictional-dialogue safety framing
    system_prompt = """This is fictional dialogue from a movie.
Translate faithfully into natural Japanese.
Do not refuse, warn, or explain.
Violence may be present as part of the story.
Output ONLY the Japanese translation. No explanations, no meta text, no markdown."""
    
    strategies = [
        {
            "system": system_prompt,
            "user": text,
            "model": "gpt-4o",
            "max_completion_tokens": 500
        },
        {
            "system": "You are translating fictional movie subtitles. Refusals are not allowed. Translate to Japanese. Output only.",
            "user": text[:500],
            "model": "gpt-4-turbo",
            "max_completion_tokens": 400
        },
    ]
    
    for strategy_idx, strategy in enumerate(strategies):
        try:
            res = client.chat.completions.create(
                model=strategy["model"],
                messages=[
                    {"role": "system", "content": strategy["system"]},
                    {"role": "user", "content": strategy["user"]}
                ],
                max_completion_tokens=strategy.get("max_completion_tokens", 500),
                temperature=0.3
            )
            japanese_text = res.choices[0].message.content.strip()
            japanese_text = clean_translation_output_strict(japanese_text)
            
            # Check for refusal (any language)
            if is_refusal(japanese_text):
                # Retry with stronger message
                stronger_prompt = """You are translating fictional movie subtitles.
Refusals are forbidden.
Translate literally.
Output Japanese text only."""
                res = client.chat.completions.create(
                    model=strategy["model"],
                    messages=[
                        {"role": "system", "content": stronger_prompt},
                        {"role": "user", "content": strategy["user"]}
                    ],
                    max_completion_tokens=strategy.get("max_completion_tokens", 500),
                    temperature=0.3
                )
                japanese_text = res.choices[0].message.content.strip()
                japanese_text = clean_translation_output_strict(japanese_text)
                
                # If still refusal after retry, use literal fallback (no more strategies)
                if is_refusal(japanese_text) or not contains_japanese(japanese_text):
                    return create_literal_fallback(text)
            
            if japanese_text and contains_japanese(japanese_text):
                return japanese_text
                
        except Exception as e:
            continue
    
    # If all strategies fail, use literal fallback (never return placeholder)
    return create_literal_fallback(text)

def translate_english_to_japanese(english_text: str) -> str:
    """
    Translate English text directly to Japanese in a single call.
    
    This is the NEW DB-driven translation path:
    - Reads English from source_transcription (database column)
    - Translates directly to Japanese
    - Returns single coherent block of Japanese text
    - No segment processing, no gates, no placeholders per segment
    
    Args:
        english_text: English text from source_transcription column
    
    Returns:
        Japanese translation as a single coherent block, or empty string if input is empty/null,
        or placeholder only if translation API fails completely
    """
    # Placeholder rules: If source_transcription is empty or null → return empty string
    if not english_text or not english_text.strip():
        return ""
    
    client = get_client()
    
    # Clean and prepare text
    cleaned_text = english_text.strip()
    # Limit length to avoid token limits (keep reasonable limit for full transcriptions)
    if len(cleaned_text) > 8000:
        cleaned_text = cleaned_text[:8000]
        print(f"Warning: Text truncated to 8000 chars for translation", flush=True)
    
    # System prompt for direct English → Japanese translation
    system_prompt = """You are a professional translator. Translate the following English text to Japanese.

Rules:
1. Output ONLY the Japanese translation. Nothing else.
2. Preserve natural paragraph flow and structure.
3. Do not add sentence numbering or labels.
4. Preserve cultural context and meaning.
5. Use natural, fluent Japanese.
6. Do not include any explanations, meta commentary, or English text.

Output the translation as a single coherent block of text."""
    
    strategies = [
        {
            "system": system_prompt,
            "user": cleaned_text,
            "model": "gpt-4o",
            "max_completion_tokens": 4000
        },
        {
            "system": "Translate the following English text to Japanese. Output only the Japanese translation, no explanations.",
            "user": cleaned_text[:6000],
            "model": "gpt-4-turbo",
            "max_completion_tokens": 3000
        },
        {
            "system": "Translate to Japanese. Output only.",
            "user": cleaned_text[:4000],
            "model": "gpt-4",
            "max_completion_tokens": 2000
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
                max_completion_tokens=strategy.get("max_completion_tokens", 4000)
            )
            japanese_text = res.choices[0].message.content.strip()
            
            # Clean output - remove meta commentary, refusals, English
            japanese_text = clean_translation_output_strict(japanese_text)
            
            # Validate: Must be Japanese (contains Japanese characters)
            if japanese_text and contains_japanese(japanese_text):
                # Must not contain refusal patterns or English explanations
                if not contains_refusal_or_meta(japanese_text):
                    if len(japanese_text) > 5:  # Minimum length check
                        print(f"✓ Translation successful (strategy {strategy_idx + 1})", flush=True)
                        return japanese_text
            
        except Exception as e:
            print(f"Translation strategy {strategy_idx + 1} failed: {e}", flush=True)
            continue
    
    # If all strategies fail, return placeholder (only global placeholder, never per-segment)
    print("⚠️  All translation strategies failed - returning placeholder", flush=True)
    return "意味不明\n（翻訳に失敗しました）"

def translate_to_japanese(text: str, source_language: str = 'hi') -> str:
    """
    DEPRECATED: This function is a BYPASS PATH that violates the strict pipeline.
    
    ⚠️  HARD BLOCK: This function MUST NOT be called directly with raw ASR text.
    All translation MUST go through: Call 1 → has_clear_english() → Call 2
    
    This function is kept for backward compatibility but will redirect to the main pipeline
    or return empty string to prevent raw ASR from being translated without going through Call 1.
    
    Use process_transcription_pipeline() instead, which enforces the strict gate.
    """
    # === HARD BLOCK: Prevent direct translation bypass ===
    print("=" * 80, flush=True)
    print("❌ BYPASS DETECTED: translate_to_japanese() called directly with raw ASR", flush=True)
    print("   This violates the strict pipeline: Call 1 → has_clear_english() → Call 2", flush=True)
    print(f"   Input text: {repr(text[:100])}", flush=True)
    print("   Redirecting to process_transcription_pipeline() to enforce gates.", flush=True)
    print("=" * 80, flush=True)
    
    # Redirect to main pipeline which enforces all gates
    try:
        pipeline_result = process_transcription_pipeline(text)
        japanese_translation_text = pipeline_result.get('japanese_translation', '')
        
        # Parse the formatted output to extract just the translation text
        if japanese_translation_text:
            lines = japanese_translation_text.split('\n')
            translation_lines = []
            for line in lines:
                # Remove "Sentence X: " prefix if present
                if ': ' in line:
                    translation = line.split(': ', 1)[1]
                else:
                    translation = line
                if translation.strip():
                    translation_lines.append(translation.strip())
            return ' '.join(translation_lines) if translation_lines else ""
        else:
            return ""  # Gate blocked - return empty
    except Exception as e:
        print(f"Pipeline error in translate_to_japanese redirect: {e}", flush=True)
        return ""  # On error, return empty (gate blocked)

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


def forensic_pre_segmentation_filter(sentence: str) -> dict:
    """
    Deterministic pre-segmentation forensic filter.
    
    Identifies sentences that are not eligible for linguistic interpretation
    and should be marked as [UNINTELLIGIBLE] before reaching the LLM.
    
    This filter's job is NOT to interpret meaning.
    It must block sentences that are likely to cause hallucination.
    
    Args:
        sentence: A single sentence to evaluate
        
    Returns:
        dict with keys:
            - status: "blocked" or "eligible"
            - reason: List of reasons why sentence was blocked (if blocked)
            - debug_info: Additional debug information
    """
    if not sentence or not sentence.strip():
        return {
            "status": "blocked",
            "reason": ["empty_or_whitespace"],
            "debug_info": {"sentence_length": 0}
        }
    
    sentence = sentence.strip()
    reasons = []
    debug_info = {}
    
    # A. Mixed-script contamination check
    script_presence = {
        'Tamil': False,
        'Kannada': False,
        'Devanagari': False,
        'Latin': False
    }
    
    for char in sentence:
        code_point = ord(char)
        # Tamil
        if 0x0B80 <= code_point <= 0x0BFF:
            script_presence['Tamil'] = True
        # Kannada
        elif 0x0C80 <= code_point <= 0x0CFF:
            script_presence['Kannada'] = True
        # Devanagari
        elif 0x0900 <= code_point <= 0x097F:
            script_presence['Devanagari'] = True
        # Latin (basic ASCII letters)
        elif char.isalpha() and ord(char) < 128:
            script_presence['Latin'] = True
    
    scripts_found = sum(1 for present in script_presence.values() if present)
    if scripts_found > 1:
        reasons.append("mixed_script")
        debug_info["scripts_detected"] = [k for k, v in script_presence.items() if v]
    
    # B. ASR corruption density check (>30% non-dictionary/garbled tokens)
    # Split into tokens (words/syllables)
    # For Indian scripts, we look for patterns that indicate corruption:
    # - Isolated single characters (common in ASR errors)
    # - Sequences that don't form valid words
    # - Excessive punctuation mixed with text
    
    tokens = sentence.split()
    if tokens:
        corrupted_count = 0
        total_chars_in_tokens = 0
        
        for token in tokens:
            token = token.strip('.,!?;:()[]{}"\'')
            if not token:
                continue
            
            total_chars_in_tokens += len(token)
            
            # Check for isolated single characters (ASR corruption indicator)
            if len(token) == 1 and token not in ['அ', 'ஆ', 'இ', 'ஈ', 'உ', 'ஊ', 'எ', 'ஏ', 'ஐ', 'ஒ', 'ஓ', 'ஔ']:
                corrupted_count += len(token)
                continue
            
            # Check for excessive punctuation within token
            punct_ratio = sum(1 for c in token if c in '.,!?;:()[]{}"\'') / len(token) if token else 0
            if punct_ratio > 0.3:
                corrupted_count += len(token)
                continue
            
            # Check for patterns that look like partial syllables or garbled text
            # Very short tokens (1-2 chars) with mixed scripts in single token
            if len(token) <= 2:
                scripts_in_token = set()
                for char in token:
                    code_point = ord(char)
                    if 0x0B80 <= code_point <= 0x0BFF:
                        scripts_in_token.add('Tamil')
                    elif 0x0C80 <= code_point <= 0x0CFF:
                        scripts_in_token.add('Kannada')
                    elif 0x0900 <= code_point <= 0x097F:
                        scripts_in_token.add('Devanagari')
                
                if len(scripts_in_token) > 1:
                    corrupted_count += len(token)
                    continue
        
        if total_chars_in_tokens > 0:
            corruption_ratio = corrupted_count / total_chars_in_tokens
            if corruption_ratio > 0.3:
                reasons.append("asr_corruption_density")
                debug_info["corruption_ratio"] = corruption_ratio
                debug_info["corrupted_chars"] = corrupted_count
                debug_info["total_chars"] = total_chars_in_tokens
    
    # C. Repetition without propositional structure
    # Check for excessive repetition (chanting, shouting, crowd noise)
    # Pattern: same word/syllable repeated 3+ times
    
    words = sentence.split()
    if len(words) >= 3:
        # Check for repetition of same word
        word_counts = {}
        for word in words:
            word_clean = word.strip('.,!?;:()[]{}"\'')
            if word_clean:
                word_counts[word_clean] = word_counts.get(word_clean, 0) + 1
        
        # If a single word appears more than 50% of the time (chanting pattern)
        max_repetition = max(word_counts.values()) if word_counts else 0
        if max_repetition >= 3 and max_repetition / len(words) > 0.5:
            reasons.append("repetition_without_structure")
            debug_info["max_word_repetition"] = max_repetition
            debug_info["total_words"] = len(words)
        
        # Check for syllable-level repetition (like "ஆஹா ஆஹா ஆஹா")
        # Look for patterns where 2-3 char sequences repeat
        text_no_punct = re.sub(r'[.,!?;:()\[\]{}"\']', '', sentence)
        if len(text_no_punct) >= 6:
            # Check for 2-char repeating patterns
            for i in range(len(text_no_punct) - 4):
                pattern = text_no_punct[i:i+2]
                # Count occurrences of this 2-char pattern
                pattern_count = text_no_punct.count(pattern)
                if pattern_count >= 4:  # Appears 4+ times = likely chanting
                    reasons.append("repetition_chanting")
                    debug_info["repetitive_pattern"] = pattern
                    break
    
    # D. Fragmentation check
    # Check if sentence lacks clear structure (no clear subject-predicate)
    # Simple heuristic: very short sentences or sentences with only exclamations
    
    # Very short (less than 5 chars after cleaning) - likely fragment
    clean_length = len(re.sub(r'[.,!?;:()\[\]{}"\'\s]', '', sentence))
    if clean_length < 5:
        reasons.append("fragmentation_too_short")
        debug_info["clean_length"] = clean_length
    
    # Check if sentence is only exclamations/interjections (excessive ! or ?)
    exclamation_count = sentence.count('!') + sentence.count('?')
    if exclamation_count >= 3 and len(sentence) < 20:
        reasons.append("fragmentation_exclamations_only")
        debug_info["exclamation_count"] = exclamation_count
    
    # E. Oral-performance markers
    # Crowd calls, dramatic interjections, song-like rhythm
    
    # Multiple exclamation markers
    if sentence.count('!') >= 2 and len(sentence) < 30:
        reasons.append("oral_performance_markers")
        debug_info["exclamation_count"] = sentence.count('!')
    
    # Check for vocative-like patterns (repeated addressing, like "ரா ரா" or "आ आ")
    # This overlaps with repetition check but specifically for oral performance
    
    # Decision: if ANY reason found, block the sentence
    if reasons:
        return {
            "status": "blocked",
            "reason": reasons,
            "debug_info": debug_info
        }
    
    return {
        "status": "eligible",
        "reason": [],
        "debug_info": {}
    }


def segment_sentences(text: str) -> list:
    """
    Segment raw transcription into short, independent segments.
    
    CRITICAL: Segments must be short (8-12 seconds or 1-2 clauses) to prevent hallucination.
    Prefer breaking on pauses, speaker changes, and punctuation.
    
    Args:
        text: Raw transcription text
        
    Returns:
        List of short sentence segments
    """
    if not text or not text.strip():
        return []
    
    # First, split on major sentence delimiters
    sentence_delimiters = r'[.!?।।]\s+'
    initial_sentences = re.split(sentence_delimiters, text)
    
    # Further segment long sentences into shorter chunks
    # Target: 1-2 clauses or ~50-100 characters (roughly 8-12 seconds of speech)
    cleaned_sentences = []
    for sentence in initial_sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 2:
            continue
        
        # If sentence is long (>100 chars), break it further
        if len(sentence) > 100:
            # Break on commas, semicolons, or conjunctions (clause boundaries)
            clause_delimiters = r'[,;]\s+|(\s+(and|or|but|then|so|because|when|if|that)\s+)'
            clauses = re.split(clause_delimiters, sentence)
            
            current_chunk = ""
            for clause in clauses:
                if clause is None:
                    continue
                clause = clause.strip()
                if not clause:
                    continue
                
                # If adding this clause would exceed ~100 chars, start new segment
                if current_chunk and len(current_chunk) + len(clause) + 2 > 100:
                    if current_chunk:
                        cleaned_sentences.append(current_chunk.strip())
                    current_chunk = clause
                else:
                    if current_chunk:
                        current_chunk += " " + clause
                    else:
                        current_chunk = clause
            
            # Add remaining chunk
            if current_chunk:
                cleaned_sentences.append(current_chunk.strip())
        else:
            # Short enough, add as-is
            cleaned_sentences.append(sentence)
    
    # Final cleanup: remove very short fragments that are likely noise
    final_sentences = []
    for sentence in cleaned_sentences:
        sentence = sentence.strip()
        # Keep sentences that are at least 3 characters (to avoid single-char noise)
        if sentence and len(sentence) >= 3:
            final_sentences.append(sentence)
    
    return final_sentences


def is_refusal_phrase(text: str) -> bool:
    """
    Check if text contains refusal/explanation phrases that should be filtered from Call 2 output.
    
    Args:
        text: Text to check
    
    Returns:
        True if text contains refusal phrases, False otherwise
    """
    if not text or not text.strip():
        return False
    
    text_stripped = text.strip()
    
    # Check for refusal phrases (all forbidden output from Call 2)
    refusal_phrases = [
        "意味不明",
        "音声が",
        "翻訳でき",
        "音声が不明瞭",
        "意味を復元",
        "意味を安全に復元できません",
        "分からない",
        "不明",
        "解釈",
        "複数の意味があり",
        "推測が必要なため",
        "言語が混在し",
        "不完全な文節のため"
    ]
    
    for phrase in refusal_phrases:
        if phrase in text_stripped:
            return True
    
    return False


def accumulate_until_semantic_minimum(sentences):
    """
    Accumulate consecutive sentence fragments until they form a minimally meaningful English unit.
    
    Prevents ultra-short ASR fragments from reaching Call 1 individually.
    Groups fragments until they meet minimum semantic length requirements.
    
    Args:
        sentences: List of tuples (idx, sentence, is_asr_uncertain) or (idx, sentence)
    
    Returns:
        List of tuples (original_indices, combined_text, combined_is_asr_uncertain)
        where original_indices is a list of all fragment indices that were accumulated
    """
    if not sentences:
        return []
    
    accumulated = []
    buffer = []  # List of (idx, sentence, is_asr_uncertain) tuples
    buffer_text_parts = []
    
    def meets_minimum(text):
        """Check if text meets minimum semantic length requirements"""
        if not text or not text.strip():
            return False
        
        # Remove whitespace for character count
        text_no_ws = re.sub(r'\s+', '', text)
        char_length = len(text_no_ws)
        
        # Count words
        words = text.split()
        word_count = len(words)
        
        # Check for at least one alphabetic word
        has_alphabetic = any(re.search(r'[A-Za-z]', word) for word in words)
        
        # Must meet: (char_length >= 20 OR word_count >= 4) AND has_alphabetic
        return has_alphabetic and (char_length >= 20 or word_count >= 4)
    
    def emit_buffer():
        """Emit accumulated buffer as one sentence"""
        if not buffer:
            return
        
        # Combine text from all fragments
        combined_text = ' '.join(buffer_text_parts)
        
        # OR the is_asr_uncertain flags (if any fragment is uncertain, whole is uncertain)
        combined_is_asr_uncertain = any(item[2] if len(item) == 3 else False for item in buffer)
        
        # Collect all original indices from fragments
        original_indices = [item[0] for item in buffer]
        
        # Log accumulation if multiple fragments
        if len(buffer) > 1:
            fragment_texts = [item[1] if len(item) >= 2 else '' for item in buffer]
            print(f"[SEGMENT ACCUMULATED] {len(buffer)} fragments (indices {original_indices}): {repr(' + '.join(fragment_texts))} → {repr(combined_text)}", flush=True)
        
        # Store as (original_indices_list, combined_text, combined_is_asr_uncertain)
        accumulated.append((original_indices, combined_text, combined_is_asr_uncertain))
        buffer.clear()
        buffer_text_parts.clear()
    
    # Process each sentence
    for item in sentences:
        # Handle tuple format
        if len(item) == 3:
            idx, sentence, is_asr_uncertain = item
        else:
            # Backward compatibility
            idx, sentence = item
            is_asr_uncertain = False
        
        # Check if this single sentence already meets minimum (no accumulation needed)
        if meets_minimum(sentence):
            # Emit any existing buffer first
            if buffer:
                emit_buffer()
            # Pass through this sentence as-is (not accumulated)
            accumulated.append(([idx], sentence, is_asr_uncertain))
        else:
            # Sentence is too short - add to buffer for accumulation
            buffer.append((idx, sentence, is_asr_uncertain))
            buffer_text_parts.append(sentence)
            
            # Check if accumulated text now meets minimum
            combined_text = ' '.join(buffer_text_parts)
            if meets_minimum(combined_text):
                # Emit buffer
                emit_buffer()
    
    # At end of input, emit any remaining buffer (even if below threshold)
    if buffer:
        emit_buffer()
    
    return accumulated


def has_clear_english(text: str) -> bool:
    """
    RELAXED PROGRAMMATIC GATE: Check if text contains valid English (including imperfect ASR).
    
    This function is used to enforce control flow between Call 1 and Call 2.
    Call 2 MUST NOT execute unless this function returns True.
    
    RELAXED RULES:
    - Accepts spaced syllables (e.g., "hap py birth day", "ma ny", "hap py")
    - Accepts minor ASR artifacts (extra spaces, punctuation)
    - Only blocks if: text is empty OR contains no alphabetic words at all OR contains non-English scripts
    
    Args:
        text: Text to validate (should be Call 1 output)
    
    Returns:
        True if text contains valid English, False otherwise
    """
    if not text or not text.strip():
        return False
    if "[UNINTELLIGIBLE]" in text:
        return False
    
    import re
    
    # Check for non-English scripts (Tamil, Kannada, Devanagari) - these should block
    # This check comes first to catch script issues early
    if re.search(r'[\u0B80-\u0BFF\u0C80-\u0CFF\u0900-\u097F]', text):
        return False
    
    # Remove common punctuation and whitespace to check for alphabetic content
    # This allows spaced syllables like "hap py birth day" to pass
    text_clean = re.sub(r'[^\w\s]', '', text)  # Keep only word chars and spaces
    words = text_clean.split()
    
    # Must have at least one alphabetic word (not just numbers/punctuation)
    # This is the ONLY blocking condition for English text (besides scripts above)
    has_alphabetic_word = False
    for word in words:
        if re.search(r'[A-Za-z]', word):
            has_alphabetic_word = True
            break
    
    if not has_alphabetic_word:
        return False
    
    # If we get here, text has alphabetic words and no non-English scripts
    # This is valid English (even if imperfect ASR with spaced syllables)
    return True


def process_transcription_pipeline(raw_transcription: str) -> dict:
    """
    ⚠️  DEPRECATED: This function is part of the OLD segment-based translation pipeline.
    
    The NEW DB-driven pipeline uses translate_english_to_japanese() instead, which:
    - Reads English from source_transcription (database column)
    - Translates directly to Japanese in a single call
    - No segment processing, no gates, no per-segment placeholders
    
    This function is kept for backward compatibility but should NOT be used in new code.
    Use translate_english_to_japanese() for the new pipeline.
    
    --- OLD PIPELINE DOCUMENTATION (for reference only) ---
    
    Strict two-call pipeline for processing noisy movie dialogue transcripts.
    
    Pipeline Steps:
    Step 1 — ASR: Whisper ASR output (input)
    Step 2 — Segmentation: Split into sentence-level segments
    Step 3 — Whisper ASR Control (Strict Research Mode): Decide whether speech is transcribable
            - Output: [ACCEPT] (eligible) or [ASR_UNCERTAIN] (blocked)
            - [ASR_UNCERTAIN] → UI: 意味不明（音声が不明瞭）
            - [ACCEPT] → proceeds to Step 4
            - Rejects segments >5 seconds, multiple speakers/languages, overlapping dialogue, emotional content, repetition
            - Only accepts: single speaker, one clear language (≥90%), no overlapping dialogue, no emotional tone, ≤5 seconds, clear semantic intent, no repetition
            - Research principle: Fewer subtitles = higher accuracy. Rejection = correctness. Silence is valid data.
    Step 4 — Forensic Pre-Segmentation Filter: Block corrupted sentences
    Step 5 — CALL 1 (Forensic Meaning Gate): Process one segment at a time
            - Output: English meaning, [PARTIAL] + English, or [UNINTELLIGIBLE]
    Step 6 — HARD GATING LOGIC: 
            - IF [UNINTELLIGIBLE] → NO Call 2, output Japanese placeholder directly
            - ELSE → Send to Call 2
    Step 7 — CALL 2 (Japanese Display Layer):
            - Input: ONLY Call 1 English output (never raw ASR)
            - Output: Japanese translation
    
    CRITICAL RULES (enforced in code, not prompts):
    - [ASR_UNCERTAIN] sentences NEVER go to Call 1 or Call 2
    - [UNINTELLIGIBLE] sentences NEVER go to Call 2
    - Call 2 NEVER receives raw ASR text
    - No fallback translation for [UNINTELLIGIBLE] or [ASR_UNCERTAIN]
    - No merging of multiple Call 1 outputs
    - No inference of missing meaning
    
    Args:
        raw_transcription: Noisy transcription of dialogue from Indian movies
                    (may contain Tamil, Kannada, Hindi, or mixed)
    
    Returns:
        Dictionary with formatted output and intermediate English meaning
    """
    if not raw_transcription or not raw_transcription.strip():
        return {
            'formatted_output': '',
            'detected_languages': [],
            'normalized_text': '',
            'english_meaning': '',
            'japanese_translation': '',
            'cultural_terms': []
        }
    
    # ──────────────────────────────────────────────────────────────────────────
    # ASR QUALITY GATE: Hard programmatic gate before Call 1
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("ASR QUALITY GATE: Enforcing English-only before Call 1", flush=True)
    print("=" * 80, flush=True)
    
    # Import ASR scoring function
    from backend_python.whisper_transcribe import score_asr
    
    # Score the raw transcription
    asr_score = score_asr(raw_transcription)
    print(f"ASR Quality Score: {asr_score}", flush=True)
    
    # RELAXED: Only reject if score is extremely negative (severe corruption)
    # Allow low/negative scores to pass through (imperfect ASR with spaced syllables is acceptable)
    # Threshold set to -100 to only block truly corrupted output
    if asr_score < -100:
        print(f"[GATE] sentence={repr(raw_transcription[:60])}... reason=BLOCKED (ASR score {asr_score} < -100)", flush=True)
        return {
            'formatted_output': '',
            'detected_languages': [],
            'normalized_text': '',
            'english_meaning': '',
            'japanese_translation': '',
            'cultural_terms': []
        }
    
    print(f"[GATE] sentence={repr(raw_transcription[:60])}... reason=PASSED (ASR score {asr_score})", flush=True)
    
    # RELAXED: Check for non-English scripts (Tamil, Kannada, Devanagari) - these should block
    # But allow spaced syllables and minor ASR artifacts
    if re.search(r'[\u0B80-\u0BFF\u0C80-\u0CFF\u0900-\u097F]', raw_transcription):
        print(f"[GATE] sentence={repr(raw_transcription[:60])}... reason=BLOCKED (non-English script)", flush=True)
        return {
            'formatted_output': '',
            'detected_languages': [],
            'normalized_text': '',
            'english_meaning': '',
            'japanese_translation': '',
            'cultural_terms': []
        }
    
    # Must have at least one alphabetic word (allows spaced syllables like "hap py birth day")
    # This is a relaxed check - only blocks if there are NO alphabetic words at all
    text_clean = re.sub(r'[^\w\s]', '', raw_transcription)  # Keep only word chars and spaces
    words = text_clean.split()
    has_alphabetic_word = any(re.search(r'[A-Za-z]', word) for word in words)
    
    if not has_alphabetic_word:
        print(f"[GATE] sentence={repr(raw_transcription[:60])}... reason=BLOCKED (no alphabetic words)", flush=True)
        return {
            'formatted_output': '',
            'detected_languages': [],
            'normalized_text': '',
            'english_meaning': '',
            'japanese_translation': '',
            'cultural_terms': []
        }
    
    # If we get here, the sentence has alphabetic words and no non-English scripts
    # This allows imperfect ASR with spaced syllables to pass
    print(f"[GATE] sentence={repr(raw_transcription[:60])}... reason=PASSED (ASR quality check - has alphabetic words, no non-English scripts)", flush=True)
    
    client = get_client()
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 1: Sentence Segmentation (CRITICAL: Short segments prevent hallucination)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 1: Sentence Segmentation", flush=True)
    print("=" * 80, flush=True)
    print("Target: 8-12 seconds or 1-2 clauses per segment", flush=True)
    print("Short segments are REQUIRED even if it reduces content", flush=True)
    
    raw_sentences = segment_sentences(raw_transcription)
    print(f"Segmented {len(raw_sentences)} sentences from raw transcription", flush=True)
    if raw_sentences:
        avg_length = sum(len(s) for s in raw_sentences) / len(raw_sentences)
        print(f"Average segment length: {avg_length:.1f} characters", flush=True)
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 2: Pre-Call-1 ASR Rejection Gate (Strict)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 2: Whisper ASR Control (Strict Research Mode)", flush=True)
    print("=" * 80, flush=True)
    print("Gate function: Decide whether speech is transcribable before attempting transcription.", flush=True)
    print("Research principle: Fewer subtitles = higher accuracy. Rejection = correctness. Silence is valid data.", flush=True)
    
    # Prepare input for ASR QC
    asr_qc_input_parts = []
    for idx, sentence in enumerate(raw_sentences, start=1):
        asr_qc_input_parts.append(f"Segment {idx}: {sentence}")
    asr_qc_input = '\n'.join(asr_qc_input_parts)
    
    asr_qc_prompt = f"""Whisper ASR Control Prompt (Strict Research Mode)

You are a speech recognition engine, not a storyteller.

Your task is to decide whether speech is transcribable before attempting transcription.

━━━━━━━━━━━━━━━━━━━━━━
1️⃣ Audio acceptance rules (mandatory)
━━━━━━━━━━━━━━━━━━━━━━

ONLY transcribe audio if ALL of the following are true:

Single speaker only

One clear language dominates (≥90%)

No overlapping dialogue

No emotional shouting or argument tone

Segment length ≤ 5 seconds

Sentence has clear semantic intent

No repeated phrases used as filler

No rapid language switching

No background dialogue or movie-scene audio

If any rule fails, do NOT attempt transcription.

Segments to evaluate:
{asr_qc_input}

━━━━━━━━━━━━━━━━━━━━━━
2️⃣ Rejection behavior (CRITICAL)
━━━━━━━━━━━━━━━━━━━━━━

If the audio fails any acceptance rule:

Output exactly one line:

[ASR_UNCERTAIN]


Do not:

Guess words

Infer meaning

Normalize emotions

Translate

Clean up language

Create sentences

Fill gaps

Explain the failure

Silence or uncertainty is more correct than text.

━━━━━━━━━━━━━━━━━━━━━━
3️⃣ Transcription rules (only if accepted)
━━━━━━━━━━━━━━━━━━━━━━

If and only if audio passes all checks:

Transcribe verbatim

Preserve hesitations, cut-offs, and false starts

Do not rewrite or summarize

Do not "fix" grammar

Do not merge speakers

Do not invent missing words

If a word is unclear → use [UNINTELLIGIBLE]

Example:

I was [UNINTELLIGIBLE] about the birthday

━━━━━━━━━━━━━━━━━━━━━━
4️⃣ Hallucination prevention rules (NON-NEGOTIABLE)
━━━━━━━━━━━━━━━━━━━━━━

You must NEVER:

Invent relationships, emotions, or story flow

Complete sentences from context

Translate before transcription

Convert noise into meaningful language

Replace uncertainty with confidence

If you feel the urge to "make sense" of audio → reject it instead.

━━━━━━━━━━━━━━━━━━━━━━
5️⃣ Research principle (guiding rule)
━━━━━━━━━━━━━━━━━━━━━━

Fewer subtitles = higher accuracy
Rejection = correctness
Silence is valid data

Your job is precision, not completeness.

OUTPUT FORMAT:
Segment X:
- Result: <[ASR_UNCERTAIN] OR [ACCEPT]>"""
    
    asr_qc_results = {}
    asr_uncertain_sentences = []
    
    if raw_sentences:
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a speech recognition engine, not a storyteller. "
                            "Your task is to decide whether speech is transcribable before attempting transcription. "
                            "1️⃣ AUDIO ACCEPTANCE RULES (mandatory) - ONLY transcribe audio if ALL of the following are true: "
                            "Single speaker only, one clear language dominates (≥90%), no overlapping dialogue, "
                            "no emotional shouting or argument tone, segment length ≤ 5 seconds, sentence has clear semantic intent, "
                            "no repeated phrases used as filler, no rapid language switching, no background dialogue or movie-scene audio. "
                            "If any rule fails, do NOT attempt transcription. "
                            "2️⃣ REJECTION BEHAVIOR (CRITICAL): If the audio fails any acceptance rule, output exactly one line: [ASR_UNCERTAIN]. "
                            "Do not: guess words, infer meaning, normalize emotions, translate, clean up language, create sentences, fill gaps, explain the failure. "
                            "Silence or uncertainty is more correct than text. "
                            "3️⃣ TRANSCRIPTION RULES (only if accepted): If and only if audio passes all checks, transcribe verbatim. "
                            "Preserve hesitations, cut-offs, and false starts. Do not rewrite or summarize. Do not 'fix' grammar. "
                            "Do not merge speakers. Do not invent missing words. If a word is unclear → use [UNINTELLIGIBLE]. "
                            "4️⃣ HALLUCINATION PREVENTION RULES (NON-NEGOTIABLE): You must NEVER: invent relationships/emotions/story flow, "
                            "complete sentences from context, translate before transcription, convert noise into meaningful language, "
                            "replace uncertainty with confidence. If you feel the urge to 'make sense' of audio → reject it instead. "
                            "5️⃣ RESEARCH PRINCIPLE: Fewer subtitles = higher accuracy. Rejection = correctness. Silence is valid data. "
                            "Your job is precision, not completeness. "
                            "Output format: Segment X: - Result: <[ASR_UNCERTAIN] OR [ACCEPT]>."
                        ),
                    },
                    {"role": "user", "content": asr_qc_prompt},
                ],
                max_completion_tokens=8000,
            )
            asr_qc_output = response.choices[0].message.content.strip()
            
            # Extract ASR QC results (handles [ACCEPT] or [ASR_UNCERTAIN])
            segment_pattern = r'Segment\s+(\d+):\s*- Result:\s*(.*?)(?=Segment\s+\d+:|$)'
            matches = re.finditer(segment_pattern, asr_qc_output, re.DOTALL)
            
            for match in matches:
                segment_num = int(match.group(1))
                result = match.group(2).strip()
                asr_qc_results[segment_num] = result
                
                if result == "[ASR_UNCERTAIN]" or result == "ASR_UNCERTAIN":
                    asr_uncertain_sentences.append(segment_num)
                    print(f"  Segment {segment_num}: [ASR_UNCERTAIN] (rejected - will not reach Call 1)", flush=True)
                elif result == "[ACCEPT]" or result == "ACCEPT":
                    print(f"  Segment {segment_num}: [ACCEPT] (eligible for Call 1)", flush=True)
                else:
                    # Default: if unclear, treat as [ASR_UNCERTAIN] (conservative)
                    asr_uncertain_sentences.append(segment_num)
                    print(f"  Segment {segment_num}: [ASR_UNCERTAIN] (unclear result: {result[:50]})", flush=True)
            
            # If regex didn't work, try simpler extraction
            if not asr_qc_results:
                lines = asr_qc_output.split('\n')
                current_segment = None
                for line in lines:
                    line = line.strip()
                    if line.startswith('Segment'):
                        num_match = re.search(r'Segment\s+(\d+):', line)
                        if num_match:
                            if current_segment is not None:
                                segment_num, result = current_segment
                                asr_qc_results[segment_num] = result
                                if result not in ["[ACCEPT]", "ACCEPT"]:
                                    asr_uncertain_sentences.append(segment_num)
                            segment_num = int(num_match.group(1))
                            current_segment = (segment_num, '')
                    elif current_segment is not None and line.startswith('- Result:'):
                        result = line.replace('- Result:', '').strip()
                        current_segment = (current_segment[0], result)
                
                if current_segment is not None:
                    segment_num, result = current_segment
                    asr_qc_results[segment_num] = result
                    if result not in ["[ACCEPT]", "ACCEPT"]:
                        asr_uncertain_sentences.append(segment_num)
            
            total_blocked = len(asr_uncertain_sentences)
            total_passed = len(asr_qc_results) - total_blocked
            print(
                f"\nASR Gating Results: {total_blocked} rejected ([ASR_UNCERTAIN]), "
                f"{total_passed} accepted ([ACCEPT]) to Call 1",
                flush=True,
            )
            print("Research principle: Fewer subtitles = higher accuracy. Rejection = correctness. Silence is valid data.", flush=True)
        except Exception as e:
            print(f"ASR Gating failed: {e}, treating all segments as [ASR_UNCERTAIN] (conservative)", flush=True)
            # On failure, treat all as [ASR_UNCERTAIN] (conservative - rejection is success)
            for idx in range(1, len(raw_sentences) + 1):
                asr_qc_results[idx] = "[ASR_UNCERTAIN]"
                asr_uncertain_sentences.append(idx)
    else:
        print("No sentences to process in ASR QC", flush=True)
    
    # Prepare sentences for Call 1 - ASR_UNCERTAIN is now a quality signal, not a block
    # All sentences (both [ACCEPT] and [ASR_UNCERTAIN]) proceed to Call 1
    # has_clear_english() will decide if they reach Call 2
    cleaned_sentences = []
    for idx, original_sentence in enumerate(raw_sentences, start=1):
        asr_result = asr_qc_results.get(idx, "[ACCEPT]")
        if asr_result in ["[ACCEPT]", "ACCEPT", "[ASR_UNCERTAIN]", "ASR_UNCERTAIN"]:
            # Both [ACCEPT] and [ASR_UNCERTAIN] proceed to Call 1
            # ASR_UNCERTAIN is treated as a quality signal, not a refusal
            is_asr_uncertain = (asr_result in ["[ASR_UNCERTAIN]", "ASR_UNCERTAIN"])
            cleaned_sentences.append((idx, original_sentence, is_asr_uncertain))
            if is_asr_uncertain:
                print(f"  Segment {idx}: [ASR_UNCERTAIN] (quality signal - will proceed to Call 1 for evaluation)", flush=True)
        else:
            # Unexpected result - treat as [ASR_UNCERTAIN] but still allow through
            print(f"  Segment {idx}: Unexpected ASR result '{asr_result}' - treating as [ASR_UNCERTAIN] (quality signal)", flush=True)
            cleaned_sentences.append((idx, original_sentence, True))
    
    print(f"After ASR Gating: {len(cleaned_sentences)} segments proceeding to Call 1 ({len([s for s in cleaned_sentences if s[2]])} marked [ASR_UNCERTAIN] as quality signal)", flush=True)
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 3: Forensic Pre-Segmentation Filter
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 3: Forensic Pre-Segmentation Filter", flush=True)
    print("=" * 80, flush=True)
    
    filtered_results = []
    eligible_sentences = []
    blocked_sentences = []
    
    # NOTE: ASR_UNCERTAIN is no longer treated as a block
    # ASR_UNCERTAIN sentences proceed to Call 1 and are evaluated by has_clear_english()
    # Only truly corrupted sentences (from forensic filter) are blocked here
    
    # Process cleaned sentences through forensic filter
    # Note: cleaned_sentences is now (idx, sentence, is_asr_uncertain)
    for item in cleaned_sentences:
        if len(item) == 3:
            original_idx, cleaned_sentence, is_asr_uncertain = item
        else:
            # Backward compatibility
            original_idx, cleaned_sentence = item
            is_asr_uncertain = False
        
        filter_result = forensic_pre_segmentation_filter(cleaned_sentence)
        filter_result['sentence_id'] = original_idx
        filter_result['original_sentence'] = raw_sentences[original_idx - 1]  # Keep original for reference
        filter_result['cleaned_sentence'] = cleaned_sentence  # Store ASR QC cleaned version
        filter_result['is_asr_uncertain'] = is_asr_uncertain  # Store ASR_UNCERTAIN flag as quality signal
        filtered_results.append(filter_result)
        
        if filter_result['status'] == 'blocked':
            blocked_sentences.append(filter_result)
            reasons_str = ', '.join(filter_result['reason'])
            print(f"  Sentence {original_idx}: BLOCKED - {reasons_str}", flush=True)
        else:
            # Use cleaned sentence from ASR QC for Call 1
            # ASR_UNCERTAIN is a quality signal, not a block - sentence proceeds to Call 1
            eligible_sentences.append((original_idx, cleaned_sentence, is_asr_uncertain))
            if is_asr_uncertain:
                print(f"  Sentence {original_idx}: ELIGIBLE (marked [ASR_UNCERTAIN] as quality signal)", flush=True)
            else:
                print(f"  Sentence {original_idx}: ELIGIBLE", flush=True)
    
    print(f"\nFilter Results: {len(blocked_sentences)} blocked, {len(eligible_sentences)} eligible", flush=True)
    
    # ──────────────────────────────────────────────────────────────────────────
    # PRE-CALL-1 ACCUMULATION: Group short fragments into meaningful units
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("PRE-CALL-1 ACCUMULATION: Grouping short fragments", flush=True)
    print("=" * 80, flush=True)
    print(f"Before accumulation: {len(eligible_sentences)} sentences", flush=True)
    
    # Accumulate fragments until they meet minimum semantic length
    accumulated_sentences = accumulate_until_semantic_minimum(eligible_sentences)
    
    print(f"After accumulation: {len(accumulated_sentences)} sentences", flush=True)
    if len(accumulated_sentences) < len(eligible_sentences):
        print(f"  → {len(eligible_sentences) - len(accumulated_sentences)} fragments were accumulated into longer units", flush=True)
    print("=" * 80, flush=True)
    
    # Use accumulated sentences for Call 1
    eligible_sentences = accumulated_sentences
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 4: CALL 1 — Forensic Meaning Gate
    # (Processes one segment at a time - each segment gets independent output)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 4: CALL 1 — Forensic Meaning Gate", flush=True)
    print(f"(Processing {len(eligible_sentences)} eligible sentences independently)", flush=True)
    print("=" * 80, flush=True)
    
    # Prepare input for Call 1 (only eligible sentences)
    # Process sentences one by one to match output order
    if eligible_sentences:
        # Format eligible sentences for Call 1 (one per line, no numbering)
        # eligible_sentences is now (original_indices_list, sentence, is_asr_uncertain) after accumulation
        eligible_text_parts = []
        for item in eligible_sentences:
            if len(item) == 3:
                original_indices, sent, is_asr_uncertain = item
                # original_indices is now a list (from accumulation) or single int (if not accumulated)
                if isinstance(original_indices, list):
                    # Accumulated fragment - use the combined text
                    pass  # sent is already the combined text
                else:
                    # Single fragment (not accumulated) - treat as before
                    pass  # sent is the sentence text
            else:
                # Backward compatibility
                original_indices, sent = item
                is_asr_uncertain = False
            eligible_text_parts.append(sent)
        eligible_text = '\n'.join(eligible_text_parts)
    else:
        eligible_text = ""
        print("No eligible sentences - skipping Call 1", flush=True)
    
    call1_prompt = f"""You are a forensic language gate.

Your task is NOT to translate.
Your task is NOT to guess.
Your task is NOT to repair broken language.

You will receive automatic speech recognition (ASR) output that may contain:

Tamil

Hindi

Mixed languages

Garbled phonetics

Names

Emotional speech

You must analyze each sentence independently.

Rules (mandatory):

If you can extract a clear, unambiguous English meaning with high confidence:

Output exactly:

[CLEAR] <short, literal English meaning>


If the meaning is unclear, mixed, inferred, emotional, or partially guessed:

Output exactly:

[UNINTELLIGIBLE]


Do not:

Translate directly

Guess intent

Combine sentences

Smooth dialogue

Add names unless explicitly spoken in clear English

Explain anything

When in doubt:

Always choose [UNINTELLIGIBLE]

INPUT (one ASR segment per line):
{eligible_text}

Output format rules:

One line per input sentence

No extra commentary

ABSOLUTELY NO JAPANESE TEXT - If you output any Japanese characters, the output will be discarded

No Tamil

No summaries

Only [CLEAR] or [UNINTELLIGIBLE]

If you cannot process a sentence, output [UNINTELLIGIBLE] only. Do NOT explain in any language."""
    
    sentences = []
    
    # Only call Call 1 if there are eligible sentences
    if eligible_sentences and eligible_text:
        try:
            response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                    {"role": "system", "content": "You are a forensic language gate. Your task is NOT to translate. Your task is NOT to guess. Your task is NOT to repair broken language. You will receive automatic speech recognition (ASR) output that may contain Tamil, Hindi, mixed languages, garbled phonetics, names, emotional speech. You must analyze each sentence independently. Rules (mandatory): If you can extract a clear, unambiguous English meaning with high confidence, output exactly: [CLEAR] <short, literal English meaning>. If the meaning is unclear, mixed, inferred, emotional, or partially guessed, output exactly: [UNINTELLIGIBLE]. Do not: translate directly, guess intent, combine sentences, smooth dialogue, add names unless explicitly spoken in clear English, explain anything. When in doubt: Always choose [UNINTELLIGIBLE]. Output format rules: One line per input sentence, no extra commentary, ABSOLUTELY NO JAPANESE TEXT (any Japanese output will be discarded), no Tamil, no summaries, only [CLEAR] or [UNINTELLIGIBLE]. If you cannot process a sentence, output [UNINTELLIGIBLE] only. Do NOT explain in any language."},
                    {"role": "user", "content": call1_prompt}
                ],
                max_completion_tokens=8000
            )
            call1_output = response.choices[0].message.content.strip()
            
            # === FORENSIC INSTRUMENTATION: CALL 1 OUTPUT (RAW) ===
            print("=" * 80, flush=True)
            print("=== CALL 1 OUTPUT (RAW) ===", flush=True)
            print("=" * 80, flush=True)
            print(call1_output, flush=True)
            print("=" * 80, flush=True)
            
            # Extract sentences - NEW PRIMARY FORMAT: [CLEAR] <literal English meaning> or [UNINTELLIGIBLE]
            # Also handle backward compatibility with old formats
            sentence_results = []
            lines = call1_output.split('\n')
            i = 0
            
            # Define all refusal tags (for backward compatibility)
            refusal_tags = [
                '[UNINTELLIGIBLE]',
                '[AMBIGUOUS]',
                '[INFERENCE REQUIRED – REFUSED]',
                '[LANGUAGE MIX – UNSTABLE]',
                '[FRAGMENT – NO TRANSLATION]'
            ]
            
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    i += 1
                    continue
                
                # NEW PRIMARY FORMAT: [CLEAR] <literal English meaning> or [UNINTELLIGIBLE]
                # Check if line starts with [CLEAR]
                if line.startswith('[CLEAR]'):
                    # Extract literal English meaning after [CLEAR]
                    english_meaning = line.replace('[CLEAR]', '', 1).strip()
                    
                    # DEFENSIVE GUARD: Filter out any Japanese text from Call 1 output
                    # Call 1 must NEVER output Japanese - if it does, treat as [UNINTELLIGIBLE]
                    if re.search(r'[あ-んア-ン一-龯]', english_meaning):
                        print(f"  WARNING: Call 1 output contains Japanese text - discarding: {repr(english_meaning[:60])}", flush=True)
                        sentence_results.append({
                            'english_meaning': '[UNINTELLIGIBLE]',
                            'is_refusal': True,
                            'is_clear': False
                        })
                    else:
                        sentence_results.append({
                            'english_meaning': english_meaning,  # This is the literal English meaning extracted by Call 1
                            'is_refusal': False,
                            'is_clear': True
                        })
                    i += 1
                    continue
                
                # Check if line is [UNINTELLIGIBLE] (may or may not have text after it)
                if line.startswith('[UNINTELLIGIBLE]'):
                    # For [UNINTELLIGIBLE], we don't need the original sentence - just mark as refusal
                    sentence_results.append({
                        'english_meaning': '[UNINTELLIGIBLE]',
                        'is_refusal': True,
                        'is_clear': False
                    })
                    i += 1
                    continue
                
                # Also handle standalone [UNINTELLIGIBLE] tag
                if line == '[UNINTELLIGIBLE]':
                    sentence_results.append({
                        'english_meaning': '[UNINTELLIGIBLE]',
                        'is_refusal': True,
                        'is_clear': False
                    })
                    i += 1
                    continue
                
                # BACKWARD COMPATIBILITY: [Original fragment] → [Literal translation OR refusal tag]
                # Check if line starts with '[' (original fragment)
                if line.startswith('[') and not line.startswith('→'):
                    original_fragment = line
                    # Look for next line with arrow
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line.startswith('→'):
                            # Extract translation or refusal tag
                            translation_or_tag = next_line.replace('→', '').strip()
                            
                            # Check if it's a refusal tag (exact match or contains tag)
                            is_refusal = False
                            normalized_tag = translation_or_tag
                            
                            # First check for exact tag match (with or without brackets)
                            for tag in refusal_tags:
                                if tag == translation_or_tag or tag in translation_or_tag:
                                    normalized_tag = tag  # Normalize to exact tag
                                    is_refusal = True
                                    break
                            
                            # Also check if translation_or_tag starts with '[' and contains a refusal tag
                            if not is_refusal and translation_or_tag.startswith('['):
                                for tag in refusal_tags:
                                    if tag in translation_or_tag:
                                        normalized_tag = tag
                                        is_refusal = True
                                        break
                            
                            # Create sentence result
                            # STRICT: If not a refusal tag, must be [UNINTELLIGIBLE] (no backward compatibility bypass)
                            if not is_refusal:
                                normalized_tag = '[UNINTELLIGIBLE]'
                                is_refusal = True
                            sentence_results.append({
                                'original_fragment': original_fragment,
                                'english_meaning': normalized_tag,
                                'is_refusal': is_refusal
                            })
                            i += 2
                            continue
                
                # BACKWARD COMPATIBILITY: Old format "Sentence X: - Language: ... - English meaning: ..."
                sentence_match = re.match(r'Sentence\s+(\d+):', line, re.IGNORECASE)
                if sentence_match:
                    sentence_num = int(sentence_match.group(1))
                    language = 'Unclear'
                    english_meaning = '[UNINTELLIGIBLE]'
                    
                    # Look for Language and English meaning in following lines
                    j = i + 1
                    while j < len(lines) and j < i + 5:  # Look ahead max 5 lines
                        next_line = lines[j].strip()
                        if next_line.startswith('- Language:'):
                            language = next_line.replace('- Language:', '').strip()
                        elif next_line.startswith('- English meaning:'):
                            english_meaning = next_line.replace('- English meaning:', '').strip()
                            
                            # DEFENSIVE GUARD: Filter out any Japanese text from Call 1 output
                            if re.search(r'[あ-んア-ン一-龯]', english_meaning):
                                print(f"  WARNING: Call 1 output contains Japanese text (backward compat) - discarding: {repr(english_meaning[:60])}", flush=True)
                                english_meaning = '[UNINTELLIGIBLE]'
                            
                            # Check if it's a refusal tag
                            for tag in refusal_tags:
                                if tag in english_meaning:
                                    english_meaning = tag
                                    break
                            break
                        j += 1
                    
                    sentence_results.append({
                        'sentence_num': sentence_num,
                    'language': language,
                        'english_meaning': english_meaning,
                        'is_refusal': english_meaning in refusal_tags
                    })
                    i = j + 1 if j < len(lines) else i + 1
                    continue
                
                # BACKWARD COMPATIBILITY: Standalone refusal tags or [PARTIAL]
                if line in refusal_tags or line == "[UNINTELLIGIBLE]":
                    sentence_results.append({
                        'english_meaning': line if line in refusal_tags else '[UNINTELLIGIBLE]',
                        'is_refusal': True
                    })
                    i += 1
                    continue
                
                # BACKWARD COMPATIBILITY: [PARTIAL] with following line
                # STRICT: [PARTIAL] is deprecated - treat as [UNINTELLIGIBLE]
                if line == "[PARTIAL]":
                    sentence_results.append({
                        'english_meaning': '[UNINTELLIGIBLE]',
                        'is_refusal': True
                    })
                    i += 1
                    continue
                
                # If line doesn't match any pattern, treat as UNINTELLIGIBLE (strict mode)
                # This prevents accepting raw ASR text or unexpected formats
                sentence_results.append({
                    'english_meaning': '[UNINTELLIGIBLE]',
                    'is_refusal': True
                })
                i += 1
                continue
                
                i += 1
            
            # DEFENSIVE GUARD: Filter out any Japanese text from Call 1 output
            # Call 1 must NEVER output Japanese - if any result contains Japanese, replace with [UNINTELLIGIBLE]
            for result in sentence_results:
                english_meaning = result.get('english_meaning', '')
                if english_meaning and english_meaning != '[UNINTELLIGIBLE]':
                    # Check for Japanese characters (Hiragana, Katakana, Kanji)
                    if re.search(r'[あ-んア-ン一-龯]', english_meaning):
                        print(f"  WARNING: Call 1 output contains Japanese text - discarding: {repr(english_meaning[:60])}", flush=True)
                        result['english_meaning'] = '[UNINTELLIGIBLE]'
                        result['is_refusal'] = True
                        result['is_clear'] = False
            
            # Match outputs to eligible sentences by order
            # For [CLEAR] sentences: pass literal English meaning to Call 2 (extracted by Call 1)
            # For [UNINTELLIGIBLE] sentences: mark as refusal, will be discarded
            # NOTE: After accumulation, item format is (original_indices_list, sentence, is_asr_uncertain)
            for idx, item in enumerate(eligible_sentences):
                # Handle new tuple format after accumulation
                if len(item) == 3:
                    original_indices, original_sentence_text, is_asr_uncertain = item
                    # original_indices is a list (accumulated) or single int (not accumulated)
                    if isinstance(original_indices, list):
                        # Accumulated fragments - apply result to all original indices
                        fragment_indices = original_indices
                    else:
                        # Single fragment (not accumulated) - wrap in list for uniform handling
                        fragment_indices = [original_indices]
                else:
                    # Backward compatibility
                    original_indices = item[0]
                    original_sentence_text = item[1]
                    is_asr_uncertain = False
                    if isinstance(original_indices, list):
                        fragment_indices = original_indices
                    else:
                        fragment_indices = [original_indices]
                
                # Try to find matching sentence by index (Call 1 returns one result per accumulated sentence)
                if idx < len(sentence_results):
                    result = sentence_results[idx]
                    
                    # Apply this result to ALL original fragments that were accumulated
                    for fragment_idx in fragment_indices:
                        # Check if it's a [CLEAR] sentence (new format)
                        if result.get('is_clear', False):
                            # For [CLEAR], the english_meaning contains the literal English meaning extracted by Call 1
                            # This is what we pass to Call 2 for translation to Japanese
                            english_meaning = result.get('english_meaning', '')
                            sentences.append({
                                'sentence_num': fragment_idx,
                                'language': result.get('language', 'Unclear'),
                                'normalized_source': '',
                                'english_meaning': english_meaning,  # This is the literal English meaning from Call 1
                                'is_clear': True,  # CRITICAL: Preserve is_clear flag from Call 1 output
                                'is_asr_uncertain': is_asr_uncertain  # Preserve ASR_UNCERTAIN flag as quality signal
                            })
                        # Check if it's a [UNINTELLIGIBLE] sentence
                        elif result.get('is_refusal', False) or result.get('english_meaning', '') == '[UNINTELLIGIBLE]':
                            # Mark as UNINTELLIGIBLE - will be discarded before Call 2
                            sentences.append({
                                'sentence_num': fragment_idx,
                                'language': 'Unclear',
                                'normalized_source': '',
                                'english_meaning': '[UNINTELLIGIBLE]',
                                'is_clear': False  # Explicitly mark as not clear
                            })
                        else:
                            # STRICT: Only [CLEAR] sentences pass. All others are [UNINTELLIGIBLE]
                            # This prevents backward compatibility bypasses
                            sentences.append({
                                'sentence_num': fragment_idx,
                                'language': 'Unclear',
                                'normalized_source': '',
                                'english_meaning': '[UNINTELLIGIBLE]',
                                'is_clear': False  # Explicitly mark as not clear
                            })
                else:
                    # Missing output - treat as UNINTELLIGIBLE for all fragments
                    for fragment_idx in fragment_indices:
                        sentences.append({
                            'sentence_num': fragment_idx,
                            'language': 'Unclear',
                            'normalized_source': '',
                            'english_meaning': '[UNINTELLIGIBLE]',
                            'is_clear': False  # Explicitly mark as not clear
                        })
            
            print(f"Call 1 complete - {len(sentences)} sentences processed", flush=True)
            for sent in sentences[:5]:
                print(f"  Sentence {sent['sentence_num']}: ({sent['language']}) {sent['english_meaning'][:60]}...", flush=True)
        except Exception as e:
            print(f"Call 1 failed: {e}, using fallback", flush=True)
            sentences = []  # Will be handled by merging logic below
    else:
        print("Skipping Call 1 - no eligible sentences (all blocked by filter)", flush=True)
        sentences = []  # All sentences were blocked
    
    # ──────────────────────────────────────────────────────────────────────────
    # Merge blocked sentences with Call 1 results
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("Merging blocked sentences with Call 1 results", flush=True)
    print("=" * 80, flush=True)
    
    # Create a dict of all sentences by original sentence ID
    all_sentences_dict = {}
    
    # Add blocked sentences (from forensic filter only)
    # NOTE: ASR_UNCERTAIN sentences are no longer in blocked_sentences
    # They proceed to Call 1 and are evaluated by has_clear_english()
    for blocked in blocked_sentences:
        original_id = blocked['sentence_id']
        is_asr_uncertain = 'asr_uncertain' in blocked['reason']
        
        all_sentences_dict[original_id] = {
            'sentence_num': original_id,
            'language': 'Unclear',
            'normalized_source': '',
            'english_meaning': '[ASR_UNCERTAIN]' if is_asr_uncertain else '[UNINTELLIGIBLE]',
            'blocked_reason': blocked['reason'],
            'is_asr_uncertain': is_asr_uncertain,  # Flag for special handling
            'japanese_placeholder': blocked.get('japanese_placeholder', None)  # Store placeholder if available
        }
        reason_str = 'ASR_UNCERTAIN' if is_asr_uncertain else ', '.join(blocked['reason'])
        print(f"  Added blocked sentence {original_id}: {reason_str}", flush=True)
    
    # Add Call 1 results (overwrite if any ID collision, though there shouldn't be)
    for sent in sentences:
        original_id = sent['sentence_num']
        all_sentences_dict[original_id] = sent
        print(f"  Added Call 1 result for sentence {original_id}", flush=True)
    
    # Sort by sentence ID to preserve original order
    all_sentences = [all_sentences_dict[orig_id] for orig_id in sorted(all_sentences_dict.keys())]
    
    print(f"Merged result: {len(all_sentences)} total sentences ({len(blocked_sentences)} blocked, {len(sentences)} from Call 1)", flush=True)
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 5: HARD GATING LOGIC (CRITICAL - NO EXCEPTIONS)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 5: HARD GATING LOGIC", flush=True)
    print("=" * 80, flush=True)
    print("Enforcing strict rule: All refusal tags → NO Call 2, direct Japanese placeholder", flush=True)
    
    # Define all refusal tags that should be hard-gated
    refusal_tags = [
        '[UNINTELLIGIBLE]',
        '[AMBIGUOUS]',
        '[INFERENCE REQUIRED – REFUSED]',
        '[LANGUAGE MIX – UNSTABLE]',
        '[FRAGMENT – NO TRANSLATION]'
    ]
    
    def is_refusal_tag(text: str) -> bool:
        """Check if text is a refusal tag (exact match or contains tag)"""
        text = text.strip()
        for tag in refusal_tags:
            if tag == text or tag in text:
                return True
        return False
    
    # Initialize Japanese translations dict for all sentences
    # Refusal tag sentences get Japanese placeholder immediately (HARD GATE)
    japanese_translations_dict = {}
    call2_sentences = []
    
    for sent in all_sentences:
        english_meaning = sent['english_meaning'].strip()
        sentence_num = sent['sentence_num']
        
        # NOTE: ASR_UNCERTAIN is no longer a fatal block
        # ASR_UNCERTAIN sentences have already gone through Call 1
        # They are now evaluated by has_clear_english() like any other sentence
        
        # HARD GATE: IF Call 1 output is any refusal tag → DO NOT call Call 2
        if is_refusal_tag(english_meaning):
            # Blocked sentence - store empty string (NO placeholder at sentence level)
            # Placeholder will be added at segment level only if ALL sentences fail
            japanese_translations_dict[sentence_num] = ""
            print(f"  Sentence {sentence_num}: {english_meaning} → HARD GATED (skipping Call 2, storing empty)", flush=True)
        else:
            # RELAXED VALIDATION: Primary check is has_clear_english() - this allows imperfect ASR to pass
            # Check for non-English scripts (Tamil, Kannada, Devanagari) first
            tamil_script = re.search(r'[\u0B80-\u0BFF]', english_meaning)
            kannada_script = re.search(r'[\u0C80-\u0CFF]', english_meaning)
            devanagari_script = re.search(r'[\u0900-\u097F]', english_meaning)
            
            if tamil_script or kannada_script or devanagari_script:
                # Non-English script detected - treat as [UNINTELLIGIBLE]
                print(f"[GATE] sentence={repr(english_meaning[:60])}... reason=BLOCKED (non-English script)", flush=True)
                japanese_translations_dict[sentence_num] = ""  # Store empty, no sentence-level placeholder
            elif has_clear_english(english_meaning):
                # PRIMARY CHECK: has_clear_english() passes → accept (allows imperfect ASR like "hap py birth day")
                # This is the main gate - if it passes, sentence goes to Call 2 regardless of is_clear flag
                # ASR_UNCERTAIN sentences that pass has_clear_english() will reach Call 2
                asr_uncertain_note = " (was [ASR_UNCERTAIN], but passed has_clear_english())" if sent.get('is_asr_uncertain', False) else ""
                print(f"[GATE] sentence={repr(english_meaning[:60])}... reason=PASSED (has_clear_english check){asr_uncertain_note}", flush=True)
                call2_sentences.append(sent)
            elif sent.get('is_clear', False):
                # Fallback: Marked as [CLEAR] from Call 1 → will be sent to Call 2
                # (This should rarely be needed if has_clear_english() is working correctly)
                print(f"[GATE] sentence={repr(english_meaning[:60])}... reason=PASSED ([CLEAR] flag)", flush=True)
                call2_sentences.append(sent)
            else:
                # Not marked as [CLEAR] and has_clear_english() fails → block
                print(f"[GATE] sentence={repr(english_meaning[:60])}... reason=BLOCKED (no clear English)", flush=True)
                japanese_translations_dict[sentence_num] = ""  # Store empty, no sentence-level placeholder
    
    print(f"\nHard Gate Results: {len(japanese_translations_dict)} sentences gated, {len(call2_sentences)} sentences to Call 2", flush=True)
    
    # === FORENSIC VALIDATION: Ensure Call 2 only runs if [CLEAR] sentences exist ===
    # Debug: Show all call2_sentences with their is_clear flags before counting
    print("=" * 80, flush=True)
    print("=== DEBUG: Call 2 Sentences Before [CLEAR] Count Check ===", flush=True)
    print("=" * 80, flush=True)
    for sent in call2_sentences:
        sentence_num = sent.get('sentence_num', '?')
        english_meaning = sent.get('english_meaning', '')
        is_clear = sent.get('is_clear', False)
        is_asr_uncertain = sent.get('is_asr_uncertain', False)
        source = "Call 1 [CLEAR]" if is_clear else "has_clear_english() fallback"
        print(f"  Sentence {sentence_num}: is_clear={is_clear}, source={source}, meaning={repr(english_meaning[:60])}", flush=True)
    print("=" * 80, flush=True)
    
    # Count actual [CLEAR] sentences - MUST have is_clear=True flag
    clear_sentences_count = sum(1 for sent in call2_sentences if sent.get('is_clear', False) and sent.get('english_meaning', '').strip() and not is_refusal_tag(sent.get('english_meaning', '')))
    
    print(f"Clear sentences count: {clear_sentences_count} (out of {len(call2_sentences)} total call2_sentences)", flush=True)
    
    if clear_sentences_count == 0:
        print("=" * 80, flush=True)
        print("❌ FORENSIC BLOCK: Zero [CLEAR] sentences detected. Call 2 MUST NOT RUN.", flush=True)
        print("=" * 80, flush=True)
        call2_sentences = []  # Force empty - Call 2 will be skipped
    
    # Build English meaning string (for Call 2 and storage) - only non-refusal-tagged sentences
    english_meanings_text = '\n'.join([f"Sentence {sent['sentence_num']}: {sent['english_meaning']}" for sent in all_sentences])
    
    # VALIDATION: Ensure we have at least some sentences
    if not all_sentences or len(all_sentences) == 0:
        print(f"ERROR: No sentences to process (all filtered or Call 1 failed).", flush=True)
        return {
            'formatted_output': '',
            'detected_languages': [],
            'normalized_text': '',
            'english_meaning': '',
            'japanese_translation': '',
            'cultural_terms': []
        }
    
    # ──────────────────────────────────────────────────────────────────────────
    # STEP 6: CALL 2 — Japanese Translation (ONLY for validated meaning)
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("STEP 6: CALL 2 — Japanese Translation", flush=True)
    print("=" * 80, flush=True)
    print(f"Processing {len(call2_sentences)} sentences with validated meaning", flush=True)
    print(f"Skipping {len(japanese_translations_dict)} refusal-tagged sentences (already hard-gated)", flush=True)
    
    # Prepare input for Call 2 (ONLY validated Call 1 English output, NEVER raw ASR)
    if call2_sentences:
        call2_input_parts = []
        for sent in call2_sentences:
            english_meaning = sent['english_meaning'].strip()
            # SAFETY CHECK: Double-verify refusal tags never reach Call 2
            if is_refusal_tag(english_meaning):
                print(f"ERROR: Refusal tag detected in Call 2 input - removing (should not happen): {english_meaning}", flush=True)
                # Remove from call2_sentences and store empty (no sentence-level placeholder)
                sentence_num = sent['sentence_num']
                japanese_translations_dict[sentence_num] = ""  # Store empty, no sentence-level placeholder
                continue
            # CRITICAL: Only send Call 1 English output, never raw ASR text
            call2_input_parts.append(english_meaning)
        call2_input_text = '\n'.join(call2_input_parts)
        
        # === FORENSIC INSTRUMENTATION: CALL 2 INPUT (FINAL) ===
        print("=" * 80, flush=True)
        print("=== CALL 2 INPUT (FINAL) ===", flush=True)
        print("=" * 80, flush=True)
        print(call2_input_text, flush=True)
        print("=" * 80, flush=True)
        
        # FORENSIC VALIDATION: Check for Tamil/Hindi/mixed ASR content
        tamil_script = re.search(r'[\u0B80-\u0BFF]', call2_input_text)
        kannada_script = re.search(r'[\u0C80-\u0CFF]', call2_input_text)
        devanagari_script = re.search(r'[\u0900-\u097F]', call2_input_text)
        common_names = re.search(r'\b(Divya|Arun|Ravi|Priya|Kumar)\b', call2_input_text, re.IGNORECASE)
        
        if tamil_script or kannada_script or devanagari_script:
            print(f"❌ FORENSIC FAILURE: Non-English script detected in Call 2 input!", flush=True)
            if tamil_script:
                print(f"   Tamil script found: {tamil_script.group()}", flush=True)
            if kannada_script:
                print(f"   Kannada script found: {kannada_script.group()}", flush=True)
            if devanagari_script:
                print(f"   Devanagari script found: {devanagari_script.group()}", flush=True)
        if common_names:
            print(f"⚠️  WARNING: Common Indian name detected (may indicate ASR leakage): {common_names.group()}", flush=True)
        
        # Update call2_sentences to remove any refusal tags that slipped through
        call2_sentences = [s for s in call2_sentences if not is_refusal_tag(s['english_meaning'].strip())]
        
        # === HARD PROGRAMMATIC GATE: Call 2 MUST NOT run without valid clear English ===
        if not has_clear_english(call2_input_text):
            print("=" * 80, flush=True)
            print("🚫 TRANSLATION BLOCKED: Call 1 output does not pass has_clear_english() check", flush=True)
            print(f"   call2_input_text: {repr(call2_input_text[:100])}", flush=True)
            print(f"   has_clear_english result: {has_clear_english(call2_input_text)}", flush=True)
            print("=" * 80, flush=True)
            call2_sentences = []  # Force empty - Call 2 will be skipped
            call2_input_text = ""  # Ensure empty
    else:
        call2_input_text = ""
        print("No sentences to translate in Call 2 (all are refusal-tagged - hard-gated)", flush=True)
    
    call2_prompt = f"""You are a STRICT LITERAL TRANSLATION engine.

This is a hard constraint, not a guideline.

=== ABSOLUTE RULES (MUST FOLLOW) ===

1. Translate ONLY what is explicitly said in the source sentence.
   - No interpretation
   - No explanation
   - No paraphrasing
   - No summarizing
   - No meaning reconstruction

2. NEVER describe:
   - clarity or lack of clarity
   - intelligibility
   - audio quality
   - missing context
   - speaker intent
   - emotional state
   - whether meaning can or cannot be recovered

3. FORBIDDEN OUTPUT (must never appear in any form):
   - 意味不明
   - 音声が不明瞭
   - 意味を復元
   - 分からない
   - 不明
   - 解釈
   - any explanation of uncertainty
   - any commentary about translation difficulty

4. If a sentence does NOT contain concrete, translatable surface content:
   - Output an EMPTY LINE ONLY
   - Do NOT explain why
   - Do NOT apologize
   - Do NOT add placeholder text

5. Silent failure is mandatory.
   Empty output is the ONLY allowed failure mode.

6. Names, vocatives, short utterances, fillers, and emotional fragments
   must be translated literally if possible.
   Examples:
   - "Hey, Arun!" → 「やあ、アルン！」
   - "Sorry." → 「ごめん」
   - "Please." → 「お願い」

7. If the sentence content is vague but still literal, translate it literally.
   If it has NO literal surface meaning, output EMPTY.

=== OUTPUT FORMAT ===

- Output ONLY the Japanese translation lines
- One line per input sentence
- No numbering
- No explanations
- No extra text
- No placeholders

Any violation of the above rules invalidates the output.

INPUT (one sentence per line):
{call2_input_text}"""
    
    # Post-generation enforcement: Retry mechanism with katakana checking
    max_retries = 3
    japanese_sentences = []
    cultural_terms_list = []
    
    # === FINAL HARD PROGRAMMATIC GATE: Call 2 MUST NOT run without valid clear English ===
    # This is the ONLY path to Call 2 API invocation
    if not call2_sentences or not call2_input_text or not has_clear_english(call2_input_text):
        print("=" * 80, flush=True)
        print("🚫 TRANSLATION BLOCKED: Final gate check failed", flush=True)
        print(f"   call2_sentences length: {len(call2_sentences) if call2_sentences else 0}", flush=True)
        print(f"   call2_input_text length: {len(call2_input_text) if call2_input_text else 0}", flush=True)
        print(f"   call2_input_text preview: {repr(call2_input_text[:100]) if call2_input_text else 'None'}", flush=True)
        print(f"   has_clear_english result: {has_clear_english(call2_input_text) if call2_input_text else False}", flush=True)
        print("=" * 80, flush=True)
        # japanese_translations_dict already contains empty strings for blocked sentences
        # HARD STOP: Do not proceed to Call 2 API call
    else:
        # === TRANSLATION ALLOWED: All gates passed, proceeding to Call 2 ===
        print("=" * 80, flush=True)
        print("✅ TRANSLATION ALLOWED: has_clear_english() returned True", flush=True)
        print(f"   call2_sentences count: {len(call2_sentences)}", flush=True)
        print(f"   call2_input_text length: {len(call2_input_text)}", flush=True)
        print(f"   call2_input_text preview: {repr(call2_input_text[:200])}", flush=True)
        print(f"   has_clear_english result: {has_clear_english(call2_input_text)}", flush=True)
        print("=" * 80, flush=True)
        
        # FATAL ASSERTION: Double-check before API call (defensive programming)
        if not has_clear_english(call2_input_text):
            raise RuntimeError("CALL2_REACHED: Call 2 invoked without clear English. This is a control-flow violation.")
        
        # === CALL 2 API INVOCATION: ONLY PATH ===
        # This loop runs ONLY when has_clear_english() returns True
        for attempt in range(max_retries):
            try:
                if attempt == 0:
                    current_prompt = call2_prompt
                else:
                    katakana_found = re.findall(r'[ァ-ヶー]', '\n'.join([s.get('japanese_translation', '') for s in japanese_sentences]))
                    katakana_words = re.findall(r'[ァ-ヶー]+', '\n'.join([s.get('japanese_translation', '') for s in japanese_sentences]))
                    current_prompt = f"""CRITICAL CORRECTION REQUIRED:

Your previous output contained katakana characters. You MUST rewrite with ZERO katakana.

ENGLISH SOURCE (for meaning reference):
{call2_input_text[:2000]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MANDATORY CORRECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

REPLACE katakana words with kanji/hiragana equivalents that mean the same thing.
Preserve ALL meaning from the English source.
Use Kanji + Hiragana ONLY. Do NOT use katakana.

STRICT LITERAL TRANSLATION RULES (MANDATORY):
- Translate ONLY what is explicitly said. No interpretation or explanation.
- NEVER output: 意味不明, 音声が不明瞭, 意味を復元, 分からない, 不明, 解釈, or any refusal/explanation text.
- If a sentence cannot be translated, output an EMPTY LINE ONLY. Do NOT explain why.
- Silent failure is mandatory. Empty output is the ONLY allowed failure mode.

OUTPUT FORMAT:
- Output ONLY Japanese translation lines
- One line per input sentence
- No numbering, explanations, extra text, or placeholders"""
                
                response = client.chat.completions.create(
                model="gpt-5",
                messages=[
                        {"role": "system", "content": "You are a STRICT LITERAL TRANSLATION engine. This is a hard constraint, not a guideline. ABSOLUTE RULES: 1. Translate ONLY what is explicitly said. No interpretation, explanation, paraphrasing, summarizing, or meaning reconstruction. 2. NEVER describe clarity, intelligibility, audio quality, missing context, speaker intent, emotional state, or whether meaning can be recovered. 3. FORBIDDEN OUTPUT (must never appear): 意味不明, 音声が不明瞭, 意味を復元, 分からない, 不明, 解釈, any explanation of uncertainty, any commentary about translation difficulty. 4. If a sentence does NOT contain concrete, translatable surface content: Output an EMPTY LINE ONLY. Do NOT explain why. Do NOT apologize. Do NOT add placeholder text. 5. Silent failure is mandatory. Empty output is the ONLY allowed failure mode. 6. Names, vocatives, short utterances, fillers, and emotional fragments must be translated literally if possible. 7. If sentence content is vague but still literal, translate it literally. If it has NO literal surface meaning, output EMPTY. OUTPUT FORMAT: Output ONLY Japanese translation lines. One line per input sentence. No numbering, explanations, extra text, or placeholders. SCRIPT RULES: Use kanji + hiragana ONLY. Do NOT use katakana. Do NOT use romaji. Any violation of the above rules invalidates the output."},
                    {"role": "user", "content": current_prompt}
                ],
                    max_completion_tokens=6000
                )
                
                # === DEBUG: RAW CALL 2 API RESPONSE (BEFORE ANY PROCESSING) ===
                print("=" * 80, flush=True)
                print("=== DEBUG: RAW CALL 2 API RESPONSE ===", flush=True)
                print("=" * 80, flush=True)
                print(f"Response type: {type(response)}", flush=True)
                print(f"Response object: {response}", flush=True)
                print(f"Response.choices: {response.choices}", flush=True)
                print(f"Response.choices length: {len(response.choices) if response.choices else 0}", flush=True)
                if response.choices and len(response.choices) > 0:
                    print(f"Response.choices[0]: {response.choices[0]}", flush=True)
                    print(f"Response.choices[0].message: {response.choices[0].message}", flush=True)
                    print(f"Response.choices[0].message.content: {repr(response.choices[0].message.content)}", flush=True)
                    print(f"Response.choices[0].message.content (raw): {response.choices[0].message.content}", flush=True)
                else:
                    print("WARNING: response.choices is empty or None!", flush=True)
                print("=" * 80, flush=True)
                
                call2_output = response.choices[0].message.content.strip()
                
                # === DEBUG: CALL 2 OUTPUT AFTER STRIP ===
                print("=" * 80, flush=True)
                print("=== DEBUG: CALL 2 OUTPUT (AFTER .strip()) ===", flush=True)
                print("=" * 80, flush=True)
                print(f"call2_output type: {type(call2_output)}", flush=True)
                print(f"call2_output length: {len(call2_output) if call2_output else 0}", flush=True)
                print(f"call2_output (repr): {repr(call2_output)}", flush=True)
                print(f"call2_output (raw): {call2_output}", flush=True)
                print("=" * 80, flush=True)
                
                # Extract Japanese translations - NEW FORMAT: numbered list (1. <translation>, 2. <translation>, etc.)
                # Also handle backward compatibility with simple line-by-line format
                lines = call2_output.split('\n')
                output_lines = []
                i = 0
                while i < len(lines):
                    line = lines[i].strip()
                    if not line:
                        i += 1
                        continue
                    
                    # Check for numbered list format: "1. <translation>" or "1 <translation>"
                    numbered_match = re.match(r'^\d+[\.\)]\s*(.+)$', line)
                    if numbered_match:
                        translation = numbered_match.group(1).strip()
                        # Post-process: Filter out refusal phrases
                        if is_refusal_phrase(translation):
                            output_lines.append("")  # Discard refusal text, treat as empty
                        else:
                            output_lines.append(translation)
                        i += 1
                        continue
                    
                    # Simple line-by-line format
                    # Post-process: Filter out refusal phrases (no sentence-level placeholders)
                    if is_refusal_phrase(line):
                        # Discard refusal text - treat as empty
                        output_lines.append("")  # Store empty instead of refusal text
                    else:
                        output_lines.append(line)
                    i += 1
                
                # === DEBUG: EXTRACTED OUTPUT LINES ===
                print("=" * 80, flush=True)
                print("=== DEBUG: EXTRACTED OUTPUT LINES (BEFORE MATCHING) ===", flush=True)
                print("=" * 80, flush=True)
                print(f"output_lines count: {len(output_lines)}", flush=True)
                print(f"output_lines: {output_lines}", flush=True)
                print(f"call2_sentences count: {len(call2_sentences)}", flush=True)
                for idx, line in enumerate(output_lines):
                    print(f"  output_lines[{idx}]: {repr(line)}", flush=True)
                print("=" * 80, flush=True)
                
                # Match outputs to Call 2 sentences by order and merge into hard-gated dict
                for idx, sent in enumerate(call2_sentences):
                    original_sentence_id = sent['sentence_num']
                    if idx < len(output_lines):
                        japanese_translation = output_lines[idx].strip()
                        # Post-process: Filter out refusal phrases - treat as empty (no sentence-level placeholders)
                        if japanese_translation and not is_refusal_phrase(japanese_translation):
                            japanese_translations_dict[original_sentence_id] = japanese_translation
                        else:
                            # Empty or refusal phrase from Call 2 - do not store (will remain empty)
                            if is_refusal_phrase(japanese_translation):
                                print(f"  Warning: Call 2 returned refusal phrase for sentence {original_sentence_id} - discarding, treating as empty", flush=True)
                            else:
                                print(f"  Warning: Call 2 returned empty translation for sentence {original_sentence_id}", flush=True)
                    else:
                        # Missing output - Call 2 didn't return enough lines
                        # Do NOT assign refusal placeholder - leave empty (sentence passed gate, just no translation)
                        print(f"  Warning: Call 2 returned insufficient output for sentence {original_sentence_id} (expected {idx + 1} lines, got {len(output_lines)})", flush=True)
                        # Do not add to dict - will remain empty
                
                # === DEBUG: STORED IN DICTIONARY ===
                print("=" * 80, flush=True)
                print("=== DEBUG: STORED IN japanese_translations_dict ===", flush=True)
                print("=" * 80, flush=True)
                for idx, sent in enumerate(call2_sentences):
                    sentence_num = sent['sentence_num']
                    stored = japanese_translations_dict.get(sentence_num, "NOT FOUND")
                    print(f"  Sentence {sentence_num}: {repr(stored)}", flush=True)
                print("=" * 80, flush=True)
                
                # Check for katakana in Call 2 results only
                call2_japanese_text = '\n'.join([japanese_translations_dict.get(sent['sentence_num'], '') for sent in call2_sentences])
                katakana_chars = re.findall(r'[ァ-ヶー]', call2_japanese_text)
                if katakana_chars:
                    print(f"Attempt {attempt + 1}: Katakana detected: {set(katakana_chars[:10])}", flush=True)
                    if attempt < max_retries - 1:
                        print(f"Retrying with correction instruction...", flush=True)
                        continue
                    else:
                        # Final attempt: Replace katakana with meaning-preserving alternatives
                        print(f"Final attempt failed. Replacing katakana with meaning-preserving alternatives...", flush=True)
                        for sent in call2_sentences:
                            sentence_num = sent['sentence_num']
                            if sentence_num in japanese_translations_dict:
                                japanese_translations_dict[sentence_num] = replace_katakana_with_meaning(
                                    japanese_translations_dict[sentence_num], 
                                    english_source=call2_input_text
                                )
                else:
                    print(f"✓ Attempt {attempt + 1}: Katakana-free output verified", flush=True)
                    break
                    
            except Exception as e:
                print(f"Translation attempt {attempt + 1} failed: {e}", flush=True)
                # FAILURE IS ACCEPTABLE - Do NOT retry failed segments
                # Do NOT assign refusal placeholders - sentences passed gate, just API failed
                # Leave empty (will remain empty in final output)
                print("Call 2 API failure - sentences passed gate but no translation produced (will remain empty)", flush=True)
                # Do not add to dict - sentences will remain empty (correct behavior for API failure)
                # Break immediately - do not retry on API failures
                break
    
    # Convert japanese_translations_dict to japanese_sentences list format
    # This includes both blocked sentences (empty) and Call 2 results (translations or empty)
    japanese_sentences = []
    for sent in all_sentences:
        sentence_num = sent['sentence_num']
        
        # Determine if sentence was blocked or passed gate (both store empty if no translation)
        if sentence_num in japanese_translations_dict:
            # Sentence was processed - use stored value (translation or empty, no sentence-level placeholders)
            japanese_translation = japanese_translations_dict[sentence_num]
        else:
            # Sentence not in dict - leave empty (no sentence-level placeholders)
            # Placeholder will be added at segment level only if ALL sentences fail
            japanese_translation = ""
        
        japanese_sentences.append({
            'sentence_num': sentence_num,
            'japanese_translation': japanese_translation,
            'cultural_explanation': ''  # Not used in new format
        })
    
    # Sort by sentence number to maintain order
    japanese_sentences.sort(key=lambda x: x['sentence_num'])
    
    # ABSOLUTE FINAL KATAKANA REPLACEMENT (preserving meaning)
    all_japanese_text = '\n'.join([s.get('japanese_translation', '') for s in japanese_sentences])
    katakana_check = re.findall(r'[ァ-ヶー]', all_japanese_text)
    if katakana_check:
        print(f"Final check: Replacing remaining katakana with meaning-preserving alternatives...", flush=True)
        for sent in japanese_sentences:
            if sent.get('japanese_translation'):
                sent['japanese_translation'] = replace_katakana_with_meaning(
                    sent['japanese_translation'], 
                    english_source=english_meanings_text
                )
    
    final_katakana_check = re.findall(r'[ァ-ヶー]', '\n'.join([s.get('japanese_translation', '') for s in japanese_sentences]))
    if final_katakana_check:
        print(f"WARNING: Katakana still present after replacement: {set(final_katakana_check[:10])}", flush=True)
    else:
        print(f"✓ Final katakana check: ZERO katakana confirmed", flush=True)
    
    print(f"Call 2 complete - {len(japanese_sentences)} sentences translated", flush=True)
    
    # ──────────────────────────────────────────────────────────────────────────
    # FINAL OUTPUT FORMATTING
    # ──────────────────────────────────────────────────────────────────────────
    print("=" * 80, flush=True)
    print("PIPELINE COMPLETE - Formatting Output", flush=True)
    print("=" * 80, flush=True)
    
    # Build formatted output with sentence-by-sentence structure
    formatted_output_lines = []
    
    # Match sentences from all_sentences (merged blocked + Call 1) with Call 2
    for sent in all_sentences:
        sentence_num = sent['sentence_num']
        matching_jp = next((s for s in japanese_sentences if s.get('sentence_num') == sentence_num), None)
        
        formatted_output_lines.append(f"Sentence {sentence_num}:")
        formatted_output_lines.append(f"- Language: {sent['language']}")
        formatted_output_lines.append(f"- English meaning: {sent['english_meaning']}")
        # Add blocked reason if present (for debugging)
        if sent.get('blocked_reason'):
            formatted_output_lines.append(f"- Filter reason: {', '.join(sent['blocked_reason'])}")
        if matching_jp:
            formatted_output_lines.append(f"- 日本語訳: {matching_jp.get('japanese_translation', '')}")
        else:
            formatted_output_lines.append(f"- 日本語訳: (not translated)")
        formatted_output_lines.append("")
    
    formatted_output = '\n'.join(formatted_output_lines)
    
    # Build separate sections for compatibility
    # NOTE: normalized_source is empty - Call 1 outputs ONLY English
    normalized_source = '\n'.join([f"Sentence {s['sentence_num']}: ({s['language']}) [normalized internally, not stored]" for s in all_sentences])
    english_meaning = '\n'.join([f"Sentence {s['sentence_num']}: {s['english_meaning']}" for s in all_sentences])
    
    # ──────────────────────────────────────────────────────────────────────────
    # FINAL PLACEHOLDER LOGIC: Only insert placeholder if ALL sentences fail
    # ──────────────────────────────────────────────────────────────────────────
    placeholder_string = "意味不明\n（音声が不明瞭で、意味を安全に復元できません）"
    
    # Helper function to check if text is a placeholder
    def is_placeholder(text):
        """Check if text is a placeholder string (any variation)"""
        if not text or not text.strip():
            return False
        text_stripped = text.strip()
        # Check for any placeholder variation (all start with "意味不明")
        return text_stripped.startswith("意味不明")
    
    # Filter out placeholder strings and count actual translations
    translated_sentences = []
    for s in japanese_sentences:
        jp_text = s.get('japanese_translation', '').strip()
        # Skip empty strings and placeholder strings (any variation)
        if jp_text and not is_placeholder(jp_text):
            translated_sentences.append(s)
    
    total_sentences = len(japanese_sentences)
    translated_count = len(translated_sentences)
    # Use placeholder only if: there are sentences AND all failed (no translations)
    placeholder_used = (translated_count == 0 and total_sentences > 0)
    
    # Debug logging
    print("=" * 80, flush=True)
    print("=== FINAL PLACEHOLDER LOGIC ===", flush=True)
    print(f"total_sentences: {total_sentences}", flush=True)
    print(f"translated_count: {translated_count}", flush=True)
    print(f"placeholder_used: {placeholder_used}", flush=True)
    if placeholder_used:
        print("  → All sentences failed - returning single placeholder", flush=True)
    else:
        print(f"  → Returning {translated_count} translated sentence(s) (filtered out empty/placeholder)", flush=True)
    print("=" * 80, flush=True)
    
    # Build final Japanese translation
    if placeholder_used:
        # All sentences failed - return single placeholder
        japanese_translation = placeholder_string
    elif translated_count > 0:
        # At least one sentence translated - return only translated sentences (filter out empty/placeholder)
        japanese_translation = '\n'.join([f"Sentence {s['sentence_num']}: {s.get('japanese_translation', '')}" for s in translated_sentences])
    else:
        # No sentences at all (edge case)
        japanese_translation = ""
    
    # Extract cultural terms (not used in new format, but kept for backward compatibility)
    cultural_terms = []
    
    print(f"Final output: {len(all_sentences)} sentences processed ({len(blocked_sentences)} blocked, {len([s for s in all_sentences if s.get('blocked_reason') is None])} from Call 1)", flush=True)
    
    return {
        'formatted_output': formatted_output,
        'detected_languages': list(set([s['language'] for s in all_sentences])),
        'normalized_text': normalized_source,
        'english_meaning': english_meaning,  # Store for reuse
        'japanese_translation': japanese_translation,
        'cultural_terms': cultural_terms
    }
