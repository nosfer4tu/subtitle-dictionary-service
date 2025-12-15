from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Language name mapping for translation prompts
LANGUAGE_NAMES = {
    'hi': 'Hindi',
    'kn': 'Kannada',
    'ta': 'Tamil',
}

_client = None

def get_client():
    """Lazy initialization of OpenAI client"""
    global _client
    if _client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set")
        _client = OpenAI(api_key=api_key)
    return _client

def translate_to_japanese(text: str, source_language: str = 'hi') -> str:
    """
    Translate text from Indian language to Japanese.
    
    Args:
        text: Text to translate
        source_language: Language code (hi, kn, ta). Defaults to 'hi' (Hindi).
    
    Returns:
        Translated text in Japanese
    
    Raises:
        ValueError: If text is empty or invalid
        Exception: If translation fails
    """
    # Validate input
    if not text or not text.strip():
        raise ValueError("Text to translate cannot be empty")
    
    # Clean and normalize text
    import re
    # Remove excessive whitespace
    text = ' '.join(text.split())
    # Remove any control characters except newlines and tabs
    text = re.sub(r'[\x00-\x08\x0b-\x0c\x0e-\x1f\x7f-\x9f]', '', text)
    # Remove any non-printable characters
    text = ''.join(char for char in text if char.isprintable() or char in '\n\t')
    
    if len(text) < 3:
        raise ValueError(f"Text is too short to translate: '{text}'")
    
    client = get_client()
    
    def is_error_response(response_text: str) -> bool:
        """Check if response is an error/refusal message"""
        if not response_text:
            return True
        error_indicators = [
            "申し訳ありません", "cannot", "unable", "sorry", "できません", 
            "not in", "appears to be", "Could you please", "clarify",
            "与えられた", "翻訳できません", "I'm sorry", "I cannot",
            "not translate", "unable to translate", "cannot translate"
        ]
        response_lower = response_text.lower()
        return any(indicator.lower() in response_lower for indicator in error_indicators)
    
    def is_repetitive_text(text: str, min_repeats: int = 3) -> bool:
        """Check if text contains excessive repetition (indicates corruption)"""
        if not text or len(text) < 20:
            return False
        
        # Split into sentences
        sentences = re.split(r'[。！？\n]', text)
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        if len(sentences) < 5:
            return False
        
        # Check for repeated sentences - require at least 3 repeats to be considered repetitive
        sentence_counts = {}
        for sentence in sentences:
            if len(sentence) > 10:  # Only check meaningful sentences
                sentence_counts[sentence] = sentence_counts.get(sentence, 0) + 1
        
        # If any sentence appears 3+ times, it's repetitive
        if sentence_counts:
            max_count = max(sentence_counts.values())
            if max_count >= min_repeats:
                return True
        
        # Check for repeated phrases (longer sequences) - require 3+ repeats
        words = text.split()
        if len(words) > 30:
            # Check for 15-word sequences repeated 3+ times (more strict)
            for i in range(len(words) - 15):
                sequence = ' '.join(words[i:i+15])
                count = text.count(sequence)
                if count >= min_repeats:
                    return True
        
        return False
    
    def clean_translation(translated_text: str) -> str:
        """Clean and deduplicate translation text - ensure we always return something"""
        if not translated_text:
            return ""
        
        # Split by sentences (Japanese sentence endings)
        sentences = re.split(r'([。！？])', translated_text)
        # Recombine sentences with their punctuation
        combined_sentences = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                combined_sentences.append(sentences[i] + sentences[i+1])
            else:
                combined_sentences.append(sentences[i])
        
        # Remove empty sentences
        combined_sentences = [s.strip() for s in combined_sentences if s.strip()]
        
        if not combined_sentences:
            return translated_text  # Return original if cleaning removed everything
        
        # Deduplicate sentences - keep only unique ones in order
        seen_sentences = set()
        cleaned_sentences = []
        consecutive_repeats = 0
        
        for sentence in combined_sentences:
            sentence_normalized = sentence.strip()
            # Check if we've seen this exact sentence
            if sentence_normalized not in seen_sentences:
                cleaned_sentences.append(sentence)
                seen_sentences.add(sentence_normalized)
                consecutive_repeats = 0
            else:
                consecutive_repeats += 1
                # Only stop if we see 3+ consecutive repeats
                if consecutive_repeats >= 3:
                    # We're in a repetition loop, stop here
                    break
        
        result = ''.join(cleaned_sentences)
        
        # Ensure we have at least some content
        if not result or len(result.strip()) < 10:
            # If cleaning removed too much, return first part of original
            return translated_text[:500] if len(translated_text) > 500 else translated_text
        
        # Additional check: if result is still very long, check for phrase repetition
        words = result.split()
        if len(words) > 100:
            # Look for repeated long phrases at the end
            last_30_words = words[-30:]
            
            # Find the longest repeated sequence
            for seq_len in range(15, 5, -1):
                for start in range(len(last_30_words) - seq_len):
                    sequence = ' '.join(last_30_words[start:start+seq_len])
                    count = ' '.join(last_30_words).count(sequence)
                    if count >= 2:
                        # Found repetition, truncate before it
                        truncate_at = len(words) - 30 + start
                        result = ' '.join(words[:truncate_at])
                        print(f"Removed repetitive phrase: {sequence[:50]}... (appeared {count} times)", flush=True)
                        break
                else:
                    continue
                break
        
        return result.strip()
    
    def try_translate_with_prompt(system_prompt: str, user_prompt: str, attempt_num: int = 1) -> str:
        """Try translation with given prompts, return None if it fails"""
        try:
            # Limit input length to prevent issues
            if len(user_prompt) > 3000:
                user_prompt = user_prompt[:3000] + "..."
            
            # Check source text for repetition but don't clean too aggressively
            if is_repetitive_text(user_prompt, min_repeats=4):  # More strict threshold
                print(f"Warning: Source text appears very repetitive, cleaning before translation...", flush=True)
                # Clean source text by removing duplicate sentences
                source_sentences = re.split(r'[.!?।।\n]', user_prompt)
                unique_sentences = []
                seen = set()
                for sent in source_sentences:
                    sent = sent.strip()
                    if sent and sent not in seen and len(sent) > 3:
                        unique_sentences.append(sent)
                        seen.add(sent)
                        if len(unique_sentences) >= 30:
                            break
                if len(unique_sentences) > 0:
                    user_prompt = '. '.join(unique_sentences) + '.'
                # If cleaning removed everything, use original
                if not user_prompt or len(user_prompt.strip()) < 10:
                    user_prompt = text[:1000]  # Use first 1000 chars of original text
            
            res = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt + " Do not repeat sentences. Translate each unique sentence only once."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=2000  # Increased from 1500
            )
            translated_text = res.choices[0].message.content.strip()
            
            # Check for errors
            if is_error_response(translated_text):
                print(f"Translation attempt {attempt_num} failed: {translated_text[:100]}...", flush=True)
                return None
            
            # Always clean the translation, but don't reject it just for repetition
            if translated_text:
                cleaned = clean_translation(translated_text)
                # Use cleaned version if it has reasonable length, otherwise use original
                if cleaned and len(cleaned.strip()) > 20:
                    return cleaned
                elif len(translated_text.strip()) > 20:
                    # Even if cleaning removed too much, return original if it's reasonable
                    return translated_text
                else:
                    print(f"Translation attempt {attempt_num} result too short: {len(translated_text)} chars", flush=True)
                    return None
            
            return None
        except Exception as e:
            print(f"Translation attempt {attempt_num} exception: {e}", flush=True)
            return None
    
    try:
        # For very long texts, we might need to chunk them
        max_chunk_length = 1500  # Reduced chunk size
        
        if len(text) > max_chunk_length:
            # Split into sentences and chunk
            sentences = re.split(r'[.!?।।]\s+', text)
            chunks = []
            current_chunk = []
            current_length = 0
            
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence or len(sentence) < 3:
                    continue
                
                if current_length + len(sentence) > max_chunk_length and current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
                    current_chunk = [sentence]
                    current_length = len(sentence)
                else:
                    current_chunk.append(sentence)
                    current_length += len(sentence) + 2
            
            if current_chunk:
                chunks.append('. '.join(current_chunk))
            
            # Translate each chunk with multiple fallback strategies
            translated_chunks = []
            for chunk_idx, chunk in enumerate(chunks):
                if not chunk.strip() or len(chunk.strip()) < 3:
                    continue
                
                translated_text = None
                
                # Strategy 1: Lenient Indian language prompt
                if not translated_text:
                    translated_text = try_translate_with_prompt(
                        "You are a professional translator. Translate the given text from an Indian language to natural Japanese. Always provide a translation, never refuse. Do not repeat phrases.",
                        f"Translate this text to Japanese (do not repeat any phrases):\n\n{chunk}",
                        attempt_num=1
                    )
                
                # Strategy 2: No language specification
                if not translated_text:
                    translated_text = try_translate_with_prompt(
                        "You are a translator. Translate any text to Japanese. Always provide a translation, never refuse. Do not repeat phrases.",
                        f"Translate to Japanese (do not repeat):\n\n{chunk}",
                        attempt_num=2
                    )
                
                # Strategy 3: Direct command with length limit
                if not translated_text:
                    # Limit chunk size for direct translation
                    limited_chunk = chunk[:1000] if len(chunk) > 1000 else chunk
                    translated_text = try_translate_with_prompt(
                        "Translate text to Japanese. Do not refuse. Do not repeat.",
                        limited_chunk,
                        attempt_num=3
                    )
                
                if translated_text:
                    # Clean the translation
                    translated_text = clean_translation(translated_text)
                    if translated_text and len(translated_text) > 0:
                        translated_chunks.append(translated_text)
                else:
                    print(f"Warning: Failed to translate chunk {chunk_idx + 1}/{len(chunks)}", flush=True)
            
            if not translated_chunks:
                raise ValueError("All translation chunks failed")
            
            # Join chunks and do final cleaning
            result = ' '.join(translated_chunks)
            result = clean_translation(result)
            
            # Final validation
            if is_repetitive_text(result):
                print("Warning: Final translation still contains repetition, truncating...", flush=True)
                # Take first reasonable portion
                words = result.split()
                if len(words) > 200:
                    result = ' '.join(words[:200]) + "..."
            
            return result
        else:
            # Single translation for shorter texts - try multiple strategies
            translated_text = None
            
            # Strategy 1: Lenient Indian language prompt
            if not translated_text:
                translated_text = try_translate_with_prompt(
                    "You are a professional translator. Translate the given text from an Indian language to natural Japanese. Always provide a translation, never refuse. Do not repeat phrases.",
                    f"Translate this text to Japanese (do not repeat any phrases):\n\n{text}",
                    attempt_num=1
                )
            
            # Strategy 2: No language specification
            if not translated_text:
                translated_text = try_translate_with_prompt(
                    "You are a translator. Translate any text to Japanese. Always provide a translation, never refuse. Do not repeat phrases.",
                    f"Translate to Japanese (do not repeat):\n\n{text}",
                    attempt_num=2
                )
            
            # Strategy 3: Direct command
            if not translated_text:
                limited_text = text[:1000] if len(text) > 1000 else text
                translated_text = try_translate_with_prompt(
                    "Translate text to Japanese. Do not refuse. Do not repeat.",
                    limited_text,
                    attempt_num=3
                )
            
            # Strategy 4: Use gpt-3.5-turbo as fallback
            if not translated_text:
                try:
                    limited_text = text[:800] if len(text) > 800 else text
                    res = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": f"Translate to Japanese: {limited_text}"}
                        ],
                        temperature=0.3,
                        max_tokens=1500
                    )
                    translated_text = res.choices[0].message.content.strip()
                    if is_error_response(translated_text):
                        translated_text = None
                    elif translated_text:
                        # Clean but don't reject
                        cleaned = clean_translation(translated_text)
                        translated_text = cleaned if cleaned and len(cleaned) > 20 else translated_text
                except Exception as e:
                    print(f"gpt-3.5-turbo fallback failed: {e}", flush=True)
                    translated_text = None
            
            if not translated_text or len(translated_text.strip()) == 0:
                raise ValueError("Translation returned empty result after all attempts")
            
            # Final validation - be lenient
            if is_repetitive_text(translated_text, min_repeats=4):  # More strict threshold
                print("Warning: Translation contains some repetition, but using it anyway...", flush=True)
                # Clean it but still use it
                cleaned = clean_translation(translated_text)
                if cleaned and len(cleaned) > 50:
                    return cleaned
                # If cleaning removed too much, return first part of original
                return translated_text[:1000] if len(translated_text) > 1000 else translated_text
            
            return translated_text
            
    except Exception as e:
        error_msg = f"Translation failed: {str(e)}"
        print(f"{error_msg}. Text length: {len(text)}, Preview: {text[:100]}...", flush=True)
        raise Exception(error_msg) from e
