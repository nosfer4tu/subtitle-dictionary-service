from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import json

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)
load_dotenv()

# Language name mapping
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

def extract_json_from_text(text: str) -> str:
    """Extract JSON from text that might contain markdown or other formatting"""
    # Try to find JSON object
    # Look for opening brace
    first_brace = text.find('{')
    if first_brace == -1:
        return text
    
    # Find matching closing brace
    brace_count = 0
    last_brace = first_brace
    for i in range(first_brace, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                last_brace = i
                break
    
    if brace_count == 0:
        return text[first_brace:last_brace + 1]
    
    return text[first_brace:]

def detect_terms(text: str, source_language: str = 'hi') -> dict:
    """ 
    Sends text to ChatGPT and extracts:
    - cultural terms / unique Indian words
    - meaning in Japanese
    
    Args:
        text: Source text in Indian language
        source_language: Language code (hi, kn, ta)
    
    Returns:
        Dictionary with 'terms' list, or empty dict if detection fails
    """
    if not text or not text.strip():
        print("detect_terms: Empty text provided", flush=True)
        return {"terms": []}
    
    # Clean text
    text = ' '.join(text.split())
    
    # Check for excessive repetition
    words = text.split()
    if len(words) > 50:
        word_counts = {}
        for word in words:
            if len(word) > 3:
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            max_count = max(word_counts.values())
            if max_count > len(words) * 0.3:
                print(f"Warning: Source text appears corrupted (repetitive), using first portion only", flush=True)
                text = ' '.join(words[:100])
    
    # Limit text length
    if len(text) > 3000:
        text = text[:2000] + " ... " + text[-1000:]
        print(f"Warning: Text too long for term detection, truncating", flush=True)
    
    language_name = LANGUAGE_NAMES.get(source_language, 'Hindi')
    client = get_client()
    
    # Improved prompts with better accuracy requirements
    prompts = [
        # Strategy 1: Detailed and specific prompt
        f"""Analyze this {language_name} text and extract ONLY specific cultural or India-specific terms.

A "cultural term" is a SPECIFIC word or phrase that:
- Refers to a specific Indian concept, item, or practice (e.g., "Diwali", "Sari", "Guru", "Namaste")
- Is a proper noun for Indian festivals, foods, deities, places, or cultural practices
- Is slang or expression unique to Indian culture
- Is a traditional title, honorific, or role specific to India

DO NOT include:
- Generic words that could apply to any culture (like "god", "festival", "food")
- Common words that exist in many languages
- Abstract concepts without specific cultural context

For each SPECIFIC cultural term found, provide:
1. word: the exact word/phrase as it appears in the text
2. pronunciation_japanese: how to pronounce it in Japanese using katakana (e.g., "イシュワラデーヴァ" for "Ishwaradeva")
3. meaning_japanese: what this SPECIFIC term means in Japanese (not a generic translation)
4. why_important: why THIS SPECIFIC term is culturally important in India

Return ONLY valid JSON:
{{"terms": [{{"word": "", "pronunciation_japanese": "", "meaning_japanese": "", "why_important": ""}}]}}

Text to analyze:
{text}""",
        
        # Strategy 2: More focused prompt
        f"""Extract SPECIFIC cultural terms from this {language_name} text.

Focus on:
- Specific proper nouns (names of festivals, foods, deities, places)
- Unique Indian cultural concepts
- Traditional titles or roles
- Cultural expressions or idioms

Avoid generic words. Each term should be specific to Indian culture.

For each term, return JSON with:
- word: exact term from text
- pronunciation_japanese: katakana pronunciation
- meaning_japanese: specific meaning in Japanese
- why_important: cultural significance

Text: {text}""",
        
        # Strategy 3: Simple but specific
        f"""Find SPECIFIC Indian cultural terms in: {text}

Only extract terms that are uniquely Indian (festivals, foods, deities, cultural concepts).
Return JSON: {{"terms": [{{"word": "...", "pronunciation_japanese": "...", "meaning_japanese": "...", "why_important": "..."}}]}}"""
    ]
    
    for attempt, prompt in enumerate(prompts, 1):
        try:
            print(f"detect_terms attempt {attempt}/3 for {language_name} text (length: {len(text)})", flush=True)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert in Indian culture. Extract ONLY SPECIFIC cultural terms (proper nouns, unique concepts). Avoid generic words. Always return valid JSON with pronunciation_japanese in katakana."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more consistent, accurate results
                max_tokens=3000
            )
            json_output = response.choices[0].message.content.strip()
            
            print(f"Raw response (first 800 chars): {json_output[:800]}...", flush=True)
            
            # Extract JSON
            json_text = extract_json_from_text(json_output)
            
            # Try to parse
            try:
                result = json.loads(json_text)
            except json.JSONDecodeError as e:
                print(f"JSON parse error on attempt {attempt}: {e}", flush=True)
                # Try to fix common issues
                json_text = re.sub(r',\s*}', '}', json_text)
                json_text = re.sub(r',\s*]', ']', json_text)
                try:
                    result = json.loads(json_text)
                except:
                    if attempt < len(prompts):
                        continue
                    raise
            
            # Validate structure
            if not isinstance(result, dict):
                print(f"Result is not a dict: {type(result)}", flush=True)
                if attempt < len(prompts):
                    continue
                return {"terms": []}
            
            if 'terms' not in result:
                print(f"Missing 'terms' key. Keys: {list(result.keys())}", flush=True)
                if attempt < len(prompts):
                    continue
                return {"terms": []}
            
            if not isinstance(result['terms'], list):
                print(f"Terms is not a list: {type(result['terms'])}", flush=True)
                if attempt < len(prompts):
                    continue
                return {"terms": []}
            
            # Process terms - validate and filter
            valid_terms = []
            for term in result['terms']:
                if isinstance(term, dict):
                    word = term.get('word', '').strip()
                    pronunciation = term.get('pronunciation_japanese', term.get('pronunciation', '')).strip()
                    meaning = term.get('meaning_japanese', term.get('meaning', '')).strip()
                    why_important = term.get('why_important', term.get('why_important', '')).strip()
                    
                    # Only require word - be lenient about other fields
                    if word and len(word) > 0:
                        # Ensure pronunciation is in katakana or provide fallback
                        if not pronunciation:
                            pronunciation = word  # Fallback
                        elif not any(ord(char) >= 0x30A0 and ord(char) <= 0x30FF for char in pronunciation):
                            # If not katakana, use word as fallback
                            pronunciation = word
                        
                        valid_terms.append({
                            'word': word,
                            'pronunciation_japanese': pronunciation,
                            'meaning_japanese': meaning if meaning else "文化的用語",
                            'why_important': why_important if why_important else "インド文化において重要な用語"
                        })
            
            # Filter out generic terms (optional - can be removed if too strict)
            # This helps avoid terms like "神々" which are too generic
            filtered_terms = []
            generic_indicators = ['神々', '神', '神様', '一般的', 'generic']
            for term in valid_terms:
                meaning = term.get('meaning_japanese', '').lower()
                # Skip if meaning is too generic
                if not any(indicator in meaning for indicator in generic_indicators):
                    filtered_terms.append(term)
                elif len(valid_terms) <= 3:  # Keep if we have very few terms
                    filtered_terms.append(term)
            
            if len(filtered_terms) > 0:
                result['terms'] = filtered_terms
                print(f"✓ Successfully detected {len(filtered_terms)} cultural terms on attempt {attempt}", flush=True)
                for i, term in enumerate(filtered_terms[:3], 1):
                    print(f"  Term {i}: '{term['word']}' -> {term['pronunciation_japanese']} ({term['meaning_japanese'][:40]}...)", flush=True)
                return result
            elif len(valid_terms) > 0:
                # Use valid_terms if filtering removed everything
                result['terms'] = valid_terms
                print(f"✓ Detected {len(valid_terms)} terms (filtering removed some generic ones)", flush=True)
                return result
            else:
                print(f"No valid terms found on attempt {attempt}, trying next strategy...", flush=True)
                if attempt < len(prompts):
                    continue
        
        except Exception as e:
            import traceback
            print(f"Error on attempt {attempt}: {e}", flush=True)
            if attempt < len(prompts):
                continue
            print(f"Traceback: {traceback.format_exc()}", flush=True)
    
    # All attempts failed
    print("All detect_terms attempts failed, returning empty list", flush=True)
    return {"terms": []}