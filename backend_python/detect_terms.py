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
    Detect cultural terms in text.
    Preserves proper nouns and religious/folklore terms in transliteration.
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
    
    # Update the prompt to emphasize preserving proper nouns
    prompt = f"""
    Analyze the following {language_name} text and extract cultural or India-specific terms.
    
    IMPORTANT RULES:
    1. Do NOT directly translate religious or folklore terms (e.g., Ishvara, Dharma, Brahmarakshasa)
    2. Keep proper nouns in transliteration (e.g., Ishvara, Dharma, Brahmarakshasa)
    3. Only translate terms that have clear Japanese equivalents
    4. For religious/folklore terms, provide explanation in Japanese but keep the term in transliteration
    
    Return format (JSON only):
    {{
        "terms": [
            {{
                "word": "exact word from text",
                "pronunciation_japanese": "カタカナでの発音",
                "meaning_japanese": "日本語での意味",
                "why_important": "文化的な重要性の説明"
            }}
        ]
    }}
    
    Text to analyze:
    {text}
    """
    
    # Retry loop with up to 3 attempts
    for attempt in range(1, 4):
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
                    if attempt < 3:
                        continue
                    raise
            
            # Validate structure
            if not isinstance(result, dict):
                print(f"Result is not a dict: {type(result)}", flush=True)
                if attempt < 3:
                    continue
                return {"terms": []}
            
            if 'terms' not in result:
                print(f"Missing 'terms' key. Keys: {list(result.keys())}", flush=True)
                if attempt < 3:
                    continue
                return {"terms": []}
            
            if not isinstance(result['terms'], list):
                print(f"Terms is not a list: {type(result['terms'])}", flush=True)
                if attempt < 3:
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
            
            # Filter out generic terms, hallucinated terms, and plain English common nouns
            # This helps avoid terms like "神々" (too generic), generic food words like "Papaya",
            # and cultural terms that do NOT actually appear in the source text (hallucinations).
            filtered_terms = []
            generic_indicators = ['神々', '神', '神様', '一般的', 'generic']
            # Markers that indicate a true cultural/proper-noun concept
            cultural_markers = [
                '人名', '人物', '固有名詞',
                '神話', '女神', '神', '神格', '宗教',
                '祭り', '祝祭',
                '都市', '街', '村', '州', '地方', '地域', '聖地', '寺院',
                '王国', '王朝', '王',
                '伝説', '英雄', '叙事詩'
            ]
            source_text_lower = text.lower()
            
            for term in valid_terms:
                meaning = term.get('meaning_japanese', '')
                why_important = term.get('why_important', '')
                combined_explanation = f"{meaning} {why_important}"
                
                # 1. Filter out overly generic meanings
                meaning_lower = meaning.lower()
                if any(indicator in meaning_lower for indicator in generic_indicators):
                    # Keep if we have very few terms (for robustness), otherwise skip
                    if len(valid_terms) > 3:
                        continue
                
                word = term.get('word', '')
                
                # 2. Drop hallucinated terms whose surface word never appears in the source text
                #    (case-insensitive, simple substring check).
                if word:
                    word_lower = word.lower()
                    if word_lower not in source_text_lower:
                        # If the word itself never appears in the transcript, treat as hallucination.
                        # This will drop things like "Ishvara", "Dharma", "Brahmarakshasa" if they
                        # are invented by the model instead of being in the actual text.
                        continue
                
                # 3. Reject plain English common nouns unless clearly cultural/proper nouns
                is_ascii = all(ord(c) < 128 for c in word) if word else False
                if is_ascii:
                    # Explicitly block known generic English terms regardless of explanation
                    blocked_english_terms = {
                        'papaya',
                        'sesame seed',
                        'sesame seeds',
                        'non-vegetarian food',
                        'non vegetarian food',
                        'non-veg food',
                        'non veg food',
                    }
                    if word.lower() in blocked_english_terms:
                        continue
                    
                    # Allow only if explanation clearly marks it as proper noun / unique concept
                    if not any(marker in combined_explanation for marker in cultural_markers):
                        # This will drop things like "Papaya", "Sesame seed", "Non-vegetarian food"
                        # whose explanations are generic (common food, common category, etc.)
                        continue
                
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
                if attempt < 3:
                    continue
        
        except Exception as e:
            import traceback
            print(f"Error on attempt {attempt}: {e}", flush=True)
            if attempt < 3:
                continue
            print(f"Traceback: {traceback.format_exc()}", flush=True)
    
    # All attempts failed
    print("All detect_terms attempts failed, returning empty list", flush=True)
    return {"terms": []}