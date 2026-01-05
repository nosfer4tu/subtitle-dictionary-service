import os
import subprocess
import tempfile
from pydub import AudioSegment
from backend_python.whisper_transcribe import transcribe_audio, transcribe_audio_with_timestamps
from backend_python.detect_terms import detect_terms
import re
import unicodedata

def verify_video_file(video_path: str) -> bool:
    """Verify that a video file is valid and not corrupted"""
    import subprocess
    try:
        # Use ffprobe to check if the file is valid
        result = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10
        )
        if result.returncode == 0 and result.stdout:
            duration = result.stdout.decode('utf-8').strip()
            if duration and float(duration) > 0:
                return True
        return False
    except Exception:
        return False

def verify_and_repair_video(input_path: str, output_path: str) -> str:
    """Verify video file and remux it to ensure it's a valid MP4 with proper moov atom"""
    import subprocess
    
    try:
        # Try to remux the video to ensure it has proper MP4 structure
        # This fixes "moov atom not found" errors
        result = subprocess.run(
            [
                'ffmpeg',
                '-i', input_path,
                '-c', 'copy',  # Copy streams without re-encoding (fast)
                '-movflags', '+faststart',  # Move moov atom to beginning (web-friendly)
                '-y',  # Overwrite
                output_path
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60
        )
        
        if result.returncode == 0 and os.path.exists(output_path):
            if os.path.getsize(output_path) > 0:
                return output_path
        
        # If remuxing failed, return original path
        return input_path
    except Exception as e:
        print(f"Video repair failed: {e}", flush=True)
        return input_path

def extract_audio_from_video(video_path: str, output_audio_path: str = None) -> str:
    """Extract audio from video file using ffmpeg"""
    if output_audio_path is None:
        output_audio_path = video_path.rsplit('.', 1)[0] + '.mp3'
    
    # Check if input file exists
    if not os.path.exists(video_path):
        raise Exception(f"Video file not found: {video_path}")
    
    # Verify file is not empty
    file_size = os.path.getsize(video_path)
    if file_size == 0:
        raise Exception(f"Video file is empty: {video_path}")
    
    # Verify video file is valid (optional check, might be slow)
    # Uncomment if you want to verify before processing
    # if not verify_video_file(video_path):
    #     raise Exception(f"Video file appears to be corrupted or invalid: {video_path}")
    
    try:
        result = subprocess.run(
            [
                'ffmpeg',
                '-i', video_path,
                '-vn',  # No video
                '-c:a', 'libmp3lame',  # Use modern codec syntax
                '-b:a', '192k',  # Audio bitrate
                '-ar', '44100',  # Sample rate
                '-ac', '2',  # Stereo
                '-y',  # Overwrite output file
                '-loglevel', 'error',  # Only show errors
                output_audio_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=300  # 5 minute timeout
        )
        
        # Verify output file was created
        if not os.path.exists(output_audio_path):
            raise Exception(f"Audio extraction completed but output file not found: {output_audio_path}")
        
        # Verify output file is not empty
        if os.path.getsize(output_audio_path) == 0:
            raise Exception(f"Audio extraction completed but output file is empty: {output_audio_path}")
        
        return output_audio_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        raise Exception(f"Failed to extract audio: {error_msg}")
    except FileNotFoundError:
        raise Exception("ffmpeg not found. Please install ffmpeg.")

def transcribe_with_timestamps(audio_path: str, language: str = None) -> list:
    """
    Transcribe audio and return segments with timestamps and detected language.
    Uses Whisper's automatic language detection per segment.
    
    Args:
        audio_path: Path to audio file
        language: Language hint (optional, Whisper will auto-detect per segment)
    
    Returns:
        List of segments with 'text', 'start', 'end', 'detected_language', 'script', 'confidence'
    """
    try:
        # Use Whisper with auto-detection (don't pass language parameter)
        result = transcribe_audio_with_timestamps(audio_path, language=None)
        
        # Process segments
        if hasattr(result, 'segments') and result.segments:
            segments = []
            from backend_python.translate import detect_script_from_text, LANGUAGE_NAMES
            
            for seg in result.segments:
                # Handle both dict and object formats
                if isinstance(seg, dict):
                    text = seg.get('text', '').strip()
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                    detected_lang = seg.get('detected_language') or seg.get('language')
                    # Get no_speech_prob if available (confidence indicator)
                    no_speech_prob = seg.get('no_speech_prob', 0.0)
                else:
                    text = getattr(seg, 'text', '').strip()
                    start = getattr(seg, 'start', 0)
                    end = getattr(seg, 'end', 0)
                    detected_lang = getattr(seg, 'detected_language', None) or getattr(seg, 'language', None)
                    no_speech_prob = getattr(seg, 'no_speech_prob', 0.0)
                
                if not text:
                    continue
                
                # Detect script (separate from language)
                script = detect_script_from_text(text)
                
                # Map Whisper language codes to our codes
                if detected_lang:
                    lang_map = {'hi': 'hi', 'kn': 'kn', 'ta': 'ta', 'te': 'kn', 'mr': 'hi'}
                    detected_lang = lang_map.get(detected_lang, detected_lang)
                else:
                    # Fallback: use passed language or default
                    detected_lang = language or 'hi'
                
                # Calculate confidence (1 - no_speech_prob, higher is better)
                confidence = 1.0 - no_speech_prob if no_speech_prob else 0.8
                
                # Reject low-confidence segments
                CONFIDENCE_THRESHOLD = 0.5
                if confidence < CONFIDENCE_THRESHOLD:
                    print(f"Warning: Low confidence segment ({confidence:.2f}) - marking for review: {text[:50]}...", flush=True)
                    needs_review = True
                else:
                    needs_review = False
                
                segments.append({
                    'text': text,
                    'start': start,
                    'end': end,
                    'detected_language': detected_lang,
                    'script': script,
                    'confidence': confidence,
                    'needs_review': needs_review
                })
            
            # AGGRESSIVE SEGMENTATION: Split segments longer than 6 seconds
            # Target: 3-6 seconds max per segment
            split_segments = []
            MAX_SEGMENT_DURATION = 6.0  # 6 seconds max
            TARGET_SEGMENT_DURATION = 4.5  # Target ~4.5 seconds
            
            for seg in segments:
                duration = seg['end'] - seg['start']
                
                if duration <= MAX_SEGMENT_DURATION:
                    # Segment is acceptable length
                    split_segments.append(seg)
                else:
                    # Split long segment into smaller chunks
                    num_splits = int(duration / TARGET_SEGMENT_DURATION) + 1
                    split_duration = duration / num_splits
                    
                    # Split text by sentences or punctuation
                    import re
                    text = seg['text']
                    # Try to split on sentence boundaries
                    sentences = re.split(r'([.!?।।]\s+)', text)
                    
                    # If we have sentence boundaries, split evenly across them
                    if len(sentences) > 1:
                        sentences_per_chunk = max(1, len(sentences) // num_splits)
                        current_chunk_text = []
                        current_chunk_start = seg['start']
                        chunk_idx = 0
                        
                        for i, sentence in enumerate(sentences):
                            current_chunk_text.append(sentence)
                            
                            # Create chunk when we have enough sentences or reach end
                            if (i + 1) % sentences_per_chunk == 0 or i == len(sentences) - 1:
                                chunk_text = ''.join(current_chunk_text).strip()
                                if chunk_text:
                                    chunk_end = seg['start'] + (chunk_idx + 1) * split_duration
                                    if chunk_end > seg['end']:
                                        chunk_end = seg['end']
                                    
                                    split_segments.append({
                                        'text': chunk_text,
                                        'start': current_chunk_start,
                                        'end': chunk_end,
                                        'detected_language': seg['detected_language'],
                                        'script': seg['script'],
                                        'confidence': seg['confidence'],
                                        'needs_review': seg['needs_review']
                                    })
                                    
                                    current_chunk_start = chunk_end
                                    current_chunk_text = []
                                    chunk_idx += 1
                    else:
                        # No sentence boundaries - split by character count
                        chars_per_chunk = len(text) // num_splits
                        for i in range(num_splits):
                            chunk_start_idx = i * chars_per_chunk
                            chunk_end_idx = (i + 1) * chars_per_chunk if i < num_splits - 1 else len(text)
                            chunk_text = text[chunk_start_idx:chunk_end_idx].strip()
                            
                            if chunk_text:
                                chunk_start = seg['start'] + i * split_duration
                                chunk_end = seg['start'] + (i + 1) * split_duration
                                if chunk_end > seg['end']:
                                    chunk_end = seg['end']
                                
                                split_segments.append({
                                    'text': chunk_text,
                                    'start': chunk_start,
                                    'end': chunk_end,
                                    'detected_language': seg['detected_language'],
                                    'script': seg['script'],
                                    'confidence': seg['confidence'],
                                    'needs_review': seg['needs_review']
                                })
            
            segments = split_segments
            
            if segments:
                print(f"Transcribed {len(segments)} segments with language detection (after aggressive segmentation)", flush=True)
                # Log language distribution
                lang_counts = {}
                for seg in segments:
                    lang = seg['detected_language']
                    lang_counts[lang] = lang_counts.get(lang, 0) + 1
                print(f"Language distribution: {lang_counts}", flush=True)
                return segments
        
        # Fallback: use text transcription and estimate timestamps
        if hasattr(result, 'text'):
            transcription = result.text
        else:
            transcription = transcribe_audio(audio_path, language=None)
        
    except Exception as e:
        print(f"Error getting timestamps, using fallback: {e}", flush=True)
        transcription = transcribe_audio(audio_path, language=None)
    
    # Fallback: Split transcription into sentences and estimate timestamps
    sentence_delimiters = r'[.!?।।]\s+'
    sentences = re.split(sentence_delimiters, transcription)
    
    # Estimate timestamps (simple approach)
    estimated_duration = 30  # Assume 30 seconds if unknown
    time_per_char = estimated_duration / len(transcription) if transcription else 0.1
    
    segments = []
    current_time = 0.0
    from backend_python.translate import detect_script_from_text
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence or len(sentence) < 3:
            continue
        
        sentence_duration = len(sentence) * time_per_char
        script = detect_script_from_text(sentence)
        
        segments.append({
            'text': sentence,
            'start': current_time,
            'end': current_time + sentence_duration,
            'detected_language': language or 'hi',  # Fallback
            'script': script,
            'confidence': 0.7,  # Lower confidence for fallback
            'needs_review': True
        })
        current_time += sentence_duration + 0.5  # 0.5 second gap
    
    return segments

def create_srt_file(segments: list, output_path: str):
    """Create SRT subtitle file from segments"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, segment in enumerate(segments, 1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            text = segment['text']
            
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def process_video_with_subtitles(
    video_path: str,
    output_path: str,
    source_segments: list,
    japanese_translation: str,
    detected_terms: dict = None
) -> str:
    """
    Burn subtitles into video using ffmpeg.
    Creates a video with source language, Japanese translation, and cultural term meanings.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        source_segments: List of segments with timestamps in source language
        japanese_translation: Japanese translation text
        detected_terms: Dictionary with detected cultural terms (optional)
    """
    # Split Japanese translation into segments matching source segments
    japanese_sentences = re.split(r'[。！？]\s*', japanese_translation)
    japanese_sentences = [s.strip() for s in japanese_sentences if s.strip()]
    
    # Create a mapping of cultural terms to their full information
    term_info_map = {}  # word -> {pronunciation, meaning, why_important}
    term_words_lower = {}  # For case-insensitive matching
    
    if detected_terms and isinstance(detected_terms, dict) and 'terms' in detected_terms:
        terms_list = detected_terms.get('terms', [])
        print(f"Processing {len(terms_list)} cultural terms for subtitle matching", flush=True)
        
        for term in terms_list:
            if isinstance(term, dict):
                word = term.get('word', '').strip()
                pronunciation = term.get('pronunciation_japanese', '').strip()
                meaning = term.get('meaning_japanese', '').strip()
                why_important = term.get('why_important', '').strip()
                
                if word and meaning:
                    # Store full term information
                    term_info_map[word] = {
                        'pronunciation': pronunciation or word,
                        'meaning': meaning,
                        'why_important': why_important or "文化的に重要な用語"
                    }
                    term_words_lower[word.lower()] = (word, term_info_map[word])
                    print(f"  - Mapped term: '{word}' -> 発音: {pronunciation}, 意味: {meaning[:30]}...", flush=True)
    
    if not term_info_map:
        print("Warning: No cultural terms to match in subtitles", flush=True)
    else:
        print(f"Term map contains {len(term_info_map)} terms: {list(term_info_map.keys())[:5]}...", flush=True)
    
    # Create temporary SRT file for subtitles
    with tempfile.NamedTemporaryFile(mode='w', suffix='.srt', delete=False, encoding='utf-8') as srt_file:
        # Create bilingual subtitles with cultural term annotations
        for i, segment in enumerate(source_segments, 1):
            start_time = format_timestamp(segment['start'])
            end_time = format_timestamp(segment['end'])
            
            # Match Japanese translation to source segment
            japanese_text = japanese_sentences[i-1] if i-1 < len(japanese_sentences) else ""
            
            # Check if any cultural terms appear in this segment
            source_text = segment['text']
            
            # Sanitize source text to prevent rendering issues
            source_text = sanitize_subtitle_text(source_text)
            
            cultural_annotations = []
            
            if term_info_map and source_text:
                found_terms = set()  # Track which terms we've found to avoid duplicates
                
                # Normalize source text for better matching (remove extra spaces)
                source_normalized = ' '.join(source_text.split())
                
                # Method 1: Direct substring match (no word boundaries - better for Indian scripts)
                for term_word, term_info in term_info_map.items():
                    if term_word not in found_terms:
                        # Simple substring match - Indian scripts don't use word boundaries the same way
                        if term_word in source_normalized:
                            annotation = f"{term_info['pronunciation']} ({term_info['meaning']})"
                            cultural_annotations.append(annotation)
                            found_terms.add(term_word)
                            print(f"  ✓ Found term '{term_word}' in segment {i} (substring match): '{source_text[:60]}...'", flush=True)
                
                # Method 2: Try without spaces (some transcriptions might have different spacing)
                if not found_terms:
                    source_no_spaces = source_normalized.replace(' ', '')
                    for term_word, term_info in term_info_map.items():
                        if term_word not in found_terms:
                            term_no_spaces = term_word.replace(' ', '')
                            if term_no_spaces and term_no_spaces in source_no_spaces:
                                annotation = f"{term_info['pronunciation']} ({term_info['meaning']})"
                                cultural_annotations.append(annotation)
                                found_terms.add(term_word)
                                print(f"  ✓ Found term '{term_word}' in segment {i} (no-space match)", flush=True)
                
                # Method 3: Partial match - check if term is contained in segment or vice versa
                if not found_terms:
                    for term_word, term_info in term_info_map.items():
                        if term_word not in found_terms:
                            # Check if term is a substring of source, or source contains most of term
                            if len(term_word) >= 3:  # Only for meaningful terms
                                # Try matching at least 70% of the term
                                min_match_len = max(3, int(len(term_word) * 0.7))
                                for start_idx in range(len(source_normalized) - min_match_len + 1):
                                    substring = source_normalized[start_idx:start_idx + len(term_word)]
                                    # Check if there's significant overlap
                                    if len(set(term_word) & set(substring)) >= min_match_len:
                                        annotation = f"{term_info['pronunciation']} ({term_info['meaning']})"
                                        cultural_annotations.append(annotation)
                                        found_terms.add(term_word)
                                        print(f"  ✓ Found term '{term_word}' in segment {i} (fuzzy match)", flush=True)
                                        break
                
                # Method 4: Character-based matching for Indian scripts
                # Some transcriptions might have slight character variations
                if not found_terms and len(source_normalized) > 10:
                    for term_word, term_info in term_info_map.items():
                        if term_word not in found_terms and len(term_word) >= 4:
                            # Check if most characters of the term appear in sequence in the source
                            term_chars = list(term_word.replace(' ', ''))
                            source_chars = list(source_normalized.replace(' ', ''))
                            
                            # Find if term characters appear in order in source
                            term_idx = 0
                            for char in source_chars:
                                if term_idx < len(term_chars) and char == term_chars[term_idx]:
                                    term_idx += 1
                                    if term_idx >= len(term_chars) * 0.8:  # 80% match
                                        annotation = f"{term_info['pronunciation']} ({term_info['meaning']})"
                                        cultural_annotations.append(annotation)
                                        found_terms.add(term_word)
                                        print(f"  ✓ Found term '{term_word}' in segment {i} (character sequence match)", flush=True)
                                        break
            
            # Debug: log if we have terms but found none in this segment
            if term_info_map and not cultural_annotations and i <= 5:
                print(f"  Segment {i} text: '{source_text[:80]}...' - No terms matched", flush=True)
                print(f"  Looking for terms: {list(term_info_map.keys())[:3]}...", flush=True)
            
            srt_file.write(f"{i}\n")
            srt_file.write(f"{start_time} --> {end_time}\n")
            
            # Build subtitle text
            subtitle_lines = []
            subtitle_lines.append(source_text)  # Source language (first line)
            
            # Sanitize Japanese text as well
            japanese_text_sanitized = sanitize_subtitle_text(japanese_text)
            subtitle_lines.append(japanese_text_sanitized)  # Japanese translation (second line)
            
            # Add cultural term annotations if any were found
            if cultural_annotations:
                # Join all annotations with " / " separator
                # Format: 発音1 (意味1) / 発音2 (意味2)
                annotations_text = " / ".join(cultural_annotations)
                subtitle_lines.append(f"※ {annotations_text}")  # Cultural meanings (third line with ※ marker)
                print(f"  ✓ Added cultural annotation to segment {i}: {annotations_text[:80]}...", flush=True)
            
            srt_file.write("\n".join(subtitle_lines) + "\n\n")
        
        srt_path = srt_file.name
    
    try:
        # Use ffmpeg to burn subtitles into video
        srt_path_escaped = srt_path.replace('\\', '\\\\').replace(':', '\\:')
        
        # Adjust subtitle styling to accommodate 3 lines if needed
        result = subprocess.run(
            [
                'ffmpeg',
                '-i', video_path,
                '-vf', f"subtitles='{srt_path_escaped}':force_style='FontSize=18,PrimaryColour=&Hffffff,OutlineColour=&H000000,BorderStyle=1,Alignment=2,MarginV=20'",
                '-c:a', 'copy',
                '-y',
                output_path
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        os.unlink(srt_path)
        return output_path
    except subprocess.CalledProcessError as e:
        if os.path.exists(srt_path):
            os.unlink(srt_path)
        error_msg = e.stderr.decode('utf-8') if e.stderr else str(e)
        raise Exception(f"Failed to burn subtitles: {error_msg}")
    except Exception as e:
        if os.path.exists(srt_path):
            os.unlink(srt_path)
        raise
    except FileNotFoundError:
        if os.path.exists(srt_path):
            os.unlink(srt_path)
        raise Exception("ffmpeg not found. Please install ffmpeg.")

def process_trailer_video(video_path: str, output_path: str, language: str = None) -> dict:
    """
    Complete workflow with per-segment language detection and translation.
    
    Args:
        video_path: Path to video file
        output_path: Path for output video
        language: Language hint (optional, Whisper will auto-detect per segment)
    
    Returns dict with transcription, translation, and output video path
    """
    temp_audio = None
    try:
        # Step 1: Extract audio
        temp_audio = video_path.rsplit('.', 1)[0] + '_temp_audio.mp3'
        extract_audio_from_video(video_path, temp_audio)
        
        # Step 2: Transcribe with timestamps and per-segment language detection
        source_segments = transcribe_with_timestamps(temp_audio, language=None)  # Let Whisper auto-detect
        
        if not source_segments:
            raise ValueError("No segments transcribed")
        
        # Step 3: Translate each segment using its detected language
        translated_segments = []
        all_cultural_terms = []
        
        from backend_python.translate import process_transcription_pipeline, LANGUAGE_NAMES
        from backend_python.detect_terms import detect_terms
        
        for seg_idx, segment in enumerate(source_segments):
            seg_text = segment['text']
            seg_lang = segment.get('detected_language', 'hi')
            seg_script = segment.get('script', 'Unknown')
            seg_confidence = segment.get('confidence', 0.5)
            needs_review = segment.get('needs_review', False)
            
            print(f"Segment {seg_idx + 1}/{len(source_segments)}: lang={seg_lang}, script={seg_script}, confidence={seg_confidence:.2f}", flush=True)
            
            # Skip low-confidence segments or mark for review
            if needs_review:
                print(f"  Warning: Low confidence segment, may need review", flush=True)
            
            # === USE MAIN PIPELINE: Process through Call 1 → has_clear_english() → Call 2 ===
            # This is the ONLY allowed translation path - enforces all gates
            try:
                # Process segment through the strict pipeline
                pipeline_result = process_transcription_pipeline(seg_text)
                
                # Extract Japanese translation from pipeline result
                # Format: "Sentence 1: <translation>\nSentence 2: <translation>"
                japanese_translation_text = pipeline_result.get('japanese_translation', '')
                
                # === DEBUG: Pipeline result for this segment ===
                print("=" * 80, flush=True)
                print(f"=== DEBUG: Pipeline result for segment {seg_idx + 1} ===", flush=True)
                print(f"  japanese_translation_text: {repr(japanese_translation_text[:200])}", flush=True)
                print("=" * 80, flush=True)
                
                # Parse the formatted output to extract just the translation text
                # Remove "Sentence X: " prefixes and join lines
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
                    japanese_text = ' '.join(translation_lines) if translation_lines else ""
                else:
                    japanese_text = ""
                
                # If pipeline returned empty (gate blocked or Call 2 failed), leave empty
                # DO NOT replace with placeholder - empty is correct for both cases
                if not japanese_text or not japanese_text.strip():
                    print(f"  Pipeline returned empty for segment {seg_idx + 1} - leaving empty (no placeholder)", flush=True)
                    japanese_text = ""  # Empty string - correct behavior (gate blocked OR Call 2 returned empty)
                
                # Detect cultural terms for this segment (after translation)
                segment_terms = []
                try:
                    terms_result = detect_terms(seg_text, source_language=seg_lang)
                    if terms_result and isinstance(terms_result, dict):
                        terms_list = terms_result.get('terms', [])
                        segment_terms = terms_list
                        all_cultural_terms.extend(terms_list)
                except Exception as e:
                    print(f"  Error detecting terms for segment {seg_idx + 1}: {e}", flush=True)
                
                translated_segments.append({
                    'text_original': seg_text,
                    'language': seg_lang,
                    'script': seg_script,
                    'translation_ja': japanese_text,
                    'cultural_terms': segment_terms,
                    'start': segment['start'],
                    'end': segment['end'],
                    'confidence': seg_confidence,
                    'needs_review': needs_review
                })
                
            except Exception as e:
                print(f"  Error translating segment {seg_idx + 1}: {e}", flush=True)
                # Add segment with empty translation (not error message)
                translated_segments.append({
                    'text_original': seg_text,
                    'language': seg_lang,
                    'script': seg_script,
                    'translation_ja': "",  # Empty, not error message
                    'cultural_terms': [],
                    'start': segment['start'],
                    'end': segment['end'],
                    'confidence': seg_confidence,
                    'needs_review': True
                })
        
        # Combine translations (filter out empty translations)
        source_text = ' '.join([seg['text_original'] for seg in translated_segments])
        japanese_text = ' '.join([seg['translation_ja'] for seg in translated_segments if seg['translation_ja']])
        
        # === DEBUG: Final combined japanese_text before return ===
        print("=" * 80, flush=True)
        print("=== DEBUG: Final combined japanese_text ===", flush=True)
        print(f"  japanese_text length: {len(japanese_text) if japanese_text else 0}", flush=True)
        print(f"  japanese_text preview: {repr(japanese_text[:200]) if japanese_text else 'EMPTY'}", flush=True)
        print(f"  Contains placeholder: {'意味不明' in japanese_text if japanese_text else False}", flush=True)
        print("=" * 80, flush=True)
        
        # Aggregate cultural terms (deduplicate)
        unique_terms = {}
        for term in all_cultural_terms:
            term_word = term.get('word', '')
            if term_word and term_word not in unique_terms:
                unique_terms[term_word] = term
        detected_terms = {"terms": list(unique_terms.values())}
        
        print(f"Translation complete: {len(translated_segments)} segments, {len(detected_terms.get('terms', []))} unique cultural terms", flush=True)
        
        # Step 4 & 5: Burn subtitles into video
        # Convert translated_segments to format expected by process_video_with_subtitles
        subtitle_segments = []
        for seg in translated_segments:
            subtitle_segments.append({
                'text': seg['text_original'],
                'start': seg['start'],
                'end': seg['end']
            })
        
        process_video_with_subtitles(video_path, output_path, subtitle_segments, japanese_text, detected_terms=detected_terms)
        
        # === FINAL SAFEGUARD: Ensure empty strings are NEVER replaced with placeholders ===
        # If japanese_text is empty (all segments filtered out or Call 2 returned empty), leave it empty
        # DO NOT add placeholder - empty is correct for sentences that passed gate but have no translation
        if not japanese_text or not japanese_text.strip():
            print("=" * 80, flush=True)
            print("FINAL SAFEGUARD: japanese_text is empty - leaving empty (NO placeholder)", flush=True)
            print(f"  This is correct for: passed sentences with empty Call 2 output", flush=True)
            print("=" * 80, flush=True)
            japanese_text = ""  # Explicitly ensure it's empty, not a placeholder
        
        # Verify no placeholder was accidentally added
        if "意味不明" in japanese_text and not any(seg.get('translation_ja', '').startswith("意味不明") for seg in translated_segments if seg.get('translation_ja')):
            print("=" * 80, flush=True)
            print("WARNING: Placeholder detected in final japanese_text but not in segments!", flush=True)
            print(f"  japanese_text: {repr(japanese_text[:100])}", flush=True)
            print("=" * 80, flush=True)
        
        return {
            'source_transcription': source_text,
            'japanese_translation': japanese_text,  # Empty string if no translations - DO NOT replace
            'detected_terms': detected_terms,
            'output_video': output_path,
            'segments': translated_segments,  # Full structured output
            'language': 'mixed' if len(set(s['language'] for s in translated_segments)) > 1 else translated_segments[0]['language'] if translated_segments else 'unknown'
        }
    finally:
        # Clean up temporary audio file
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)

def sanitize_subtitle_text(text: str) -> str:
    """
    Sanitize subtitle text to remove characters that cause rendering issues in FFmpeg subtitles.
    """
    if not text:
        return text
    
    import unicodedata
    
    result = []
    for char in text:
        # Get Unicode category
        category = unicodedata.category(char)
        
        # Keep:
        # - Letters (L*)
        # - Numbers (N*)
        # - Punctuation (P*)
        # - Symbols (S*)
        # - Spaces, newlines, tabs
        # - Common punctuation marks
        
        if category[0] in 'LNP' or category == 'Zs' or char in '\n\t':
            # Additional check: ensure it's a valid, renderable character
            try:
                # Try to normalize and encode
                normalized = unicodedata.normalize('NFKC', char)
                normalized.encode('utf-8')
                result.append(normalized)
            except:
                # Skip characters that can't be encoded
                continue
        elif category[0] == 'C':
            # Control characters - replace with space (except newline/tab)
            if char not in '\n\t':
                result.append(' ')
        else:
            # Keep other characters (spaces, etc.)
            result.append(char)
    
    # Clean up multiple spaces
    cleaned = ''.join(result)
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
    
    return cleaned.strip()

