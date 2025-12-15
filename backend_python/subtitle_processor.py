import os
import subprocess
import tempfile
from pydub import AudioSegment
from backend_python.whisper_transcribe import transcribe_audio, transcribe_audio_with_timestamps
from backend_python.translate import translate_to_japanese
from backend_python.detect_terms import detect_terms
import re

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
    Transcribe audio and return segments with timestamps.
    Uses Whisper API with word-level timestamps for accurate timing.
    
    Args:
        audio_path: Path to audio file
        language: Language code (hi, kn, ta). If None, auto-detect.
    """
    try:
        # Try to get word-level timestamps
        result = transcribe_audio_with_timestamps(audio_path, language=language)
        
        # Use segments from Whisper if available
        if hasattr(result, 'segments') and result.segments:
            segments = []
            for seg in result.segments:
                # Handle both dict and object formats
                if isinstance(seg, dict):
                    text = seg.get('text', '').strip()
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                else:
                    # It's a TranscriptionSegment object
                    text = getattr(seg, 'text', '').strip()
                    start = getattr(seg, 'start', 0)
                    end = getattr(seg, 'end', 0)
                
                if text:
                    segments.append({
                        'start': start,
                        'end': end,
                        'text': text
                    })
            
            if segments:
                return segments
        
        # Fallback: use text transcription and estimate timestamps
        if hasattr(result, 'text'):
            transcription = result.text
        else:
            transcription = transcribe_audio(audio_path, language=language)
        
    except Exception as e:
        print(f"Error getting timestamps, using fallback: {e}", flush=True)
        transcription = transcribe_audio(audio_path, language=language)
    
    # Fallback: Split transcription into sentences and estimate timestamps
    # Use appropriate sentence delimiters for different languages
    sentence_delimiters = r'[.!?।।]\s+'  # Works for Hindi, Kannada, Tamil
    sentences = re.split(sentence_delimiters, transcription)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    segments = []
    current_time = 0.0
    
    for sentence in sentences:
        if not sentence:
            continue
        
        # Estimate duration based on character count (rough: 10 chars per second)
        duration = max(2.0, len(sentence) / 10.0)
        end_time = current_time + duration
        
        segments.append({
            'start': current_time,
            'end': end_time,
            'text': sentence
        })
        
        current_time = end_time + 0.5  # 0.5 second gap between subtitles
    
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
            subtitle_lines.append(japanese_text)  # Japanese translation (second line)
            
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
    Complete workflow with multi-language support:
    1. Extract audio from video
    2. Transcribe audio (Hindi, Kannada, or Tamil)
    3. Translate to Japanese
    4. Generate subtitles
    5. Burn subtitles into video
    
    Args:
        video_path: Path to video file
        output_path: Path for output video
        language: Language code (hi, kn, ta). If None, auto-detect.
    
    Returns dict with transcription, translation, and output video path
    """
    temp_audio = None
    try:
        # Step 1: Extract audio
        temp_audio = video_path.rsplit('.', 1)[0] + '_temp_audio.mp3'
        extract_audio_from_video(video_path, temp_audio)
        
        # Step 2: Transcribe with timestamps
        source_segments = transcribe_with_timestamps(temp_audio, language=language)
        source_text = ' '.join([seg['text'] for seg in source_segments])
        
        # Validate transcription
        if not source_text or not source_text.strip():
            raise ValueError("Transcription resulted in empty text. The audio might not contain speech or the language might be incorrect.")
        
        print(f"Transcribed text length: {len(source_text)} characters", flush=True)
        print(f"Transcribed text preview: {source_text[:200]}...", flush=True)
        
        # Step 3: Translate to Japanese
        try:
            japanese_text = translate_to_japanese(source_text, source_language=language or 'hi')
            
            # Validate translation
            if not japanese_text or not japanese_text.strip():
                raise ValueError("Translation resulted in empty text")
            
            print(f"Translation successful. Length: {len(japanese_text)} characters", flush=True)
        except Exception as translate_error:
            print(f"Translation error: {translate_error}", flush=True)
            # Return a placeholder or re-raise
            raise Exception(f"Failed to translate text: {str(translate_error)}")
        
        # Step 3.5: Detect cultural terms (non-blocking, continue even if it fails)
        detected_terms = None
        try:
            from backend_python.detect_terms import detect_terms
            detected_terms = detect_terms(source_text, source_language=language or 'hi')
            
            # Validate the result
            if detected_terms and isinstance(detected_terms, dict):
                terms_list = detected_terms.get('terms', [])
                if terms_list and len(terms_list) > 0:
                    print(f"Detected {len(terms_list)} cultural terms", flush=True)
                else:
                    print(f"No cultural terms detected (returned empty list)", flush=True)
            else:
                print(f"detect_terms returned invalid format: {detected_terms}", flush=True)
                detected_terms = {"terms": []}
        except Exception as e:
            print(f"Error detecting terms (non-critical): {e}", flush=True)
            detected_terms = {"terms": []}
        
        # Step 4 & 5: Burn subtitles into video (pass detected_terms)
        process_video_with_subtitles(video_path, output_path, source_segments, japanese_text, detected_terms=detected_terms)
        
        return {
            'source_transcription': source_text,
            'japanese_translation': japanese_text,
            'detected_terms': detected_terms,
            'output_video': output_path,
            'segments': source_segments,
            'language': language or 'auto-detected'
        }
    finally:
        # Clean up temporary audio file
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)

