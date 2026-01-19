import os
import subprocess
import tempfile
from pydub import AudioSegment
from backend_python.whisper_transcribe import transcribe_audio, transcribe_audio_with_timestamps
from backend_python.detect_terms import detect_terms
import re
import unicodedata
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter

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

MIN_SUBTITLE_DURATION_SECONDS = 1.5
SILENCE_THRESHOLD_SECONDS = 0.8  # Max gap between speech segments to merge into a block


def format_timestamp(seconds: float) -> str:
    """Format seconds to SRT timestamp format (HH:MM:SS,mmm)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def build_speech_blocks(asr_segments: list) -> list:
    """
    Build speech blocks from ASR segments by merging adjacent segments whose
    silence gap is less than SILENCE_THRESHOLD_SECONDS and that contain text.
    
    Music-only or silent gaps MUST NOT produce blocks.
    """
    if not asr_segments:
        return []

    # Consider only segments with non-empty text as speech
    speech_segments = [
        seg for seg in asr_segments
        if str(seg.get("text", "")).strip()
    ]
    if not speech_segments:
        return []

    # Ensure segments are sorted by start time
    speech_segments.sort(key=lambda s: s.get("start", 0.0))

    blocks = []
    first_seg = speech_segments[0]
    current_block = {
        "start": first_seg.get("start", 0.0),
        "end": first_seg.get("end", first_seg.get("start", 0.0)),
        "segments": [first_seg],
    }
    last_seg_end = current_block["end"]

    silence_gaps = []

    for seg in speech_segments[1:]:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", seg_start)
        gap = seg_start - last_seg_end

        if gap >= 0:
            silence_gaps.append(gap)

        if gap < SILENCE_THRESHOLD_SECONDS:
            # Same speech block
            current_block["end"] = max(current_block["end"], seg_end)
            current_block["segments"].append(seg)
        else:
            # Start a new speech block
            blocks.append(current_block)
            current_block = {
                "start": seg_start,
                "end": seg_end,
                "segments": [seg],
            }

        last_seg_end = seg_end

    # Append the last block
    blocks.append(current_block)

    avg_silence_gap = sum(silence_gaps) / len(silence_gaps) if silence_gaps else 0.0

    # Attach debug info via an attribute on the list (without changing signatures)
    try:
        blocks.debug = {
            "total_asr_segments": len(asr_segments),
            "total_speech_blocks": len(blocks),
            "avg_silence_gap": avg_silence_gap,
        }
    except Exception:
        # If attribute setting fails for any reason, ignore — it's only for logging
        pass

    return blocks


def align_japanese_to_segments(asr_segments: list, full_japanese_translation: str) -> list:
    """
    Simple order-based alignment:
    - Split full Japanese text into sentences by 。！？.
    - Assign sentence i to ASR segment i (by index).
    - Preserve ASR start/end timestamps exactly.
    """
    aligned_subtitles = []

    # Split Japanese translation into sentences, preserving punctuation and order
    japanese_sentences = []
    if full_japanese_translation:
        current = ""
        for ch in full_japanese_translation:
            current += ch
            if ch in "。！？":
                sentence = current.strip()
                if sentence:
                    japanese_sentences.append(sentence)
                current = ""
        if current.strip():
            japanese_sentences.append(current.strip())

    # Map sentences to ASR segments by index
    for idx, seg in enumerate(asr_segments):
        start = seg.get("start", 0.0)
        end = seg.get("end", start)
        text = japanese_sentences[idx] if idx < len(japanese_sentences) else ""
        aligned_subtitles.append({
            "start": start,
            "end": end,
            "text": text,
        })

    return aligned_subtitles

def process_video_with_subtitles(
    video_path: str,
    output_path: str,
    source_segments: list,
    detected_terms: dict = None
) -> str:
    """
    Burn subtitles into video using ffmpeg.
    Creates a video with source language, Japanese translation, and cultural term meanings.
    
    Args:
        video_path: Path to input video
        output_path: Path to output video
        source_segments: List of segments with timestamps in source language.
                        Each segment must include:
                          - 'text': source (English) text
                          - 'start': start time (seconds)
                          - 'end': end time (seconds)
                          - 'japanese_text': per-segment Japanese translation
        detected_terms: Dictionary with detected cultural terms (optional)
    """
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
            
            # Use per-segment Japanese translation
            japanese_text = segment.get('japanese_text', '')
            
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
            
            # Build subtitle text - ONLY Japanese (removed source text and terms)
            # Sanitize Japanese text
            japanese_text_sanitized = sanitize_subtitle_text(japanese_text)
            srt_file.write(f"{japanese_text_sanitized}\n\n")
            
            # Store terms for this segment (will be displayed separately on right side)
            if cultural_annotations:
                # Store segment index and annotations for later ASS file creation
                if not hasattr(srt_file, '_term_segments'):
                    srt_file._term_segments = {}
                srt_file._term_segments[i - 1] = cultural_annotations  # Store with 0-based index
                annotations_text = " / ".join(cultural_annotations)
                print(f"  ✓ Found terms in segment {i}: {annotations_text[:80]}...", flush=True)
        
        srt_path = srt_file.name
        term_segments_map = getattr(srt_file, '_term_segments', {})
    
    # Create separate ASS file for detected terms (middle-right, vertical line)
    terms_ass_path = None
    if term_segments_map:
        try:
            # Get actual video resolution for proper positioning
            video_width = 1920
            video_height = 1080
            try:
                result = subprocess.run(
                    ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                     '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0', video_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=10
                )
                if result.returncode == 0 and result.stdout:
                    resolution = result.stdout.decode('utf-8').strip()
                    if 'x' in resolution:
                        video_width, video_height = map(int, resolution.split('x'))
                        print(f"Detected video resolution: {video_width}x{video_height}", flush=True)
            except Exception as e:
                print(f"Could not detect video resolution, using default 1920x1080: {e}", flush=True)
            
            # Create ASS file for terms
            play_res_x = video_width
            play_res_y = video_height
            center_y = video_height // 2  # Middle of screen
            
            ass_content = []
            ass_content.append("[Script Info]")
            ass_content.append("Title: Detected Terms")
            ass_content.append("ScriptType: v4.00+")
            ass_content.append(f"PlayResX: {play_res_x}")
            ass_content.append(f"PlayResY: {play_res_y}")
            ass_content.append("")
            ass_content.append("[V4+ Styles]")
            ass_content.append("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding")
            # Style: TermsRight - positioned on middle-right, cyan color
            # Alignment=6 = middle-right (for vertical centering)
            # We'll override with \an6\pos() in dialogue lines for exact positioning
            # Make font larger and more visible - increased to 34
            ass_content.append("Style: TermsRight,Arial,34,&H00FFFF,&H00FFFF,&H000000,&H80000000,0,0,0,0,100,100,0,0,1,3,0,6,0,0,0,1")
            ass_content.append("")
            ass_content.append("[Events]")
            ass_content.append("Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text")
            
            # Create dialogue entries per segment - show terms only when they appear
            dialogue_count = 0
            sorted_segments = sorted(term_segments_map.items())
            
            for seg_idx, annotations in sorted_segments:
                if seg_idx < len(source_segments):
                    segment = source_segments[seg_idx]
                    start_time = segment['start']
                    end_time = segment['end']
                    start_str = format_timestamp(start_time)
                    end_str = format_timestamp(end_time)
                    
                    # Sort annotations for consistent vertical order
                    sorted_annotations = sorted(annotations)
                    num_terms = len(sorted_annotations)
                    
                    # Stack terms vertically from center with more spacing
                    # Increased spacing to 60 pixels for better vertical line separation
                    vertical_spacing = 60
                    start_offset = -(num_terms - 1) * vertical_spacing // 2
                    
                    for idx, term in enumerate(sorted_annotations):
                        y_pos = center_y + start_offset + (idx * vertical_spacing)
                        x_pos = video_width - 50  # Right side with margin
                        
                        # Parse term to extract pronunciation and meaning
                        # Format is currently: "pronunciation (meaning)"
                        # Convert BOTH pronunciation AND meaning to true vertical Japanese text.
                        # Each character is on its own line, and we add a reference mark at the top.
                        if ' (' in term and term.endswith(')'):
                            # Split pronunciation and meaning
                            parts = term.rsplit(' (', 1)
                            pronunciation = parts[0]
                            meaning = parts[1].rstrip(')')
                            
                            # Make pronunciation vertical: each character on its own line
                            vertical_pronunciation = "\\N".join(list(pronunciation))
                            
                            # Make meaning vertical: each character on its own line
                            vertical_meaning = "\\N".join(list(meaning))
                            
                            # Reference mark at the very top (e.g. ※), then a blank line,
                            # then vertical pronunciation, another blank line,
                            # then vertical meaning. All vertical.
                            reference_mark = "※"
                            formatted_term = (
                                f"{reference_mark}"
                                f"\\N"            # blank line after reference mark (since next starts with \N)
                                f"\\N{vertical_pronunciation}"
                                f"\\N"            # extra space between pronunciation and meaning
                                f"\\N{vertical_meaning}"
                            )
                        else:
                            # Fallback: convert entire term to vertical character by character
                            formatted_term = "\\N".join(list(term))
                        
                        # Use \an6\pos() - \an6 sets anchor to middle-right (6)
                        # \pos() then positions at exact coordinates
                        # \N creates a line break in ASS format
                        dialogue_line = f"Dialogue: 0,{start_str},{end_str},TermsRight,,0,0,0,,{{\\an6\\pos({x_pos},{y_pos})}}{formatted_term}"
                        ass_content.append(dialogue_line)
                        dialogue_count += 1
                        print(f"  Segment {seg_idx+1}, Dialogue {dialogue_count}: pos=({x_pos},{y_pos}), term='{formatted_term[:50]}'", flush=True)
            
            # Write ASS file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.ass', delete=False, encoding='utf-8') as terms_ass_file:
                terms_ass_path = terms_ass_file.name
                full_content = "\n".join(ass_content) + "\n"
                terms_ass_file.write(full_content)
                terms_ass_file.flush()
                os.fsync(terms_ass_file.fileno())
            
            print(f"Created terms ASS file with {dialogue_count} dialogue entries for middle-right vertical display", flush=True)
            print(f"ASS file created: {terms_ass_path}, size: {os.path.getsize(terms_ass_path)} bytes", flush=True)
            
            # Verify ASS file content
            if os.path.getsize(terms_ass_path) > 0:
                with open(terms_ass_path, 'r', encoding='utf-8') as verify_file:
                    full_content_check = verify_file.read()
                    sample_content = full_content_check[:500]
                    print(f"ASS file sample (first 500 chars): {sample_content}", flush=True)
                    dialogue_lines = [line for line in full_content_check.split('\n') if line.startswith('Dialogue:')]
                    print(f"Found {len(dialogue_lines)} dialogue lines in ASS file", flush=True)
                    if dialogue_lines:
                        print(f"First dialogue line: {dialogue_lines[0][:150]}", flush=True)
                        # Check if positioning tags are present
                        if '\\pos(' in dialogue_lines[0] or '\\an6' in dialogue_lines[0]:
                            print(f"✓ Positioning tags found in ASS file", flush=True)
                        else:
                            print(f"⚠ WARNING: No positioning tags found in first dialogue line!", flush=True)
        except Exception as e:
            print(f"ERROR creating ASS file for terms: {e}", flush=True)
            import traceback
            print(f"Traceback: {traceback.format_exc()}", flush=True)
            terms_ass_path = None
    
    try:
        # Use ffmpeg to burn subtitles into video
        srt_path_escaped = srt_path.replace('\\', '\\\\').replace(':', '\\:')
        
        # Build filter: main subtitles (Japanese only) + terms overlay (middle-right)
        if terms_ass_path:
            # Escape paths for ffmpeg filter (handle backslashes, colons, and single quotes)
            terms_path_escaped = terms_ass_path.replace('\\', '\\\\').replace(':', '\\:').replace("'", "'\\''")
            # Use filter_complex to apply both subtitle filters
            # Try using 'subtitles' filter for ASS file - it might work better when chained
            # The 'subtitles' filter supports ASS files and might handle the chain better
            filter_complex = (
                f"[0:v]subtitles='{srt_path_escaped}':force_style='FontSize=18,PrimaryColour=&Hffffff,"
                f"OutlineColour=&H000000,BorderStyle=1,Alignment=2,MarginV=20'[v1];"
                f"[v1]subtitles='{terms_path_escaped}'[v]"
            )
            print(f"Applying subtitles: main SRT at {srt_path}, terms ASS at {terms_ass_path}", flush=True)
            print(f"Filter complex: {filter_complex}", flush=True)
            
            result = subprocess.run(
                [
                    'ffmpeg',
                    '-i', video_path,
                    '-filter_complex', filter_complex,
                    '-map', '[v]',
                    '-map', '0:a',
                    '-c:a', 'copy',
                    '-y',
                    output_path
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            # Log FFmpeg stderr to check for any ASS file processing issues
            if result.stderr:
                stderr_output = result.stderr.decode('utf-8', errors='ignore')
                # Check for ASS-related errors or warnings
                if 'ass' in stderr_output.lower() or 'subtitle' in stderr_output.lower():
                    print(f"FFmpeg stderr (relevant parts): {stderr_output[:1000]}", flush=True)
        else:
            # Only main subtitles (no terms)
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
        if terms_ass_path and os.path.exists(terms_ass_path):
            os.unlink(terms_ass_path)
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
    Complete workflow with DB-driven English → Japanese translation.
    
    NEW PIPELINE:
    1. Extract audio and transcribe to get source_transcription (English)
    2. Read source_transcription and translate directly to Japanese in a single call
    3. Store result as final Japanese output
    4. Burn subtitles into video
    
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
        
        # Step 2: Transcribe with timestamps to get source_transcription
        source_segments = transcribe_with_timestamps(temp_audio, language=None)
        
        if not source_segments:
            raise ValueError("No segments transcribed")
        
        # Step 2.1: Check for generic/demo content
        full_transcription = ' '.join(seg.get('text', '') for seg in source_segments if seg.get('text', '').strip())
        generic_check = detect_generic_demo_content(full_transcription, source_segments)
        
        if generic_check['is_generic']:
            reasons_str = '; '.join(generic_check['reasons'])
            error_msg = (
                f"This trailer appears to contain generic/demo content rather than actual movie dialogue. "
                f"Confidence: {generic_check['confidence']:.1%}. "
                f"Reasons: {reasons_str}. "
                f"Please try a different trailer or movie."
            )
            print(f"⚠️ GENERIC CONTENT DETECTED: {error_msg}", flush=True)
            raise ValueError(error_msg)
        
        # Step 2.5: Per-segment cleanup and translation for perfect timing (OPTIMIZED: Parallel)
        from backend_python.translate import cleanup_english_transcription, translate_english_to_japanese
        from backend_python.detect_terms import detect_terms

        subtitle_segments = []
        cleaned_segments = []
        print("=" * 80, flush=True)
        print("=== PER-SEGMENT TRANSLATION PIPELINE (PARALLEL) ===", flush=True)
        
        # Prepare segment data for parallel processing
        segment_data = []
        for idx, seg in enumerate(source_segments):
            raw_text = seg.get('text', '') or ''
            start = seg.get('start', 0.0)
            end = seg.get('end', start)
            duration = end - start
            
            cleaned_en = cleanup_english_transcription(raw_text) or ""
            segment_data.append({
                'idx': idx,
                'raw_text': raw_text,
                'start': start,
                'end': end,
                'duration': duration,
                'cleaned_en': cleaned_en
            })
            cleaned_segments.append(cleaned_en)
        
        # Parallel translation of segments
        def translate_segment(seg_data):
            """Helper function to translate a single segment"""
            idx = seg_data['idx']
            cleaned_en = seg_data['cleaned_en']
            jp_text = ""
            
            if cleaned_en.strip():
                try:
                    from backend_python.translate import translate_subtitle_segment
                    jp_text = translate_subtitle_segment(cleaned_en)
                except Exception as e:
                    print(f"[SEG {idx}] Translation error: {e}", flush=True)
                    jp_text = ""
            
            return {
                'idx': idx,
                'jp_text': jp_text,
                'seg_data': seg_data
            }
        
        # Execute translations in parallel (max 10 concurrent to avoid API rate limits)
        translated_results = [None] * len(segment_data)
        max_workers = min(10, len(segment_data))  # Limit concurrent API calls
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all translation tasks
            future_to_seg = {executor.submit(translate_segment, seg_data): seg_data for seg_data in segment_data}
            
            # Collect results as they complete
            for future in as_completed(future_to_seg):
                try:
                    result = future.result()
                    translated_results[result['idx']] = result
                    idx = result['idx']
                    jp_text = result['jp_text']
                    duration = result['seg_data']['duration']
                    print(f"[SEG {idx}] duration={duration:.2f}s", flush=True)
                    print(f"[SEG {idx}] JP: {repr(jp_text[:120])}", flush=True)
                except Exception as e:
                    seg_data = future_to_seg[future]
                    print(f"[SEG {seg_data['idx']}] Unexpected error: {e}", flush=True)
                    # Create a fallback result
                    translated_results[seg_data['idx']] = {
                        'idx': seg_data['idx'],
                        'jp_text': '',
                        'seg_data': seg_data
                    }
        
        # Build subtitle_segments in order (ensuring all segments are included)
        for idx, result in enumerate(translated_results):
            if result is None:
                # Fallback: create empty entry if translation completely failed
                seg_data = segment_data[idx]
                subtitle_segments.append({
                    'text': seg_data['raw_text'],
                    'start': seg_data['start'],
                    'end': seg_data['end'],
                    'japanese_text': '',
                })
            else:
                seg_data = result['seg_data']
                subtitle_segments.append({
                    'text': seg_data['raw_text'],
                    'start': seg_data['start'],
                    'end': seg_data['end'],
                    'japanese_text': result['jp_text'],
                })
        
        print("=" * 80, flush=True)
        
        # Detect cultural terms from the full source text (optional, for subtitle annotations)
        detected_terms = {"terms": []}
        try:
            # Use the most common language from segments for term detection
            lang_counts = {}
            for seg in source_segments:
                lang = seg.get('detected_language', 'hi')
                lang_counts[lang] = lang_counts.get(lang, 0) + 1
            most_common_lang = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else 'hi'
            
            # Use combined cleaned English for term detection
            source_text_for_terms = ' '.join(s for s in cleaned_segments if s.strip())
            terms_result = detect_terms(source_text_for_terms, source_language=most_common_lang)
            if terms_result and isinstance(terms_result, dict):
                detected_terms = terms_result
                print(f"Detected {len(detected_terms.get('terms', []))} cultural terms", flush=True)
        except Exception as e:
            print(f"Error detecting terms: {e}", flush=True)
        
        # Step 4 & 5: Burn subtitles into video using per-segment translations
        process_video_with_subtitles(video_path, output_path, subtitle_segments, detected_terms=detected_terms)
        
        # Determine language (for compatibility)
        lang_counts = {}
        for seg in source_segments:
            lang = seg.get('detected_language', 'hi')
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        detected_language = max(lang_counts.items(), key=lambda x: x[1])[0] if lang_counts else 'unknown'
        if len(lang_counts) > 1:
            detected_language = 'mixed'
        
        # Build segments output for compatibility
        translated_segments = []
        for idx, seg in enumerate(source_segments):
            translated_segments.append({
                'text_original': seg['text'],
                'language': seg.get('detected_language', 'hi'),
                'script': seg.get('script', 'Unknown'),
                'translation_ja': subtitle_segments[idx].get('japanese_text', ''),
                'cultural_terms': [],
                'start': seg['start'],
                'end': seg['end'],
                'confidence': seg.get('confidence', 0.5),
                'needs_review': seg.get('needs_review', False)
            })
        
        print("=" * 80, flush=True)
        print("=== TRANSLATION PIPELINE COMPLETE ===", flush=True)
        full_cleaned = ' '.join(s for s in cleaned_segments if s.strip())
        print(f"Source transcription (cleaned, combined): {len(full_cleaned)} chars", flush=True)
        total_jp_chars = sum(len(s.get('japanese_text', '') or '') for s in subtitle_segments)
        print(f"Total Japanese translation chars (per-segment sum): {total_jp_chars}", flush=True)
        print("=" * 80, flush=True)
        
        return {
            'source_transcription': full_cleaned,
            'japanese_translation': '\n'.join(s.get('japanese_text', '') or '' for s in subtitle_segments),
            'detected_terms': detected_terms,
            'output_video': output_path,
            'segments': translated_segments,  # For compatibility (translation_ja not used)
            'language': detected_language
        }
    finally:
        # Clean up temporary audio file
        if temp_audio and os.path.exists(temp_audio):
            os.unlink(temp_audio)

def detect_generic_demo_content(transcription: str, segments: list = None) -> dict:
    """
    Detect if transcription contains generic/demo content rather than actual movie dialogue.
    
    Args:
        transcription: Full transcription text
        segments: List of segments (optional, for more detailed analysis)
    
    Returns:
        dict with keys:
            - is_generic: bool (True if detected as generic/demo content)
            - confidence: float (0.0 to 1.0, higher = more confident it's generic)
            - reasons: list of strings explaining why it was flagged
    """
    if not transcription or not transcription.strip():
        return {
            'is_generic': False,
            'confidence': 0.0,
            'reasons': []
        }
    
    transcription_lower = transcription.lower()
    reasons = []
    confidence_score = 0.0
    
    # Pattern 1: Common generic/demo phrases
    generic_phrases = [
        'if you have any questions',
        'please post them in the comments',
        'transcribe all audio',
        'including dialogue',
        'narration',
        'voiceovers',
        'any spoken content',
        'this is a movie trailer',
        'transcribe accurately',
        'please subscribe',
        'like and share',
        'click the bell icon',
        'for more videos',
        'check out our channel',
        'available on',
        'pc, ps4, xbox',
        'coming soon',
        'watch now',
        'download now',
        'visit our website',
    ]
    
    found_phrases = []
    for phrase in generic_phrases:
        if phrase in transcription_lower:
            found_phrases.append(phrase)
            confidence_score += 0.15
    
    if found_phrases:
        reasons.append(f"Found generic phrases: {', '.join(found_phrases[:3])}")
    
    # Pattern 2: High repetition (same sentence repeated multiple times)
    if segments:
        segment_texts = [seg.get('text', '').strip().lower() for seg in segments if seg.get('text', '').strip()]
        if len(segment_texts) > 1:
            # Count how many segments are very similar
            text_counts = Counter(segment_texts)
            most_common = text_counts.most_common(1)[0]
            if most_common[1] > len(segment_texts) * 0.6:  # Same text in >60% of segments
                confidence_score += 0.4
                reasons.append(f"High repetition: same text appears in {most_common[1]}/{len(segment_texts)} segments")
    
    # Pattern 3: Very short or very repetitive transcription
    words = transcription.split()
    if len(words) < 20:  # Very short transcription
        confidence_score += 0.2
        reasons.append("Transcription is very short (<20 words)")
    
    # Check for repetitive words
    if len(words) > 0:
        word_counts = Counter(word.lower() for word in words)
        most_common_word = word_counts.most_common(1)[0]
        if most_common_word[1] > len(words) * 0.3:  # Same word appears in >30% of words
            confidence_score += 0.25
            reasons.append(f"High word repetition: '{most_common_word[0]}' appears {most_common_word[1]} times")
    
    # Pattern 4: Technical/instructional language
    technical_keywords = ['transcribe', 'audio', 'dialogue', 'narration', 'voiceover', 'content', 'include']
    technical_count = sum(1 for keyword in technical_keywords if keyword in transcription_lower)
    if technical_count >= 3:
        confidence_score += 0.2
        reasons.append(f"Contains {technical_count} technical/instructional keywords")
    
    # Pattern 5: No variety in sentence structure (all segments very similar length)
    if segments and len(segments) > 2:
        segment_lengths = [len(seg.get('text', '').split()) for seg in segments if seg.get('text', '').strip()]
        if segment_lengths:
            avg_length = sum(segment_lengths) / len(segment_lengths)
            # If all segments are very similar length (low variance), might be repetitive
            variance = sum((l - avg_length) ** 2 for l in segment_lengths) / len(segment_lengths)
            if variance < 2.0 and len(segment_lengths) > 3:  # Very low variance
                confidence_score += 0.15
                reasons.append("Low sentence structure variety (all segments similar length)")
    
    # Normalize confidence to 0.0-1.0
    confidence = min(1.0, confidence_score)
    is_generic = confidence >= 0.4  # Threshold: 40% confidence = generic
    
    return {
        'is_generic': is_generic,
        'confidence': confidence,
        'reasons': reasons
    }

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

