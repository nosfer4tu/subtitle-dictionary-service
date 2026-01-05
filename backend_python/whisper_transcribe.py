from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import os
import re
import warnings

# Suppress warnings for optional audio libraries
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import audio processing libraries (optional)
try:
    import librosa  # type: ignore
    import soundfile as sf  # type: ignore
    import numpy as np  # type: ignore
    AUDIO_LIBS_AVAILABLE = True
except ImportError:
    AUDIO_LIBS_AVAILABLE = False
    # Fallback: will use basic file operations for duration only
    # Note: All functions that use np/librosa check AUDIO_LIBS_AVAILABLE first

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


# ============================================================================
# CALL 0 ASR PRE-PROCESSING GATE
# ============================================================================
# This gate decides whether an audio segment is safe to send to Whisper-1
# for transcription. It operates on audio BEFORE transcription.

def duration(audio_file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Duration in seconds
    """
    if not os.path.exists(audio_file_path):
        return 0.0
    
    if AUDIO_LIBS_AVAILABLE:
        try:
            y, sr = librosa.load(audio_file_path, sr=None)
            duration_sec = len(y) / sr
            return duration_sec
        except Exception as e:
            print(f"Warning: Could not load audio with librosa: {e}", flush=True)
            # Fallback to basic file size estimation (very rough)
            file_size = os.path.getsize(audio_file_path)
            # Rough estimate: assume ~16kbps (very conservative)
            estimated_duration = file_size / 2000  # bytes to seconds (rough)
            return estimated_duration
    else:
        # Fallback: estimate from file size (very rough)
        file_size = os.path.getsize(audio_file_path)
        # Rough estimate: assume ~16kbps
        estimated_duration = file_size / 2000
        return estimated_duration


def estimate_speakers(audio_file_path: str) -> int:
    """
    Estimate the number of speakers in an audio segment.
    
    This is a placeholder implementation. For production, use:
    - pyannote.audio for speaker diarization
    - speechbrain for speaker verification
    - Custom ML models trained on speaker separation
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Estimated number of speakers (1, 2, or more)
        Returns 1 if unable to determine (conservative - assume single speaker)
    """
    if not AUDIO_LIBS_AVAILABLE:
        # Without audio libraries, cannot analyze - assume single speaker (conservative)
        return 1
    
    try:
        # Load audio
        y, sr = librosa.load(audio_file_path, sr=16000)  # Standardize to 16kHz
        
        # Basic heuristic: analyze spectral characteristics
        # Multiple speakers often have more variation in spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Calculate variation metrics
        centroid_std = np.std(spectral_centroids)
        rolloff_std = np.std(spectral_rolloff)
        
        # Heuristic: high variation suggests multiple speakers or overlapping speech
        # Thresholds are tuned for typical speech
        if centroid_std > 500 or rolloff_std > 1000:
            # Likely multiple speakers or overlapping
            return 2  # Conservative: assume 2+ speakers
        
        # Additional check: energy variation (speakers often have different energy levels)
        rms = librosa.feature.rms(y=y)[0]
        rms_std = np.std(rms)
        if rms_std > 0.1:  # High energy variation
            return 2
        
        return 1  # Single speaker likely
    except Exception as e:
        print(f"Warning: Could not analyze speakers: {e}. Assuming single speaker.", flush=True)
        return 1  # Conservative: assume single speaker on error


def detect_emotion(audio_file_path: str) -> dict:
    """
    Detect emotional vocalization in audio (shouting, high emotion, argumentative tone).
    
    This is a placeholder implementation. For production, use:
    - wav2vec2 for emotion recognition
    - Custom emotion classification models
    - Prosody analysis (pitch, intensity, tempo)
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        dict with keys:
            - has_shouting: bool
            - has_high_emotion: bool
            - has_argumentative_tone: bool
            - overall_emotional: bool (True if any emotional indicator detected)
    """
    if not AUDIO_LIBS_AVAILABLE:
        # Without audio libraries, cannot analyze - assume neutral (conservative)
        return {
            'has_shouting': False,
            'has_high_emotion': False,
            'has_argumentative_tone': False,
            'overall_emotional': False
        }
    
    try:
        y, sr = librosa.load(audio_file_path, sr=16000)
        
        # Analyze prosodic features
        # 1. Intensity (RMS energy) - shouting has high energy
        rms = librosa.feature.rms(y=y)[0]
        max_rms = np.max(rms)
        avg_rms = np.mean(rms)
        
        # 2. Pitch (F0) - emotional speech often has wider pitch range
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            pitch_range = np.max(pitch_values) - np.min(pitch_values) if len(pitch_values) > 1 else 0
            avg_pitch = np.mean(pitch_values)
        else:
            pitch_range = 0
            avg_pitch = 0
        
        # 3. Zero crossing rate - emotional speech can have higher ZCR
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zcr)
        
        # Heuristics for emotional detection
        has_shouting = max_rms > 0.3 or (max_rms / avg_rms) > 2.0  # High energy or sudden spikes
        has_high_emotion = pitch_range > 200 or avg_pitch > 300  # Wide pitch range or high pitch
        has_argumentative_tone = avg_zcr > 0.15 or (max_rms > 0.25 and pitch_range > 150)  # High ZCR or combination
        
        overall_emotional = has_shouting or has_high_emotion or has_argumentative_tone
        
        return {
            'has_shouting': has_shouting,
            'has_high_emotion': has_high_emotion,
            'has_argumentative_tone': has_argumentative_tone,
            'overall_emotional': overall_emotional
        }
    except Exception as e:
        print(f"Warning: Could not analyze emotion: {e}. Assuming neutral.", flush=True)
        return {
            'has_shouting': False,
            'has_high_emotion': False,
            'has_argumentative_tone': False,
            'overall_emotional': False
        }


def detect_languages(audio_file_path: str) -> dict:
    """
    Detect languages present in audio segment.
    
    This is a placeholder implementation. For production, use:
    - Whisper's language detection (but we can't use it pre-transcription)
    - wav2vec2 language identification models
    - Custom language classification models
    
    For now, we use a conservative approach: transcribe a small sample and analyze.
    This is not ideal but works as a gate.
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        dict with keys:
            - languages: list of detected language codes
            - is_mixed: bool (True if multiple languages detected)
            - has_english_clauses: bool (True if English sentences/clauses detected)
    """
    # Note: True language detection requires transcription or specialized models
    # For a pre-transcription gate, we use a conservative heuristic:
    # If we can't determine, assume single language (conservative - don't reject)
    
    # For production, you would:
    # 1. Use a lightweight language ID model (e.g., wav2vec2-based)
    # 2. Or transcribe a small sample with Whisper and analyze the text
    # 3. Or use acoustic language models
    
    # Placeholder: return conservative default
    # In production, implement proper language detection
    return {
        'languages': ['unknown'],  # Cannot determine without transcription
        'is_mixed': False,  # Conservative: assume not mixed
        'has_english_clauses': False  # Conservative: assume no English
    }


def repeated_phrase_check(audio_file_path: str) -> dict:
    """
    Check for repeated phrases or proper names in audio.
    
    This is a placeholder implementation. For production, use:
    - Speech recognition on audio to detect repeated text
    - Acoustic similarity matching
    - Pattern detection in audio features
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        dict with keys:
            - has_repetition: bool
            - repetition_type: str ('phrase', 'name', 'none')
    """
    if not AUDIO_LIBS_AVAILABLE:
        return {'has_repetition': False, 'repetition_type': 'none'}
    
    try:
        y, sr = librosa.load(audio_file_path, sr=16000)
        
        # Basic heuristic: detect repetitive patterns in audio features
        # Extract MFCC features (commonly used for speech)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Calculate similarity between time frames
        # If there are highly similar frames, it might indicate repetition
        frame_similarities = []
        for i in range(min(50, mfccs.shape[1] - 1)):  # Check first 50 frames
            for j in range(i + 1, min(i + 20, mfccs.shape[1])):  # Compare with nearby frames
                # Cosine similarity
                similarity = np.dot(mfccs[:, i], mfccs[:, j]) / (
                    np.linalg.norm(mfccs[:, i]) * np.linalg.norm(mfccs[:, j]) + 1e-10
                )
                if similarity > 0.95:  # Very high similarity
                    frame_similarities.append(similarity)
        
        # If many highly similar frames, likely repetition
        has_repetition = len(frame_similarities) > 5
        
        return {
            'has_repetition': has_repetition,
            'repetition_type': 'phrase' if has_repetition else 'none'
        }
    except Exception as e:
        print(f"Warning: Could not check repetition: {e}. Assuming no repetition.", flush=True)
        return {'has_repetition': False, 'repetition_type': 'none'}


def detect_intent_count(audio_file_path: str) -> int:
    """
    Detect the number of communicative intents in audio.
    
    This is a placeholder implementation. For production, use:
    - Speech-to-text + intent classification
    - Prosody-based intent detection
    - Acoustic pattern analysis
    
    Args:
        audio_file_path: Path to audio file
        
    Returns:
        Estimated number of communicative intents (1, 2, or more)
    """
    if not AUDIO_LIBS_AVAILABLE:
        return 1  # Conservative: assume single intent
    
    try:
        y, sr = librosa.load(audio_file_path, sr=16000)
        
        # Heuristic: analyze prosodic breaks and energy patterns
        # Multiple intents often have prosodic boundaries (pauses, energy drops)
        rms = librosa.feature.rms(y=y)[0]
        
        # Detect significant energy drops (potential intent boundaries)
        energy_drops = 0
        for i in range(1, len(rms)):
            if rms[i] < rms[i-1] * 0.5:  # Significant drop
                energy_drops += 1
        
        # Detect pauses (low energy regions)
        low_energy_frames = np.sum(rms < np.mean(rms) * 0.3)
        pause_ratio = low_energy_frames / len(rms)
        
        # Multiple intents likely if:
        # - Multiple energy drops
        # - Significant pause regions
        if energy_drops > 2 or pause_ratio > 0.3:
            return 2  # Likely multiple intents
        
        return 1  # Single intent
    except Exception as e:
        print(f"Warning: Could not detect intent count: {e}. Assuming single intent.", flush=True)
        return 1  # Conservative: assume single intent


def call0_asr_preprocessing_gate(audio_file_path: str) -> dict:
    """
    Call 0: Pre-Whisper segmentation and rejection gate for real speech research.
    
    Role: Decide whether an audio segment is safe to send to Whisper.
    Task is NOT transcription, translation, or repair.
    
    HARD ACCEPTANCE CONDITIONS (ALL must be true):
    1. Duration ≤ 5.0 seconds
    2. Exactly ONE speaker intent (question OR statement OR apology OR command, not multiple)
    3. Single dominant language (Tamil OR English, not both; ignore loanwords only if ≤ 1 short word)
    4. No clause repetition (same semantic idea must not appear twice, even paraphrased)
    5. Emotionally stable (no shouting, no emotional escalation, no dramatic emphasis mixed with dialogue)
    6. No speaker switch (if speaker identity changes or overlaps → reject)
    
    IMMEDIATE REJECTION CONDITIONS (ANY triggers rejection):
    1. English sentence embedded inside Tamil clause
    2. Tamil sentence embedded inside English clause
    3. Repetition, echoing, or restart ("… sorry sorry", "hey hey")
    4. Emotional shouting mixed with speech
    5. Argument + explanation in same segment
    6. Question followed by justification
    7. Apology + defense in same segment
    8. Whisper would need to "guess" missing structure
    
    Args:
        audio_file_path: Path to audio file to evaluate
        
    Returns:
        dict with keys:
            - status: "ACCEPT" or "REJECT"
            - reason: Specific reason for acceptance/rejection
    """
    if not os.path.exists(audio_file_path):
        return {"status": "REJECT", "reason": "file_not_found"}
    
    # HARD ACCEPTANCE CONDITION 1: Duration ≤ 5.0 seconds
    duration_sec = duration(audio_file_path)
    if duration_sec > 5.0:
        return {"status": "REJECT", "reason": "duration_exceeds_5_seconds"}
    
    # HARD ACCEPTANCE CONDITION 2: Exactly ONE speaker intent
    speaker_count = estimate_speakers(audio_file_path)
    if speaker_count > 1:
        return {"status": "REJECT", "reason": "multiple_speakers_or_overlap"}
    
    # Check for speaker switch/overlap (part of condition 6)
    # This is detected by the speaker count check above
    
    # HARD ACCEPTANCE CONDITION 3: Single dominant language
    language_result = detect_languages(audio_file_path)
    if language_result['is_mixed']:
        return {"status": "REJECT", "reason": "mixed_languages"}
    if language_result['has_english_clauses']:
        return {"status": "REJECT", "reason": "english_sentence_in_tamil_or_vice_versa"}
    
    # IMMEDIATE REJECTION: English sentence embedded inside Tamil clause / Tamil in English
    # (covered by language_result above)
    
    # HARD ACCEPTANCE CONDITION 4: No clause repetition
    repetition_result = repeated_phrase_check(audio_file_path)
    if repetition_result['has_repetition']:
        return {"status": "REJECT", "reason": "repetition_echoing_or_restart"}
    
    # IMMEDIATE REJECTION: Repetition, echoing, or restart
    # (covered by repetition_result above)
    
    # HARD ACCEPTANCE CONDITION 5: Emotionally stable
    emotion_result = detect_emotion(audio_file_path)
    if emotion_result['has_shouting']:
        return {"status": "REJECT", "reason": "emotional_shouting"}
    if emotion_result['has_high_emotion']:
        return {"status": "REJECT", "reason": "emotional_escalation"}
    if emotion_result['has_argumentative_tone']:
        # Check if argument + explanation (IMMEDIATE REJECTION condition 5)
        intent_count = detect_intent_count(audio_file_path)
        if intent_count > 1:
            return {"status": "REJECT", "reason": "argument_plus_explanation"}
        return {"status": "REJECT", "reason": "dramatic_emphasis_mixed_with_dialogue"}
    
    # IMMEDIATE REJECTION: Emotional shouting mixed with speech
    # (covered by emotion checks above)
    
    # HARD ACCEPTANCE CONDITION 6: Exactly ONE speaker intent (not multiple intents)
    intent_count = detect_intent_count(audio_file_path)
    if intent_count > 1:
        # Could be: argument + explanation, question + justification, apology + defense
        # These are IMMEDIATE REJECTION conditions 5, 6, 7
        return {"status": "REJECT", "reason": "multiple_intents_detected"}
    
    # IMMEDIATE REJECTION: Question followed by justification
    # IMMEDIATE REJECTION: Apology + defense in same segment
    # (covered by intent_count check above)
    
    # IMMEDIATE REJECTION: Whisper would need to "guess" missing structure
    # This is a catch-all for segments that pass basic checks but seem unstable
    # We check for very short segments (< 1 second) which might be fragments
    if duration_sec < 1.0:
        return {"status": "REJECT", "reason": "whisper_would_need_to_guess_missing_structure"}
    
    # All hard acceptance conditions passed, no immediate rejection conditions triggered
    return {"status": "ACCEPT", "reason": "single_intent_clean_segment"}

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


def score_asr(text: str) -> int:
    """
    HARD PROGRAMMATIC ASR QUALITY GATE: Score ASR output for quality.
    
    This function is used to select the best ASR output from multiple passes.
    Negative scores indicate rejection-worthy output.
    
    Args:
        text: ASR transcription text to score
    
    Returns:
        Score (higher is better). Negative scores indicate rejection.
    """
    if not text or not isinstance(text, str):
        return -100
    
    score = 0
    
    # Penalize mixed scripts (Tamil + Latin)
    tamil_script = re.search(r'[\u0B80-\u0BFF]', text)  # Tamil Unicode range
    latin_script = re.search(r'[A-Za-z]', text)
    if tamil_script and latin_script:
        score -= 50
    
    # Reward clean ASCII English
    if re.match(r'^[\x00-\x7F\s.,\'"!?()-]+$', text):
        score += 30
    
    # Penalize corrupted characters (replacement character)
    if '\ufffd' in text or '' in text:
        score -= 30
    
    # Penalize overly long hallucinated output
    if len(text) > 400:
        score -= 20
    
    return score


def transcribe_audio_multi_pass(file_path: str, use_call0_gate: bool = True) -> dict:
    """
    Multi-pass ASR with language-locked passes and quality scoring.
    
    Runs multiple ASR passes:
    - Pass A: language = "ta" (Tamil only)
    - Pass B: language = "en" (English only)
    - Pass C: language = "auto" (Whisper auto-detect)
    
    Selects the best output based on ASR quality score.
    Hard rejects if no output scores >= 0.
    
    Args:
        file_path: Path to audio file
        use_call0_gate: If True, apply Call 0 ASR preprocessing gate before transcription
    
    Returns:
        dict with keys:
            - 'text': Best ASR output (or None if all rejected)
            - 'language': Language of best output ('ta', 'en', or 'auto')
            - 'score': Quality score of best output
            - 'rejected': True if all passes were rejected
    """
    # Call 0 ASR Pre-processing Gate
    if use_call0_gate:
        gate_result = call0_asr_preprocessing_gate(file_path)
        if gate_result["status"] == "REJECT":
            print(f"ASR REJECTED: Call 0 Gate rejected - reason: {gate_result['reason']}", flush=True)
            return {'text': None, 'language': None, 'score': -100, 'rejected': True}
    
    client = get_client()
    candidates = []
    
    # Pass A: Tamil only
    try:
        with open(file_path, "rb") as f:
            params = {
                "file": f,
                "model": "whisper-1",
                "language": "ta",
                "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
            }
            res = client.audio.transcriptions.create(**params)
            ta_text = clean_transcription(res.text, "ta")
            ta_score = score_asr(ta_text)
            candidates.append({'lang': 'ta', 'text': ta_text, 'score': ta_score})
            print(f"ASR Pass A (Tamil): score={ta_score}, text={repr(ta_text[:50])}", flush=True)
    except Exception as e:
        print(f"ASR Pass A (Tamil) failed: {e}", flush=True)
        candidates.append({'lang': 'ta', 'text': '', 'score': -100})
    
    # Pass B: English only
    try:
        with open(file_path, "rb") as f:
            params = {
                "file": f,
                "model": "whisper-1",
                "language": "en",
                "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
            }
            res = client.audio.transcriptions.create(**params)
            en_text = clean_transcription(res.text, "en")
            en_score = score_asr(en_text)
            candidates.append({'lang': 'en', 'text': en_text, 'score': en_score})
            print(f"ASR Pass B (English): score={en_score}, text={repr(en_text[:50])}", flush=True)
    except Exception as e:
        print(f"ASR Pass B (English) failed: {e}", flush=True)
        candidates.append({'lang': 'en', 'text': '', 'score': -100})
    
    # Pass C: Auto-detect
    try:
        with open(file_path, "rb") as f:
            params = {
                "file": f,
                "model": "whisper-1",
                # No language parameter - let Whisper auto-detect
                "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
            }
            res = client.audio.transcriptions.create(**params)
            auto_text = clean_transcription(res.text, None)
            auto_score = score_asr(auto_text)
            candidates.append({'lang': 'auto', 'text': auto_text, 'score': auto_score})
            print(f"ASR Pass C (Auto): score={auto_score}, text={repr(auto_text[:50])}", flush=True)
    except Exception as e:
        print(f"ASR Pass C (Auto) failed: {e}", flush=True)
        candidates.append({'lang': 'auto', 'text': '', 'score': -100})
    
    # Select best candidate by score
    candidates_with_scores = [c for c in candidates if c['text']]
    if not candidates_with_scores:
        print("ASR REJECTED: All passes failed or produced empty output", flush=True)
        return {'text': None, 'language': None, 'score': -100, 'rejected': True}
    
    best = max(candidates_with_scores, key=lambda x: x['score'])
    
    # HARD STOP — do not continue pipeline if score < 0
    if best['score'] < 0:
        print(f"ASR REJECTED: Best score {best['score']} < 0, language={best['lang']}, text={repr(best['text'][:50])}", flush=True)
        return {'text': None, 'language': None, 'score': best['score'], 'rejected': True}
    
    # Enforce English-only before Call 1
    # Only ASCII English (0x00-0x7F) plus common punctuation allowed
    if not re.match(r'^[\x00-\x7F\s.,\'"!?()-]+$', best['text']):
        print(f"ASR REJECTED: Non-English text detected, language={best['lang']}, text={repr(best['text'][:50])}", flush=True)
        return {'text': None, 'language': None, 'score': best['score'], 'rejected': True}
    
    # Must have at least one letter (not just punctuation/whitespace)
    if not re.search(r'[A-Za-z]', best['text']):
        print(f"ASR REJECTED: No English letters found, text={repr(best['text'][:50])}", flush=True)
        return {'text': None, 'language': None, 'score': best['score'], 'rejected': True}
    
    print(f"ASR ACCEPTED: Best pass={best['lang']}, score={best['score']}, text={repr(best['text'][:50])}", flush=True)
    return {'text': best['text'], 'language': best['lang'], 'score': best['score'], 'rejected': False}


def transcribe_audio(file_path: str, language: str = None, use_call0_gate: bool = True) -> str:
    """
    Transcribe audio using COMPETING ASR PASSES and return best English text.
    
    Runs multiple ASR passes (en, ta, auto) and selects the BEST English candidate
    based on ASR quality scoring BEFORE applying any gates.
    
    Args:
        file_path: Path to audio file
        language: DEPRECATED - competing passes are always run (en, ta, auto)
        use_call0_gate: If True, apply Call 0 ASR preprocessing gate before transcription.
                        If gate rejects, returns empty string.
    
    Returns:
        Best English transcribed text, or empty string if all passes rejected or no English candidate
    """
    # Use multi-pass ASR which handles competing passes and selection
    multi_pass_result = transcribe_audio_multi_pass(file_path, use_call0_gate=use_call0_gate)
    
    if multi_pass_result['rejected'] or not multi_pass_result['text']:
        return ""  # Rejected or no text
    
    return multi_pass_result['text']

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

def transcribe_audio_with_timestamps(file_path: str, language: str = None, use_call0_gate: bool = False) -> dict:
    """
    Transcribe audio with word-level timestamps using COMPETING ASR PASSES.
    
    Runs multiple ASR passes (en, ta, auto) and selects the BEST English candidate
    based on ASR quality scoring BEFORE applying any gates.
    
    Note: Call 0 gate is disabled by default for this function as it's typically used
    for full audio files, not individual segments. Enable only if processing segments.
    
    Args:
        file_path: Path to audio file
        language: DEPRECATED - competing passes are always run (en, ta, auto)
        use_call0_gate: If True, apply Call 0 ASR preprocessing gate before transcription.
                        If gate rejects, returns empty result.
    
    Returns:
        Transcription result with segments from the BEST English ASR candidate, or empty dict if all rejected
    """
    # Call 0 ASR Pre-processing Gate (optional for full-file transcription)
    if use_call0_gate:
        gate_result = call0_asr_preprocessing_gate(file_path)
        if gate_result["status"] == "REJECT":
            print(f"Call 0 Gate: Audio segment rejected - reason: {gate_result['reason']} - not sending to Whisper", flush=True)
            return {'text': '', 'segments': []}  # Rejected - return empty
    
    print("=" * 80, flush=True)
    print("COMPETING ASR PASSES: Running en, ta, and auto-detect", flush=True)
    print("=" * 80, flush=True)
    
    client = get_client()
    candidates = []
    
    # Pass 1: English only
    try:
        print("ASR Pass 1: language='en' (English only)", flush=True)
        with open(file_path, "rb") as f:
            params = {
                "file": f,
                "model": "whisper-1",
                "language": "en",
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
                "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
            }
            res_en = client.audio.transcriptions.create(**params)
            en_text = clean_transcription(res_en.text if hasattr(res_en, 'text') else '', "en")
            en_score = score_asr(en_text)
            candidates.append({'lang': 'en', 'result': res_en, 'text': en_text, 'score': en_score})
            print(f"  Pass 1 (en) score: {en_score}, text preview: {repr(en_text[:50])}", flush=True)
    except Exception as e:
        print(f"  Pass 1 (en) failed: {e}", flush=True)
        candidates.append({'lang': 'en', 'result': None, 'text': '', 'score': -100})
    
    # Pass 2: Tamil only
    try:
        print("ASR Pass 2: language='ta' (Tamil only)", flush=True)
        with open(file_path, "rb") as f:
            params = {
                "file": f,
                "model": "whisper-1",
                "language": "ta",
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
                "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
            }
            res_ta = client.audio.transcriptions.create(**params)
            ta_text = clean_transcription(res_ta.text if hasattr(res_ta, 'text') else '', "ta")
            ta_score = score_asr(ta_text)
            candidates.append({'lang': 'ta', 'result': res_ta, 'text': ta_text, 'score': ta_score})
            print(f"  Pass 2 (ta) score: {ta_score}, text preview: {repr(ta_text[:50])}", flush=True)
    except Exception as e:
        print(f"  Pass 2 (ta) failed: {e}", flush=True)
        candidates.append({'lang': 'ta', 'result': None, 'text': '', 'score': -100})
    
    # Pass 3: Auto-detect
    try:
        print("ASR Pass 3: language=None (auto-detect)", flush=True)
        with open(file_path, "rb") as f:
            params = {
                "file": f,
                "model": "whisper-1",
                # No language parameter - let Whisper auto-detect
                "response_format": "verbose_json",
                "timestamp_granularities": ["segment"],
                "prompt": "This is a movie trailer with dialogue, narration, character lines, and background descriptions. Transcribe ALL audio including dialogue, narration, voiceovers, and any spoken content. Transcribe accurately.",
            }
            res_auto = client.audio.transcriptions.create(**params)
            auto_text = clean_transcription(res_auto.text if hasattr(res_auto, 'text') else '', None)
            auto_score = score_asr(auto_text)
            candidates.append({'lang': 'auto', 'result': res_auto, 'text': auto_text, 'score': auto_score})
            print(f"  Pass 3 (auto) score: {auto_score}, text preview: {repr(auto_text[:50])}", flush=True)
    except Exception as e:
        print(f"  Pass 3 (auto) failed: {e}", flush=True)
        candidates.append({'lang': 'auto', 'result': None, 'text': '', 'score': -100})
    
    # Select BEST English candidate
    # Filter to only English candidates (en pass or auto pass that is English)
    english_candidates = []
    for cand in candidates:
        if cand['lang'] == 'en':
            # English pass - always consider
            english_candidates.append(cand)
        elif cand['lang'] == 'auto' and cand['text']:
            # Auto pass - check if it's English
            if re.match(r'^[\x00-\x7F\s.,\'"!?()-]+$', cand['text']) and re.search(r'[A-Za-z]', cand['text']):
                english_candidates.append(cand)
    
    if not english_candidates:
        print("ASR REJECTED: No English candidates found from competing passes", flush=True)
        return {'text': '', 'segments': []}
    
    # Select best by score
    best = max(english_candidates, key=lambda x: x['score'])
    print(f"SELECTED: Best English candidate = {best['lang']} pass, score={best['score']}", flush=True)
    
    if not best['result']:
        print("ASR REJECTED: Best candidate has no result", flush=True)
        return {'text': '', 'segments': []}
    
    res = best['result']
    
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