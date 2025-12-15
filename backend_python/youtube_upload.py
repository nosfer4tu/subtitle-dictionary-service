from urllib.parse import urlparse, parse_qs
import yt_dlp

def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    if parsed.query:
        params = parse_qs(parsed.query)
        if "v" in params:
            return params["v"][0]
    if "youtu.be" in parsed.netloc:
        return parsed.path.lstrip("/")
    raise ValueError("Invalid YouTube URL")

def download_audio(video_id: str, output_path="audio.mp3"):
    url = f"https://www.youtube.com/watch?v={video_id}"
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "quiet": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192"
            }
        ],
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output_path

def download_video(video_url: str, output_path="video.mp4"):
    """Download video from YouTube URL"""
    import os
    import time
    
    base_path = output_path.rsplit('.', 1)[0] if '.' in output_path else output_path
    
    # Remove any existing file with the same base path to avoid conflicts
    for ext in ['.mp4', '.webm', '.mkv', '.m4a', '.mp3']:
        test_path = f"{base_path}{ext}"
        if os.path.exists(test_path):
            try:
                os.remove(test_path)
            except:
                pass
    
    # Try multiple format options in order of preference
    format_options = [
        "best[ext=mp4]/best",  # Prefer MP4
        "bestvideo[ext=mp4]+bestaudio/best",  # Merge video+audio if needed
        "best",  # Fallback to best available
    ]
    
    ydl_opts = {
        "format": format_options[0],
        "outtmpl": f"{base_path}.%(ext)s",
        "quiet": False,  # Set to False to see download progress
        "noplaylist": True,
        "no_warnings": False,
        "extract_flat": False,
    }
    
    actual_filename = None
    download_error = None
    
    # First, extract info to verify URL is valid
    try:
        test_opts = {"quiet": True, "noplaylist": True}
        with yt_dlp.YoutubeDL(test_opts) as ydl:
            info = ydl.extract_info(video_url, download=False)
            if not info:
                raise Exception("Failed to extract video information")
            
            # Check if video is available
            if info.get('availability') == 'private' or info.get('availability') == 'unavailable':
                raise Exception(f"Video is {info.get('availability')}")
    except Exception as e:
        raise Exception(f"Failed to get video info: {str(e)}")
    
    # Now try downloading with different format options
    download_success = False
    last_error = None
    
    for fmt in format_options:
        try:
            ydl_opts["format"] = fmt
            print(f"Trying format: {fmt}", flush=True)
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Get the expected filename
                info = ydl.extract_info(video_url, download=False)
                actual_filename = ydl.prepare_filename(info)
                print(f"Downloading video to: {actual_filename}", flush=True)
                
                # Download the video
                ydl.download([video_url])
            
            download_success = True
            break
            
        except Exception as e:
            last_error = str(e)
            print(f"Format {fmt} failed: {e}", flush=True)
            # Clean up any partial files
            if actual_filename and os.path.exists(actual_filename):
                try:
                    if os.path.getsize(actual_filename) == 0:
                        os.remove(actual_filename)
                except:
                    pass
            actual_filename = None
            continue
    
    if not download_success:
        download_error = last_error or "Unknown error"
        raise Exception(f"Download failed with all format options. Last error: {download_error}")
    
    # Wait for file to be fully written and stabilize
    max_wait = 50  # Wait up to 5 seconds
    wait_count = 0
    last_size = -1
    stable_count = 0
    
    while wait_count < max_wait:
        if actual_filename and os.path.exists(actual_filename):
            current_size = os.path.getsize(actual_filename)
            if current_size > 0:
                if current_size == last_size:
                    stable_count += 1
                    if stable_count >= 5:  # File size stable for 5 checks
                        break
                else:
                    stable_count = 0
                last_size = current_size
        time.sleep(0.1)
        wait_count += 1
    
    # Return the actual downloaded file path (check if it exists and has content)
    if actual_filename and os.path.exists(actual_filename):
        file_size = os.path.getsize(actual_filename)
        if file_size == 0:
            # Try to get more info about why it's empty
            raise Exception(f"Downloaded video file is empty (0 bytes): {actual_filename}. Download may have failed.")
        print(f"Successfully downloaded video: {actual_filename} ({file_size} bytes)", flush=True)
        return actual_filename
    
    # If the prepared filename doesn't exist, try the base path with common extensions
    for ext in ['.mp4', '.webm', '.mkv', '.m4a']:
        test_path = f"{base_path}{ext}"
        if os.path.exists(test_path):
            file_size = os.path.getsize(test_path)
            if file_size > 0:
                print(f"Found video file: {test_path} ({file_size} bytes)", flush=True)
                return test_path
    
    # If nothing found, raise an error with more details
    error_msg = f"Downloaded video file not found. Expected: {actual_filename or 'unknown'} or {base_path}.*"
    if download_error:
        error_msg += f" Download error: {download_error}"
    raise Exception(error_msg)
