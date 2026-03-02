"""
real_world_audio_utils.py — shared helpers for scanning and serving real-world audio.

Used by annotate.py and inspector.py.
"""

import mimetypes
import os
import random
import subprocess
from pathlib import Path

AUDIO_EXTENSIONS = {'.mp3', '.wav', '.flac', '.ogg', '.m4a', '.aac'}

# Duration thresholds — skip short clips (<1 min) and DJ sets (>15 min)
MIN_DURATION = 60
MAX_DURATION = 900

_CACHE_DIR = Path('./cache/music-lib')


def audio_duration(path: str | Path) -> float | None:
    """Return duration in seconds via ffprobe, or None on failure."""
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.strip())
    except ValueError:
        return None


def is_valid_duration(path: str | Path) -> bool:
    """Return True if the file's duration is within MIN_DURATION–MAX_DURATION."""
    dur = audio_duration(path)
    return dur is not None and MIN_DURATION <= dur <= MAX_DURATION


def scan_audio_files(
    root: str | Path,
    *,
    use_cache: bool = True,
) -> list[str]:
    """
    Recursively scan root for audio files.

    Returns a shuffled list of absolute path strings.  No duration filtering
    is applied — call is_valid_duration() lazily before using each file.
    Result is cached to ./cache/music-lib/<sanitized-root>.txt.
    Pass use_cache=False to force a fresh scan (e.g. after adding new music).
    """
    cache = _CACHE_DIR / f'{str(root).replace("/", "-")}.txt'

    if use_cache and cache.exists():
        print(f'[music-lib] loading music file list from cache: {cache}')
        return [l for l in cache.read_text().splitlines() if l]

    all_files: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if Path(fn).suffix.lower() in AUDIO_EXTENSIONS:
                all_files.append(str(Path(dirpath) / fn))

    random.shuffle(all_files)
    print(f'[music-lib] found {len(all_files)} audio files')

    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text('\n'.join(all_files))
        print(f'[music-lib] cached {len(all_files)} paths → {cache}')

    return all_files


def make_audio_response(filepath: str | Path):
    """
    Return a Flask Response for the given audio file.

    WAV and FLAC are transcoded to OGG Vorbis on the fly so the browser
    can play them.  All other formats are served directly.
    """
    from flask import Response, abort, send_file

    fp  = str(filepath)
    ext = Path(fp).suffix.lower()

    if ext in {'.wav', '.flac', '.aiff', '.aif'}:
        cmd = [
            'ffmpeg', '-v', 'warning',
            '-i', fp,
            '-ac', '1', '-c:a', 'libvorbis', '-qscale:a', '5',
            '-f', 'ogg', 'pipe:1',
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if proc.returncode != 0:
            print(f'[audio] ffmpeg transcode failed: {proc.stderr.decode()[:200]}')
            abort(500)
        return Response(proc.stdout, mimetype='audio/ogg')

    mime = mimetypes.guess_type(fp)[0] or 'application/octet-stream'
    return send_file(fp, mimetype=mime, conditional=True)
