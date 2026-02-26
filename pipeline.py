"""
Render MIDI → chunk in one atomic pass. WAV is written to a temp file and deleted
immediately after chunking, so disk usage stays proportional to max-processes, not
total song count.

Usage:
    python pipeline.py \
        --midi-path midi_files \
        --soundfont-path soundfonts \
        --out-path chunks/v2 \
        --target-count 500 \
        --chunks-per-song 3 \
        --max-processes 6
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import librosa
import mido
import soundfile

from config import render_samplerate, samplerate as CHUNK_SR
from midi_utils import (
    calc_bar_starts_in_seconds,
    find_soundfonts,
    find_soundfonts_single,
    has_note_phase_alignment,
    has_valid_time_signature,
    jitter_drum_notes,
    read_events,
    read_note_on_ticks,
    render_wav,
)
from chunk import process_audio

parser = argparse.ArgumentParser(
    description='Render MIDI + chunk in one pass (no persistent WAV).')

# MIDI / render args
parser.add_argument('--midi-path', type=str, default='midi_files')
parser.add_argument('--soundfont-path', type=str, default='soundfonts')
parser.add_argument('--single-soundfont', type=str, default=None,
    help='path to one .sf2 file to use instead of scanning --soundfont-path')
parser.add_argument('--target-count', type=int, default=None,
    help='stop after this many successfully processed songs')
parser.add_argument('--max-song-length', type=int, default=60 * 20,
    help='skip songs longer than this many seconds')
parser.add_argument('--midi-jitter-max-seconds', type=float, default=0.05,
    help='max jitter on drum note_on events in seconds. 0 disables jitter.')
parser.add_argument('--render-timeout-seconds', type=float, default=None,
    help='hard timeout per render call. Defaults to max(120, duration*10).')
parser.add_argument('--phase-grid-denominator', type=int, default=16)
parser.add_argument('--note-phase-tolerance', type=float, default=0.035)
parser.add_argument('--min-aligned-note-ratio', type=float, default=0.6)
parser.add_argument('--min-notes-for-alignment-check', type=int, default=24)
parser.add_argument('--max-processes', type=int, default=4)

# Chunk args
parser.add_argument('--out-path', type=str, required=True,
    help='directory to write chunk .ogg + .bars files')
parser.add_argument('--chunks-per-song', type=int, default=3)
parser.add_argument('--pitch-range', type=float, default=2.0,
    help='max pitch shift in semitones')
parser.add_argument('--overwrite', default=False, action='store_true')

args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)


def _make_stem(midi_rel, soundfont_name):
    midi_part = midi_rel.replace(os.sep, '_').replace('/', '_')
    sf_part = soundfont_name[:30].replace(' ', '_')
    return f'{midi_part}__{sf_part}'


def _validate_midi(midifile, soundfonts, midi_base_path):
    """Parse and filter one MIDI file. Returns (midifile_str, sf, midi_rel, duration, bar_starts) or (None, reason)."""
    midi_rel = str(midifile.relative_to(midi_base_path).with_suffix(''))
    try:
        mid = mido.MidiFile(str(midifile))
        if mid.length > args.max_song_length:
            return None, 'too long'
        tempo_changes, time_signatures, max_tick = read_events(mid)
        if not has_valid_time_signature(time_signatures):
            return None, 'no valid time signature'
        note_ticks = read_note_on_ticks(mid)
        aligned_ok, aligned_ratio, aligned_count = has_note_phase_alignment(
            note_ticks, time_signatures, mid.ticks_per_beat,
            denominator=args.phase_grid_denominator,
            tolerance=args.note_phase_tolerance,
            min_ratio=args.min_aligned_note_ratio,
            min_notes=args.min_notes_for_alignment_check,
        )
        if not aligned_ok:
            return None, f'poor alignment ({aligned_ratio:.2f}/{aligned_count})'
        bar_starts = calc_bar_starts_in_seconds(
            time_signatures, tempo_changes, mid.ticks_per_beat, max_tick)
        if len(bar_starts) < 2:
            return None, 'bar extraction failed'
    except Exception as e:
        return None, f'parse error: {e}'

    sf = random.choice(soundfonts)
    return (str(midifile), sf, midi_rel, mid.length, bar_starts), None


def render_and_chunk(midifile, soundfonts, midi_base_path):
    """Validate, render to temp WAV, chunk, delete WAV. Returns number of chunks written."""
    candidate, reason = _validate_midi(midifile, soundfonts, midi_base_path)
    if candidate is None:
        return 0

    midifile_str, soundfont, midi_rel, duration, bar_starts = candidate

    fd, tmpwav = tempfile.mkstemp(suffix='.wav')
    os.close(fd)
    jitter_tmpfile = None
    try:
        render_midi_path = midifile_str
        if args.midi_jitter_max_seconds > 0:
            mid = mido.MidiFile(midifile_str)
            mid = jitter_drum_notes(mid, args.midi_jitter_max_seconds)
            fd2, jitter_tmpfile = tempfile.mkstemp(suffix='.mid')
            os.close(fd2)
            mid.save(jitter_tmpfile)
            render_midi_path = jitter_tmpfile

        timeout_s = (
            args.render_timeout_seconds
            if args.render_timeout_seconds is not None
            else max(120, duration * 10)
        )
        ok = render_wav(render_midi_path, tmpwav, soundfont, timeout_s, render_samplerate)
        if not ok:
            return 0

        audio, sr = soundfile.read(tmpwav, dtype='float32')
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        if sr != CHUNK_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=CHUNK_SR)

    finally:
        if jitter_tmpfile and os.path.exists(jitter_tmpfile):
            os.remove(jitter_tmpfile)
        if os.path.exists(tmpwav):
            os.remove(tmpwav)  # WAV never accumulates on disk

    stem = _make_stem(midi_rel, soundfont.name)
    return process_audio(audio, bar_starts, stem,
                         args.out_path, args.chunks_per_song, args.pitch_range, args.overwrite)


def main():
    if args.single_soundfont is not None:
        soundfonts = find_soundfonts_single(args.single_soundfont)
        if not soundfonts:
            print('Single soundfont not found or invalid:', args.single_soundfont)
            sys.exit(1)
    else:
        soundfonts = find_soundfonts(args.soundfont_path)
        print(f'Loaded {len(soundfonts)} soundfonts from {args.soundfont_path}')

    if not soundfonts:
        print('No .sf2 soundfonts found in', args.soundfont_path)
        sys.exit(1)

    midi_base_path = Path(args.midi_path)
    all_midi = list(midi_base_path.rglob('*.mid'))
    random.shuffle(all_midi)
    target = args.target_count if args.target_count is not None else len(all_midi)
    print(f'{len(all_midi)} MIDI files found, target={target}\n')

    success = 0
    chunks_written = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=args.max_processes) as executor:
        futures_list = [
            executor.submit(render_and_chunk, m, soundfonts, midi_base_path)
            for m in all_midi
        ]
        for future in as_completed(futures_list):
            try:
                n = future.result()
                if n > 0:
                    success += 1
                    chunks_written += n
                    print(f'[{success}/{target}] +{n} chunks ({chunks_written} total)')
            except Exception as e:
                print(f'  error: {e}')
            if success >= target:
                for f in futures_list:
                    f.cancel()
                break

    elapsed = time.time() - t0
    print(f'\nDone: {success} songs, {chunks_written} chunks written in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
