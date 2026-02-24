"""
Render MIDI files to high-quality WAV + bar_starts.json.

Usage:
    python render.py \
        --midi-path data/midi/lakh_clean \
        --soundfont-path data/soundfonts \
        --out-path data/rendered/v2 \
        --target-count 500 \
        --max-processes 6
"""

import argparse
import json
import os
import random
import sys
import tempfile
import time
from pathlib import Path

import mido

from config import render_samplerate
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

parser = argparse.ArgumentParser(description='Render MIDI files to WAV + bar_starts.json.')
parser.add_argument('--midi-path', type=str, default='data/midi/lakh_clean')
parser.add_argument('--soundfont-path', type=str, default='data/soundfonts')
parser.add_argument('--single-soundfont', type=str, default=None,
    help='path to one .sf2 file to use instead of scanning --soundfont-path')
parser.add_argument('--out-path', type=str, required=True)
parser.add_argument('--target-count', type=int, default=None,
    help='stop after this many successfully processed songs')
parser.add_argument('--max-song-length', type=int, default=60 * 20,
    help='skip songs longer than this many seconds')
parser.add_argument('--midi-jitter-max-seconds', type=float, default=0.05,
    help='max jitter on drum note_on events in seconds. 0 disables jitter.')
parser.add_argument('--overwrite', default=False, action='store_true')
parser.add_argument('--render-timeout-seconds', type=float, default=None,
    help='hard timeout for one MIDI render call. Defaults to max(120, duration*10).')
parser.add_argument('--phase-grid-denominator', type=int, default=16)
parser.add_argument('--note-phase-tolerance', type=float, default=0.035)
parser.add_argument('--min-aligned-note-ratio', type=float, default=0.6)
parser.add_argument('--min-notes-for-alignment-check', type=int, default=24)
args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)


def _make_stem(midi_rel, soundfont_name):
    """Build an output stem from MIDI path and soundfont name."""
    midi_part = midi_rel.replace(os.sep, '_').replace('/', '_')
    sf_part = soundfont_name[:30].replace(' ', '_')
    return f'{midi_part}__{sf_part}'


def process_song(midifile_str, soundfont, midi_rel, duration, bar_starts):
    """Render one (MIDI, soundfont) pair to WAV + JSON. Returns True on success."""
    stem = _make_stem(midi_rel, soundfont.name)
    wav_path = os.path.join(args.out_path, stem + '.wav')
    json_path = os.path.join(args.out_path, stem + '.json')

    if not args.overwrite and os.path.exists(json_path) and os.path.exists(wav_path):
        print('  Already exists, skipping')
        return True

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
            return False

        # Move WAV to final location
        if args.overwrite and os.path.exists(wav_path):
            os.remove(wav_path)
        os.rename(tmpwav, wav_path)
        tmpwav = None  # prevent deletion in finally

        # Write JSON
        meta = {
            'bar_starts': bar_starts,
            'source_midi': midifile_str,
            'soundfont': soundfont.path,
        }
        with open(json_path, 'w', encoding='utf8') as f:
            json.dump(meta, f)

        return True

    finally:
        if jitter_tmpfile and os.path.exists(jitter_tmpfile):
            os.remove(jitter_tmpfile)
        if tmpwav and os.path.exists(tmpwav):
            os.remove(tmpwav)


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

    all_midi = list(Path(args.midi_path).rglob('*.mid'))
    random.shuffle(all_midi)
    target = args.target_count if args.target_count is not None else len(all_midi)
    print(f'{len(all_midi)} MIDI files found, target={target}\n')

    success, attempt = 0, 0
    t0 = time.time()

    for midifile in all_midi:
        if success >= target:
            break
        attempt += 1
        midi_rel = str(midifile.relative_to(args.midi_path).with_suffix(''))
        sf = random.choice(soundfonts)
        print(f'[{success+1}/{target}] attempt={attempt}  {midi_rel}  sf={sf.name[:40]}')

        try:
            mid = mido.MidiFile(str(midifile))
            if mid.length > args.max_song_length:
                print(f'  Too long ({mid.length:.0f}s), skipping')
                continue
            tempo_changes, time_signatures, max_tick = read_events(mid)
            if not has_valid_time_signature(time_signatures):
                print('  No valid time signature, skipping')
                continue
            note_ticks = read_note_on_ticks(mid)
            aligned_ok, aligned_ratio, aligned_count = has_note_phase_alignment(
                note_ticks, time_signatures, mid.ticks_per_beat,
                denominator=args.phase_grid_denominator,
                tolerance=args.note_phase_tolerance,
                min_ratio=args.min_aligned_note_ratio,
                min_notes=args.min_notes_for_alignment_check,
            )
            if not aligned_ok:
                print(f'  Poor note/grid alignment ({aligned_ratio:.2f} over {aligned_count} notes), skipping')
                continue
            bar_starts = calc_bar_starts_in_seconds(
                time_signatures, tempo_changes, mid.ticks_per_beat, max_tick)
            if len(bar_starts) < 2:
                print('  Bar extraction failed, skipping')
                continue
        except Exception as e:
            print(f'  MIDI parse error: {e}')
            continue

        ok = process_song(str(midifile), sf, midi_rel, mid.length, bar_starts)
        if ok:
            success += 1

    elapsed = time.time() - t0
    print(f'\nDone: {success} songs rendered in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
