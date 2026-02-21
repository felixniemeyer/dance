"""
Renders MIDI files and chops them into training chunks in one pass.

Instead of:
  fluidsynth → .wav → ffmpeg(loudnorm) → .ogg → ffmpeg(pitch+trim) → chunk.ogg

This does:
  fluidsynth → tmp.wav → ffmpeg(pitch+trim+limit) → chunk.ogg → delete tmp.wav

Usage:
    python render_and_chop.py \
        --midi-path midi_files/lmd_matched \
        --soundfont-path soundfonts \
        --out-path data/chunks/v1 \
        --target-count 300 \
        --max-processes 6
"""

import math
import os
import random
import subprocess
import sys
import tempfile
import threading
import time
import argparse
from bisect import bisect_right
from pathlib import Path

import mido
import soundfile

from config import chunk_duration, frame_size, samplerate

# Extra seconds of phase labels written beyond the audio chunk duration.
# Ensures the anticipation offset can always find a valid target label,
# even for frames near the end of a chunk.
phase_label_extra_seconds = 1

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Render MIDI + chop into training chunks in one pass.')
parser.add_argument('--midi-path', type=str, default='data/midi/lakh_clean')
parser.add_argument('--soundfont-path', type=str, default='data/soundfonts')
parser.add_argument('--out-path', type=str, required=True)
parser.add_argument('--target-count', type=int, default=None,
    help='stop after this many successfully processed songs')
parser.add_argument('--max-processes', type=int, default=6,
    help='max concurrent ffmpeg chunk-writing processes')
parser.add_argument('--max-song-length', type=int, default=60 * 20,
    help='skip songs longer than this many seconds')
parser.add_argument('--min-pitch', type=float, default=0.71,
    help='minimum pitch factor for chunk augmentation')
parser.add_argument('--volume', type=float, default=2.0,
    help='level_in for alimiter (pre-limiting gain)')
parser.add_argument('--midi-jitter-max-seconds', type=float, default=0.05,
    help='max absolute jitter applied to drum note_on events in seconds. 0 disables jitter.')
parser.add_argument('--overwrite', default=False, action='store_true')
args = parser.parse_args()

os.makedirs(args.out_path, exist_ok=True)

# Semaphore limits how many ffmpeg chunk processes run concurrently.
_semaphore = threading.Semaphore(args.max_processes)

# ── data classes ──────────────────────────────────────────────────────────────

class Soundfont:
    def __init__(self, name, path):
        self.name = name
        self.path = path

class TempoChange:
    def __init__(self, tick, tempo):
        self.tick = tick
        self.tempo = tempo

class TimeSignatureChange:
    def __init__(self, tick, numerator, denominator):
        self.tick = tick
        self.numerator = numerator
        self.denominator = denominator

# ── MIDI helpers (from render_midi.py) ────────────────────────────────────────

def read_events(mid):
    tempo_changes, time_signatures = [], []
    max_tick = 0
    for track in mid.tracks:
        ticks = 0
        for msg in track:
            ticks += msg.time
            max_tick = max(max_tick, ticks)
            if msg.type == 'set_tempo':
                tempo_changes.append(TempoChange(ticks, msg.tempo))
            if msg.type == 'time_signature':
                time_signatures.append(TimeSignatureChange(ticks, msg.numerator, msg.denominator))
    tempo_changes.sort(key=lambda x: x.tick)
    time_signatures.sort(key=lambda x: x.tick)
    # deduplicate time signatures at the same tick (keep last)
    deduped = []
    for ts in time_signatures:
        if deduped and deduped[-1].tick == ts.tick:
            deduped[-1] = ts
        else:
            deduped.append(ts)
    return tempo_changes, deduped, max_tick


def has_valid_time_signature(time_signatures):
    if not time_signatures or time_signatures[0].tick != 0:
        return False
    for ts in time_signatures:
        if ts.numerator <= 0 or ts.denominator not in [1, 2, 4, 8, 16, 32]:
            return False
    return True


def ticks_to_seconds(ticks, tempo_changes, ticks_per_beat):
    if not ticks:
        return []
    if not tempo_changes or tempo_changes[0].tick != 0:
        tempo_changes = [TempoChange(0, 500000)] + tempo_changes
    results, ti, prev_tick, acc = [], 0, tempo_changes[0].tick, 0.0
    for target in sorted(ticks):
        while ti + 1 < len(tempo_changes) and tempo_changes[ti + 1].tick <= target:
            nxt = tempo_changes[ti + 1]
            acc += mido.tick2second(nxt.tick - prev_tick, ticks_per_beat, tempo_changes[ti].tempo)
            prev_tick, ti = nxt.tick, ti + 1
        results.append(acc + mido.tick2second(target - prev_tick, ticks_per_beat, tempo_changes[ti].tempo))
    return results


def calc_bar_starts_in_seconds(time_signatures, tempo_changes, ticks_per_beat, max_tick):
    if not time_signatures:
        return []
    extended = list(time_signatures) + [TimeSignatureChange(max_tick + ticks_per_beat * 16, 4, 4)]
    bar_ticks = []
    for idx, ts in enumerate(extended[:-1]):
        next_tick = extended[idx + 1].tick
        bar_len = int(ticks_per_beat * 4 * ts.numerator / ts.denominator)
        if bar_len <= 0:
            continue
        tick = ts.tick
        while tick < next_tick:
            if not bar_ticks or bar_ticks[-1] != tick:
                bar_ticks.append(tick)
            tick += bar_len
    if len(bar_ticks) < 2:
        return []
    last_ts = extended[-2]
    last_bar_len = int(ticks_per_beat * 4 * last_ts.numerator / last_ts.denominator)
    if last_bar_len > 0:
        bar_ticks.append(bar_ticks[-1] + last_bar_len)
    # treat the first bar start as the phase origin (same as render_midi.py)
    return ticks_to_seconds(bar_ticks, tempo_changes, ticks_per_beat)

# ── phase helper (from chop.py) ───────────────────────────────────────────────

def calc_phase_at_time(t, bars):
    idx = bisect_right(bars, t) - 1
    if idx < 0 or idx + 1 >= len(bars):
        return None
    bar_start, bar_end = bars[idx], bars[idx + 1]
    if bar_end <= bar_start:
        return None
    phase = (t - bar_start) / (bar_end - bar_start)
    if phase < 0:
        return None
    return phase % 1.0 if phase >= 1.0 else phase

# ── filename helper (from chop.py) ────────────────────────────────────────────

def abbreviate(s):
    s = s.replace(' ', '').replace('-', '').ljust(6, '_')
    return s[:3] + s[-3:]

# ── MIDI jitter (from render_midi.py) ─────────────────────────────────────────

def jitter_drum_notes(mid, max_seconds):
    jittered = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    max_ticks = int(max_seconds * 2 * mid.ticks_per_beat)

    for track in mid.tracks:
        # convert to absolute ticks
        absolute_messages = []
        tick = 0
        for msg in track:
            tick += msg.time
            absolute_messages.append([tick, msg.copy(time=0)])

        # jitter drum note_on events; clamp to non-negative absolute tick
        for message_data in absolute_messages:
            current_tick, msg = message_data
            if msg.type == 'note_on' and getattr(msg, 'channel', -1) == 9 and msg.velocity > 0:
                delta = int(random.uniform(-1, 1) * random.uniform(0, max_ticks))
                message_data[0] = max(0, current_tick + delta)

        # sort by absolute tick so all delta times remain non-negative
        absolute_messages.sort(key=lambda x: x[0])

        new_track = mido.MidiTrack()
        previous_tick = 0
        for absolute_tick, msg in absolute_messages:
            new_track.append(msg.copy(time=absolute_tick - previous_tick))
            previous_tick = absolute_tick
        jittered.tracks.append(new_track)

    return jittered

# ── render ────────────────────────────────────────────────────────────────────

def render_wav(midifile, wavfile, soundfont, timeout_s):
    cmd = [
        'fluidsynth', '--no-shell',
        '--fast-render', wavfile,
        '--sample-rate', str(samplerate),
        soundfont.path, midifile,
    ]
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        try:
            p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            print('  Render timed out')
            return False
    return p.returncode == 0

# ── chunk writer (runs in thread) ─────────────────────────────────────────────

def _write_chunk_thread(wavfile, outfile, start_time, pitch, sample_rate):
    try:
        newrate = sample_rate * pitch
        # Same filter chain as chop.py, minus loudnorm (wav is already clean;
        # alimiter handles level control).
        af = (
            f'asetrate={newrate},aresample={sample_rate}'
            f',atrim=start={start_time / pitch}:end={start_time / pitch + chunk_duration + 0.1}'
            f',asetpts=N/SR/TB'
            f',alimiter=level_in={args.volume}'
        )
        cmd = [
            'ffmpeg', '-v', 'warning',
            '-y' if args.overwrite else '-n',
            '-i', wavfile,
            '-ac', '1',
            '-af', af,
            '-t', str(chunk_duration),
            '-c:a', 'libvorbis', '-qscale:a', '6',
            '-ar', str(sample_rate),
            outfile + '.ogg',
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    except Exception as e:
        print(f'  ffmpeg error: {e}')
        for ext in ('.phase', '.ogg'):
            if os.path.exists(outfile + ext):
                os.remove(outfile + ext)
    finally:
        _semaphore.release()

# ── process one (midi, soundfont) pair ────────────────────────────────────────

def process(midifile_str, soundfont, bar_starts, midi_rel, duration):
    fd, wavfile = tempfile.mkstemp(suffix='.wav')
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
        ok = render_wav(render_midi_path, wavfile, soundfont, max(120, duration * 10))
        if not ok or not os.path.exists(wavfile):
            return 0

        info = soundfile.info(wavfile)
        sample_rate = info.samplerate
        song_length = info.frames  # total samples

        chunk_len = int((chunk_duration + phase_label_extra_seconds) * sample_rate)
        n_chunks = max(2, int(song_length / (chunk_len * (2 - args.min_pitch)) * 0.9))

        # random pitch factors for each chunk
        pitches = [
            args.min_pitch + random.random() * 2 * (1 - args.min_pitch)
            for _ in range(n_chunks)
        ]
        used = sum(math.ceil(chunk_len * p) for p in pitches)
        free = song_length - used
        if free <= 0:
            return 0

        # distribute chunk start positions (same algorithm as chop.py)
        spots = sorted(int(random.random() * free) for _ in range(n_chunks))
        starts, offset = [], 0
        for i in range(n_chunks - 1):
            delta = spots[i + 1] - spots[i]
            s = delta + offset
            starts.append(s)
            offset = s + int(chunk_len * pitches[i])

        rel_flat = midi_rel.replace(os.sep, '_')
        prefix = abbreviate(rel_flat) + '-' + abbreviate(soundfont.name) + '-'
        digits = max(1, math.floor(math.log10(max(n_chunks, 2))) + 1)

        chunk_threads = []
        written = 0
        for i, start in enumerate(starts):
            pitch = pitches[i]
            start_time = start / sample_rate

            frame_count = int((chunk_duration + phase_label_extra_seconds) * sample_rate / frame_size)
            phases, valid = [], True
            for fi in range(frame_count):
                orig_t = start_time + fi * frame_size / sample_rate * pitch
                phase = calc_phase_at_time(orig_t, bar_starts)
                if phase is None:
                    valid = False
                    break
                phases.append(phase)
            if not valid:
                continue

            outfile = os.path.join(args.out_path, prefix + str(i).zfill(digits))
            if not args.overwrite and os.path.exists(outfile + '.ogg'):
                continue

            with open(outfile + '.phase', 'w', encoding='utf8') as f:
                f.writelines(f'{ph:.6f}\n' for ph in phases)

            _semaphore.acquire()
            t = threading.Thread(
                target=_write_chunk_thread,
                args=(wavfile, outfile, start_time, pitch, sample_rate),
                daemon=True,
            )
            t.start()
            chunk_threads.append(t)
            written += 1

        for t in chunk_threads:
            t.join()

        print(f'  → {written} chunks written')
        return written

    finally:
        if jitter_tmpfile and os.path.exists(jitter_tmpfile):
            os.remove(jitter_tmpfile)
        if os.path.exists(wavfile):
            os.remove(wavfile)

# ── soundfont loader ──────────────────────────────────────────────────────────

def find_soundfonts():
    soundfonts = []
    for root, _, files in os.walk(args.soundfont_path):
        for f in files:
            if f.endswith('.sf2'):
                soundfonts.append(Soundfont(f[:-4], os.path.join(root, f)))
    print(f'Loaded {len(soundfonts)} soundfonts from {args.soundfont_path}')
    return soundfonts

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    soundfonts = find_soundfonts()
    if not soundfonts:
        print('No .sf2 soundfonts found in', args.soundfont_path)
        sys.exit(1)

    all_midi = list(Path(args.midi_path).rglob('*.mid'))
    random.shuffle(all_midi)
    target = args.target_count if args.target_count is not None else len(all_midi)
    print(f'{len(all_midi)} MIDI files found, target={target}\n')

    success, attempt, total_chunks = 0, 0, 0
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
            bar_starts = calc_bar_starts_in_seconds(
                time_signatures, tempo_changes, mid.ticks_per_beat, max_tick)
            if len(bar_starts) < 2:
                print('  Bar extraction failed, skipping')
                continue
        except Exception as e:
            print(f'  MIDI parse error: {e}')
            continue

        n = process(str(midifile), sf, bar_starts, midi_rel, mid.length)
        success += 1
        total_chunks += n

    elapsed = time.time() - t0
    print(f'\nDone: {success} songs processed, {total_chunks} chunks written in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
