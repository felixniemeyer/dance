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
# Provides a small buffer of future phase values beyond the chunk end.
phase_label_extra_seconds = 1.0

# ── CLI args ──────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description='Render MIDI + chop into training chunks in one pass.')
parser.add_argument('--midi-path', type=str, default='data/midi/lakh_clean')
parser.add_argument('--soundfont-path', type=str, default='data/soundfonts')
parser.add_argument('--single-soundfont', type=str, default=None,
    help='path to one .sf2 file to use instead of scanning --soundfont-path')
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
parser.add_argument('--render-timeout-seconds', type=float, default=None,
    help='hard timeout for one MIDI render call. If not set, timeout scales with song length.')
parser.add_argument('--phase-grid-denominator', type=int, default=16,
    help='phase alignment grid denominator (e.g. 16 checks 1/16 bar steps)')
parser.add_argument('--note-phase-tolerance', type=float, default=0.035,
    help='max phase error to count a note as grid-aligned')
parser.add_argument('--min-aligned-note-ratio', type=float, default=0.6,
    help='minimum fraction of note-on ticks that must align to grid')
parser.add_argument('--min-notes-for-alignment-check', type=int, default=24,
    help='minimum unique note-on ticks required for alignment check')
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

def read_note_on_ticks(mid):
    note_ticks = set()
    for track in mid.tracks:
        ticks = 0
        for msg in track:
            ticks += msg.time
            if msg.type == 'note_on' and getattr(msg, 'velocity', 0) > 0:
                note_ticks.add(ticks)
    return sorted(note_ticks)

def has_note_phase_alignment(note_ticks, time_signatures, ticks_per_beat):
    if len(note_ticks) < args.min_notes_for_alignment_check:
        return False, 0.0, len(note_ticks)
    if args.phase_grid_denominator <= 0:
        return False, 0.0, len(note_ticks)

    ts_index = 0
    aligned = 0
    total = 0

    for tick in note_ticks:
        while ts_index + 1 < len(time_signatures) and time_signatures[ts_index + 1].tick <= tick:
            ts_index += 1

        ts = time_signatures[ts_index]
        bar_ticks = int(ticks_per_beat * 4 * ts.numerator / ts.denominator)
        if bar_ticks <= 0:
            continue

        phase = ((tick - ts.tick) % bar_ticks) / bar_ticks
        scaled = phase * args.phase_grid_denominator
        nearest_error = abs(scaled - round(scaled)) / args.phase_grid_denominator

        if nearest_error <= args.note_phase_tolerance:
            aligned += 1
        total += 1

    if total == 0:
        return False, 0.0, len(note_ticks)

    ratio = aligned / total
    return ratio >= args.min_aligned_note_ratio, ratio, total


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
    n_tracks = len(mid.tracks)
    midi_type = 0 if n_tracks == 1 else 1
    jittered = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=midi_type)
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
    fluidsynth_cmd = [
        'fluidsynth', '--no-shell',
        '--fast-render', wavfile,
        '--sample-rate', str(samplerate),
        soundfont.path, midifile,
    ]
    with subprocess.Popen(fluidsynth_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as p:
        try:
            p.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            p.kill()
            p.communicate()
            print('  FluidSynth render timed out; skipping')
            return False

    if p.returncode != 0 or not os.path.exists(wavfile):
        print(f'  FluidSynth failed with code {p.returncode}; skipping')
        return False

    return True

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
        timeout_s = args.render_timeout_seconds if args.render_timeout_seconds is not None else max(120, duration * 10)
        ok = render_wav(render_midi_path, wavfile, soundfont, timeout_s)
        if not ok or not os.path.exists(wavfile):
            return 0

        info = soundfile.info(wavfile)
        sample_rate = info.samplerate
        song_length = info.frames  # total samples

        # Keep chunk placement math equivalent to chop.py: placement depends on
        # audio chunk duration only. Extra label horizon must not change scale.
        chunk_len_audio = int(chunk_duration * sample_rate)
        n_chunks = int(song_length / (chunk_len_audio * (2 - args.min_pitch)) * 0.9 - 1)
        if n_chunks < 1:
            return 0

        # random pitch factors for each chunk
        pitches = [
            args.min_pitch + random.random() * 2 * (1 - args.min_pitch)
            for _ in range(n_chunks)
        ]
        used = sum(math.ceil(chunk_len_audio * p) for p in pitches)
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
            offset = s + int(chunk_len_audio * pitches[i])

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
                # fi index is on chunk output timeline; map back to source time.
                local_time = fi * frame_size / sample_rate
                orig_t = start_time + local_time * pitch
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
    if args.single_soundfont is not None:
        sf_path = os.path.abspath(args.single_soundfont)
        if not os.path.exists(sf_path):
            print('Single soundfont not found:', sf_path)
            return []
        if not sf_path.endswith('.sf2'):
            print('Single soundfont must be .sf2:', sf_path)
            return []
        return [Soundfont(os.path.basename(sf_path)[:-4], sf_path)]

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
            note_ticks = read_note_on_ticks(mid)
            aligned_ok, aligned_ratio, aligned_count = has_note_phase_alignment(
                note_ticks, time_signatures, mid.ticks_per_beat
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

        n = process(str(midifile), sf, bar_starts, midi_rel, mid.length)
        if n > 0:
            success += 1
            total_chunks += n

    elapsed = time.time() - t0
    print(f'\nDone: {success} songs processed, {total_chunks} chunks written in {elapsed:.1f}s')


if __name__ == '__main__':
    main()
