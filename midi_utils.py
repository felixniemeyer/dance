"""Shared MIDI utilities used by render.py and chunk.py."""

import os
import random
import subprocess

import mido


# ── data classes ───────────────────────────────────────────────────────────────

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


# ── MIDI helpers ───────────────────────────────────────────────────────────────

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


def has_note_phase_alignment(
    note_ticks,
    time_signatures,
    ticks_per_beat,
    *,
    denominator=16,
    tolerance=0.035,
    min_ratio=0.6,
    min_notes=24,
):
    if len(note_ticks) < min_notes:
        return False, 0.0, len(note_ticks)
    if denominator <= 0:
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
        scaled = phase * denominator
        nearest_error = abs(scaled - round(scaled)) / denominator

        if nearest_error <= tolerance:
            aligned += 1
        total += 1

    if total == 0:
        return False, 0.0, len(note_ticks)

    ratio = aligned / total
    return ratio >= min_ratio, ratio, total


def max_silence_gap_seconds(note_ticks, tempo_changes, ticks_per_beat):
    """Return the longest gap between consecutive note-on events, in seconds."""
    if len(note_ticks) < 2:
        return 0.0
    note_times = ticks_to_seconds(note_ticks, tempo_changes, ticks_per_beat)
    return max(note_times[i + 1] - note_times[i] for i in range(len(note_times) - 1))


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
    return ticks_to_seconds(bar_ticks, tempo_changes, ticks_per_beat)


def jitter_drum_notes(mid, max_seconds):
    n_tracks = len(mid.tracks)
    midi_type = 0 if n_tracks == 1 else 1
    jittered = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=midi_type)
    max_ticks = int(max_seconds * 2 * mid.ticks_per_beat)

    for track in mid.tracks:
        absolute_messages = []
        tick = 0
        for msg in track:
            tick += msg.time
            absolute_messages.append([tick, msg.copy(time=0)])

        for message_data in absolute_messages:
            current_tick, msg = message_data
            if msg.type == 'note_on' and getattr(msg, 'channel', -1) == 9 and msg.velocity > 0:
                delta = int(random.uniform(-1, 1) * random.uniform(0, max_ticks))
                message_data[0] = max(0, current_tick + delta)

        absolute_messages.sort(key=lambda x: x[0])

        new_track = mido.MidiTrack()
        previous_tick = 0
        for absolute_tick, msg in absolute_messages:
            new_track.append(msg.copy(time=absolute_tick - previous_tick))
            previous_tick = absolute_tick
        jittered.tracks.append(new_track)

    return jittered


def render_wav(midifile, wavfile, soundfont, timeout_s, samplerate):
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


def find_soundfonts(soundfont_path):
    soundfonts = []
    for root, _, files in os.walk(soundfont_path):
        for f in files:
            if f.endswith('.sf2'):
                soundfonts.append(Soundfont(f[:-4], os.path.join(root, f)))
    return soundfonts


def find_soundfonts_single(sf2_path):
    sf_path = os.path.abspath(sf2_path)
    if not os.path.exists(sf_path):
        return []
    if not sf_path.endswith('.sf2'):
        return []
    return [Soundfont(os.path.basename(sf_path)[:-4], sf_path)]


def abbreviate(s):
    s = s.replace(' ', '').replace('-', '').ljust(6, '_')
    return s[:3] + s[-3:]
