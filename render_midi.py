"""
renders midi files to wav while exporting relevant audio events 
"""
import os
import sys
import subprocess
import time
import threading
import argparse
import random

import mido

from config import channels, samplerate

parser = argparse.ArgumentParser(description='Renders midi songs with soundfonts.')
parser.add_argument('--midi-path', type=str, default='data/midi/lakh_clean',
    help='path to the directory containing midi files. The following directory structure is assumed: midi-path/<artist>/<song-name>.mid')
parser.add_argument('--single-file', type=str, default=None,
    help='path to a single midi file instead of --midi-path')

parser.add_argument('--soundfont-path', type=str, default='data/soundfonts',
    help='path to the soundfonts')
parser.add_argument('--single-soundfont', type=str, default=None,
    help='path to a single soundfont instead of --soundfont-path')
parser.add_argument('--out-path', type=str, help='path to the output directory')

parser.add_argument('--bits', type=int, default=16,
    help='bits per sample')
parser.add_argument('--max-song-length', type=int, default=60 * 20,
    help='max song length in seconds')

parser.add_argument('--target-count', type=int, default=None,
    help='stop after rendering this many songs (randomly sampled). Default: all.')
parser.add_argument('--overwrite', default=False, action='store_true',
    help='overwrite existing files')
parser.add_argument('--max-processes', type=int, default=9,
    help='max processes to use')
parser.add_argument('--midi-jitter-max-seconds', type=float, default=0.0,
    help='max absolute jitter for drum note_on events in seconds. 0 disables jitter.')

# parse arguments
args = parser.parse_args()

if args.out_path is None:
    if args.single_file is not None:
        # get filename
        filename = os.path.basename(args.single_file)
        # remove extension (unknown length after dot)
        extension_start = filename.rfind('.')
        if extension_start != -1:
            filename = filename[0:extension_start]
        args.out_path = 'data/rendered_midi/single_songs/' + filename
    else:
        args.out_path = 'data/rendered_midi/' + os.path.basename(args.midi_path)

# mkdir out-path if it doesn't exist
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)

class Soundfont:
    """
    represents a soundfont
    """
    def __init__(self, name, path):
        self.name = name
        self.path = path

class TempoChange:
    def __init__(self, tick, tempo):
        self.tick = tick
        self.tempo = tempo

    def __str__(self):
        return f'tick: {self.tick}, tempo: {self.tempo}'

    def __repr__(self):
        return str(self)

class TimeSignatureChange:
    def __init__(self, tick, numerator, denominator):
        self.tick = tick
        self.numerator = numerator
        self.denominator = denominator

start_time = time.time()
def generate_song(midifile, outpath, soundfonts, threads):
    try:
        mid = mido.MidiFile(midifile)
        render_midifile = midifile
        if args.midi_jitter_max_seconds > 0:
            mid = jitter_drum_notes(mid, args.midi_jitter_max_seconds)
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)
            render_midifile = os.path.join(outpath, '__jittered.mid')
            mid.save(render_midifile)

        duration = mid.length

        print(f'Song length: {duration:.2f} seconds')
        if duration > args.max_song_length:
            print('Skipping song because it is too long: ' + str(mid.length))
            return False

        print(f'Track count: {len(mid.tracks)}')

        ticks_per_beat = mid.ticks_per_beat
        print('ticks per beat: ' + str(ticks_per_beat))

        tempo_changes, time_signatures, max_tick = read_events(mid)

        if not has_valid_time_signature(time_signatures):
            print('Skipping ' + midifile + ' because it has no usable time signature data')
            return False

        if not os.path.exists(outpath):
            os.makedirs(outpath, exist_ok=True)

        bar_starts = calc_bar_starts_in_seconds(
            time_signatures,
            tempo_changes,
            ticks_per_beat,
            max_tick,
            0,
        )
        if len(bar_starts) < 2:
            print('Skipping ' + midifile + ' because bar extraction failed')
            return False

        write_bar_starts(bar_starts, outpath)

        for soundfont in soundfonts:
            outfile_without_ext = os.path.join(outpath, soundfont.name)
            thread = threading.Thread(target=render_and_convert, args=(render_midifile, outfile_without_ext, soundfont, duration + 60))
            threads.append(thread)
            thread.start()
            print(f'Starting thread {len(threads)}/{args.max_processes}')

            while len(threads) >= args.max_processes:
                threads[0].join()
                threads.pop(0)

        return True

    except Exception as error:
        print('Error processing ' + midifile + ': ' + str(error))
        return False

def read_events(mid):
    tempo_changes = []
    time_signatures = []
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

    # sort everything because it may come from different tracks
    tempo_changes.sort(key=lambda x: x.tick)
    time_signatures.sort(key=lambda x: x.tick)

    deduped_time_signatures = []
    for ts in time_signatures:
        if len(deduped_time_signatures) == 0 or deduped_time_signatures[-1].tick != ts.tick:
            deduped_time_signatures.append(ts)
        else:
            deduped_time_signatures[-1] = ts

    return tempo_changes, deduped_time_signatures, max_tick

def write_bar_starts(bar_starts, outpath):
    print('Writing bar starts to file')
    bars_file_path = os.path.join(outpath, 'bars.csv')
    with open(bars_file_path, 'w', encoding='utf-8') as bars_file:
        for bar_start in bar_starts:
            bars_file.write(f'{bar_start:.6f}\n')

def has_valid_time_signature(time_signatures):
    if len(time_signatures) == 0:
        return False
    if time_signatures[0].tick != 0:
        return False
    for ts in time_signatures:
        if ts.numerator <= 0:
            return False
        if ts.denominator not in [1, 2, 4, 8, 16, 32]:
            return False
    return True

def calc_bar_starts_in_seconds(time_signatures, tempo_changes, ticks_per_beat, max_tick, first_note_tick):
    bar_starts_in_ticks = calc_bar_starts_in_ticks(time_signatures, ticks_per_beat, max_tick)
    if len(bar_starts_in_ticks) < 2:
        return []

    # Treat the first full bar at or after the first drum note as phase origin.
    origin_tick = bar_starts_in_ticks[0]
    for tick in bar_starts_in_ticks:
        if tick >= first_note_tick:
            origin_tick = tick
            break

    bar_starts_in_ticks = [tick for tick in bar_starts_in_ticks if tick >= origin_tick]
    if len(bar_starts_in_ticks) < 2:
        return []

    return ticks_to_seconds(bar_starts_in_ticks, tempo_changes, ticks_per_beat)

def calc_bar_starts_in_ticks(time_signatures, ticks_per_beat, max_tick):
    bar_starts = []
    if len(time_signatures) == 0:
        return bar_starts

    extended_time_signatures = list(time_signatures)
    extended_time_signatures.append(TimeSignatureChange(max_tick + ticks_per_beat * 16, 4, 4))

    for idx, ts in enumerate(extended_time_signatures[:-1]):
        next_tick = extended_time_signatures[idx + 1].tick
        bar_ticks = int(ticks_per_beat * 4 * ts.numerator / ts.denominator)
        if bar_ticks <= 0:
            continue

        tick = ts.tick
        while tick < next_tick:
            if len(bar_starts) == 0 or bar_starts[-1] != tick:
                bar_starts.append(tick)
            tick += bar_ticks

    # Ensure one extra bar start so phase interpolation has an end bound.
    if len(bar_starts) > 0:
        last_ts = extended_time_signatures[-2]
        last_bar_ticks = int(ticks_per_beat * 4 * last_ts.numerator / last_ts.denominator)
        if last_bar_ticks > 0:
            bar_starts.append(bar_starts[-1] + last_bar_ticks)

    return bar_starts

def ticks_to_seconds(ticks, tempo_changes, ticks_per_beat):
    if len(ticks) == 0:
        return []

    sorted_ticks = sorted(ticks)
    if len(tempo_changes) == 0 or tempo_changes[0].tick != 0:
        tempo_changes = [TempoChange(0, 500000)] + tempo_changes

    results = []
    current_tempo_index = 0
    current_tempo = tempo_changes[current_tempo_index].tempo
    previous_tempo_tick = tempo_changes[current_tempo_index].tick
    accumulated_seconds = 0.0

    for target_tick in sorted_ticks:
        while (current_tempo_index + 1) < len(tempo_changes) and tempo_changes[current_tempo_index + 1].tick <= target_tick:
            next_change = tempo_changes[current_tempo_index + 1]
            accumulated_seconds += mido.tick2second(next_change.tick - previous_tempo_tick, ticks_per_beat, current_tempo)
            previous_tempo_tick = next_change.tick
            current_tempo = next_change.tempo
            current_tempo_index += 1

        seconds = accumulated_seconds + mido.tick2second(target_tick - previous_tempo_tick, ticks_per_beat, current_tempo)
        results.append(seconds)

    return results

def jitter_drum_notes(mid, max_seconds):
    if max_seconds <= 0:
        return mid

    jittered = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat, type=mid.type)
    max_ticks = int(max_seconds * 2 * mid.ticks_per_beat)

    for track in mid.tracks:
        absolute_messages = []
        tick = 0
        for msg in track:
            tick += msg.time
            absolute_messages.append([tick, msg.copy(time=0)])

        previous_tick = 0
        for message_data in absolute_messages:
            current_tick, msg = message_data
            if msg.type == 'note_on' and getattr(msg, 'channel', -1) == 9 and msg.velocity > 0:
                delta = int(random.uniform(-1, 1) * random.uniform(0, max_ticks))
                current_tick = max(previous_tick, current_tick + delta)
                message_data[0] = current_tick
            previous_tick = current_tick

        new_track = mido.MidiTrack()
        previous_tick = 0
        for absolute_tick, msg in absolute_messages:
            delta = absolute_tick - previous_tick
            previous_tick = absolute_tick
            new_track.append(msg.copy(time=delta))
        jittered.tracks.append(new_track)

    return jittered

class RenderProcessException(Exception):
    pass

def render_and_convert(midifile, outfile_without_ext, soundfont, max_duration):
    # check if file already exists
    wavfile = outfile_without_ext + '.wav'
    oggfile = outfile_without_ext + '.ogg'

    if not args.overwrite and os.path.exists(outfile_without_ext + '.ogg'):
        print(f'Skipping because {wavfile} already exists')
        return 1

    ok = render(midifile, wavfile, soundfont)

    if ok:
        convert(midifile, wavfile, oggfile, max_duration)

    os.remove(wavfile)

    return 0


def render(midifile, wavfile, soundfont):
    renderCommand = [
        'fluidsynth',
        '--no-shell',
        '--fast-render', wavfile,
        '--sample-rate', str(samplerate),
        soundfont.path,
        midifile,
    ]

    print(f"{' '.join(renderCommand)}")
    with subprocess.Popen(
        renderCommand,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        ) as renderProcess:

        render_stdout, render_stderr = renderProcess.communicate()
        print(f'\nRendering terminated for {midifile}')
        if renderProcess.returncode != 0:
            print('Process returned non-zero exit code: ' + str(renderProcess.returncode))
        render_output = render_stdout.decode('utf-8')
        if render_output != '':
            print('Render output: ' + render_output)
        render_errors = render_stderr.decode('utf-8')
        if render_errors != '':
            print('Render errors: ' + render_errors)

    return renderProcess.returncode == 0


def convert(midifile, wavfile, oggfile, max_duration):
    convertCommand = [
        'ffmpeg',
        '-i', wavfile, 
        '-y' if args.overwrite else '-n',
        '-t', str(max_duration),
        '-ac', str(channels),
        '-af', 'loudnorm',
        '-c:a', 'libvorbis',
        '-qscale:a', '6',
        '-ar', str(samplerate),
        oggfile
    ]

    print(f"{' '.join(convertCommand)}")
    with subprocess.Popen(
        convertCommand,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        ) as convertProcess:

        convert_stdout, convert_stderr = convertProcess.communicate()
        print(f'\nConversion terminated for {midifile}')
        if convertProcess.returncode != 0:
            print('Process returned non-zero exit code: ' + str(convertProcess.returncode))
        convert_output = convert_stdout.decode('utf-8')
        if convert_output != '':
            print('Convert output' + convert_output)
        convert_errors = convert_stderr.decode('utf-8')
        if convert_errors != '':
            print('Convert errors: ' + convert_errors)

def main():
    soundfonts = find_soundfonts()
    threads = []

    if args.single_file is not None:
        print('Rendering single song' + args.single_file)
        generate_song(args.single_file, args.out_path, soundfonts, threads)
        sys.exit(0)
    else:
        from pathlib import Path
        all_midi = list(Path(args.midi_path).rglob('*.mid'))
        random.shuffle(all_midi)

        target = args.target_count if args.target_count is not None else len(all_midi)
        print(f'Rendering songs from {args.midi_path}  ({len(all_midi)} found, target={target})\n')

        success_count = 0
        attempt_count = 0
        for midifile in all_midi:
            if success_count >= target:
                break

            rel = midifile.relative_to(args.midi_path).with_suffix('')
            outpath = os.path.join(args.out_path, str(rel)) + '/'

            soundfont = random.choice(soundfonts)
            attempt_count += 1
            print(f"[{success_count + 1}/{target}] (attempt {attempt_count}) {rel}  sf={soundfont.name[:40]}")

            ok = generate_song(str(midifile), outpath, [soundfont], threads)
            if ok:
                success_count += 1

    # wait for all threads to finish
    for thread in threads:
        thread.join()

    print('Finished in ' + str(time.time() - start_time) + ' seconds')


def find_soundfonts():
    # read all available soundfonts in the soundfonts folder
    # find all .sf2 and .sf3 files in the soundfonts folder (recursively)
    soundfonts = []
    if args.single_soundfont is not None:
        soundfont_path = os.path.abspath(args.single_soundfont)
        soundfonts.append(Soundfont(os.path.basename(soundfont_path), soundfont_path))
    else:
        print('Loading soundfonts from ' + args.soundfont_path)
        for root, _dirs, files in os.walk(args.soundfont_path):
            for file in files:
                if file.endswith('.sf2'):  # timidity does not handle sf3 well
                    path = os.path.join(root, file)
                    soundfonts.append(Soundfont(file[:-4], path))
        print(f'Loaded {len(soundfonts)} soundfonts from {args.soundfont_path}')
    return soundfonts


if __name__ == '__main__':
    main()
