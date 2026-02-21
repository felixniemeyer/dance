"""
This script loads files from the out directory and chops them into 10s long files for the training. 
It also reads event files and ouputs them fitting to the chopped files.
"""

import math
import sys

import os
import random

import argparse
from bisect import bisect_right

import threading
import subprocess

import soundfile

from config import chunk_duration, frame_size, samplerate

parser = argparse.ArgumentParser(description='Chop up the songs into chunks.')

parser.add_argument('--in-path', type=str, default='data/rendered_midi/lakh_clean', help='path to the rendered audio. The directory structure is expected to be: <in-path>/<artist>/ and contain a folder <song> with n .ogg files along with one events.csv file')
parser.add_argument('--out-path', type=str, help='path to the output directory')
parser.add_argument('--sample-rate', type=int, default=samplerate, help='sample rate')
parser.add_argument('--min-pitch', type=float, default=0.71, help='will choose random pitches between this and 1')
parser.add_argument('--volume', type=float, default=2, help='output level before limiter')

parser.add_argument('--dry-run', default=False, help='dry run')

parser.add_argument('--max-processes', type=int, default=9, help='max processes to use')

# parse arguments
args = parser.parse_args()

chunk_length = chunk_duration * args.sample_rate

threads = []

# if in-path has trailing slash, remove it
if args.in_path[-1] == '/':
    args.in_path = args.in_path[:-1]

if args.out_path is None:
    args.out_path = 'data/chunks/' + os.path.basename(args.in_path)

# mk out-path if it doesn't exist
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)

def abbreviate(string):
    """
    Abbreviates a string to 6 characters.
    """
    # replace all whitespace with underscores
    string = string.replace(' ', '')
    string = string.replace('-', '')
    string = string.ljust(6, '_')
    return string[0:3] + string[-3:]

def call_ffmpeg(command, outfile):
    """
    Calls ffmpeg with the given command and outfile.
    """
    print(' '.join(ffmpeg_call))
    try:
        subprocess.check_call(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except KeyboardInterrupt:
        print('\nAborting')
        sys.exit(0)
    except Exception as error:
        print('ffmpeg failed. Rolling back.', error)
        for ext in ('.phase', '.ogg'):
            if os.path.exists(outfile + ext):
                os.remove(outfile + ext)

def read_bar_starts(file_path):
    bars = []
    with open(file_path, 'r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            if line != '':
                bars.append(float(line))
    bars.sort()
    return bars

def calc_phase_at_time(time_in_seconds, bars):
    bar_index = bisect_right(bars, time_in_seconds) - 1
    if bar_index < 0 or bar_index + 1 >= len(bars):
        return None

    bar_start = bars[bar_index]
    bar_end = bars[bar_index + 1]
    if bar_end <= bar_start:
        return None

    phase = (time_in_seconds - bar_start) / (bar_end - bar_start)
    # Keep in [0,1) for numerical stability.
    if phase < 0:
        return None
    if phase >= 1:
        phase = phase % 1.0
    return phase

from pathlib import Path

song_dirs = [
    str(p.parent) + '/'
    for p in Path(args.in_path).rglob('bars.csv')
    if any(p.parent.glob('*.ogg'))
]

for songpath in song_dirs:
    print('Processing ', songpath)
    audio_files = [f.name for f in Path(songpath).glob('*.ogg')]
    if True:

        print('Found ', str(audio_files))

        bars_path = songpath + 'bars.csv'
        if not os.path.exists(bars_path):
            print('Skipping song, missing bars.csv:', songpath)
            continue
        bar_starts = read_bar_starts(bars_path)
        if len(bar_starts) < 2:
            print('Skipping song, not enough bars:', songpath)
            continue

        for audio_file in audio_files:
            data, sample_rate = soundfile.read(songpath + audio_file)
            song_length = len(data) # in samples

            number_of_chunks = int(song_length / (chunk_length * (2 - args.min_pitch)) * 0.9 - 1)
            if number_of_chunks < 1:
                print('Song too short. Skipping.')
                continue

            chunk_pitches = []
            chunk_length_sum = 0
            for i in range(number_of_chunks):
                random_pitch = args.min_pitch + random.random() * 2 * (1 - args.min_pitch)
                chunk_pitches.append(random_pitch)
                chunk_length_sum += math.ceil(chunk_length * random_pitch)

            # selection, we pretend chunks have length 0 and randomly select the spots where they start
            free_area = song_length - chunk_length_sum

            spots = []
            for i in range(number_of_chunks):
                spots.append(int(random.random() * free_area))
            spots.sort()

            deltas = []
            for i in range(number_of_chunks - 1):
                deltas.append(spots[i + 1] - spots[i])

            print('Spots: ', spots)
            print('Song length: ', song_length)
            print('Free area: ', free_area)

            chunk_starts = []
            offset = 0
            for i, delta in enumerate(deltas):
                start = delta + offset
                chunk_starts.append(start)
                offset = start + int(chunk_length * chunk_pitches[i])

            infile = songpath + audio_file
            rel = os.path.relpath(songpath, args.in_path).replace(os.sep, '_')
            outfileprefix = args.out_path + '/' + abbreviate(rel) + '-' + abbreviate(audio_file[:-4]) + '-'
            digits = math.floor(math.log10(number_of_chunks)) + 1

            for i, start in enumerate(chunk_starts):

                outfile = outfileprefix + str(i).zfill(digits)

                pitch = chunk_pitches[i]

                start_time = start / sample_rate

                frame_count = int(chunk_duration * args.sample_rate / frame_size)
                phases = []
                chunk_is_valid = True
                for frame_index in range(frame_count):
                    local_time = frame_index * frame_size / args.sample_rate
                    original_time = start_time + local_time * pitch
                    phase = calc_phase_at_time(original_time, bar_starts)
                    if phase is None:
                        chunk_is_valid = False
                        break
                    phases.append(phase)

                if not chunk_is_valid:
                    if os.path.exists(outfile + '.events'):
                        os.remove(outfile + '.events')
                    continue

                with open(outfile + '.phase', 'w', encoding='utf8') as f:
                    for phase in phases:
                        f.write(f"{phase:.6f}\n")

                newrate = args.sample_rate * pitch

                af = f'asetrate={newrate},aresample={args.sample_rate}'
                af += f',atrim=start={start_time / pitch}:end={start_time / pitch + chunk_duration + 0.1}'
                af += ',asetpts=N/SR/TB'
                af += f',alimiter=level_in={args.volume}'

                ffmpeg_args = []
                ffmpeg_args += ['-v', 'warning']
                ffmpeg_args += ['-n'] # no overwrite
                ffmpeg_args += ['-i', infile]
                ffmpeg_args += ['-af', af]
                ffmpeg_args += ['-t', str(chunk_duration)]
                ffmpeg_args += ['-c:a', 'libvorbis']
                ffmpeg_args += ['-qscale:a', '6']
                ffmpeg_args += ['-ar', str(args.sample_rate)]
                ffmpeg_args += [outfile + '.ogg']

                ffmpeg_call = ['ffmpeg'] + ffmpeg_args
                try:
                    thread = threading.Thread(target=call_ffmpeg, args=(ffmpeg_call, outfile))
                    thread.start()
                    threads.append(thread)
                    if len(threads) >= args.max_processes:
                        threads[0].join()
                        threads.pop(0)
                except Exception as e:
                    print('Asynchronous call_ffmpeg failed.', e)

for thread in threads:
    thread.join()

print('Done.')
