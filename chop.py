"""
This script loads files from the out directory and chops them into 10s long files for the training. 
It also reads event files and ouputs them fitting to the chopped files.
"""

import math
import sys

import os
import random

import argparse

import threading
import subprocess

import soundfile

from config import chunk_duration

from audio_event import AudioEvent

parser = argparse.ArgumentParser(description='Chop up the songs into chunks.')

parser.add_argument('--in-path', type=str, default='data/rendered_midi/lakh_clean', help='path to the rendered audio. The directory structure is expected to be: <in-path>/<artist>/ and contain a folder <song> with n .ogg files along with one events.csv file')
parser.add_argument('--out-path', type=str, help='path to the output directory')
parser.add_argument('--sample-rate', type=int, default=44100, help='sample rate')
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
        os.remove(outfile + '.events')
        os.remove(outfile + '.ogg')

for artist in os.listdir(args.in_path):
    print('Artist: ', artist)
    for song in os.listdir(args.in_path + '/' + artist):
        songpath = args.in_path + '/' + artist + '/' + song + '/'
        print('Processing ', songpath)
        audio_files = []
        for file in os.listdir(songpath):
            if file[-4:] == '.ogg':
                audio_files.append(file)

        print('Found ', str(audio_files))

        events = []
        with open(songpath + 'events.csv', 'r', encoding='utf8') as f:
            for line in f:
                events.append(AudioEvent.from_csv_line(line))

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
            outfileprefix = args.out_path + '/' + '-'.join([abbreviate(artist), abbreviate(song), abbreviate(audio_file[:-4])]) + '-'
            digits = math.floor(math.log10(number_of_chunks)) + 1

            event_pointer = 0
            last_event_time = -1
            for i, start in enumerate(chunk_starts):

                outfile = outfileprefix + str(i).zfill(digits)

                pitch = chunk_pitches[i]

                start_time = start / sample_rate
                end_time = start_time + chunk_duration * pitch

                # write relevant kicks and snares to file
                with open(outfile + '.events', 'w', encoding='utf8') as f:
                    # start_ and end_time refer to the original pitch in seconds
                    while event_pointer < len(events) and events[event_pointer].time < start_time:
                        event_pointer += 1
                    while event_pointer < len(events) and events[event_pointer].time <= end_time:
                        event = events[event_pointer]
                        time = (event.time - start_time) / pitch
                        if event.volume < 0:
                            print('Volume < 0. Skipping event.')
                        else:
                            f.write(f"{time:.4f},{event.note},{event.volume:.4f}\n")
                        event_pointer += 1

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
