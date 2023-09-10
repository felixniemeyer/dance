# this script loads files from the out directory and chops them into 10s long files for the training

import math

import os
import random   

import argparse

import soundfile

import threading 
import subprocess

parser = argparse.ArgumentParser(description='Chop up the songs into chunks.')

help_text = 'path to the rendered audio. The directory structure is expected to be: <in-path>/<artist>/<song>/ and contain an .ogg along with a .kicks and .snares file for each song'
parser.add_argument('--in-path', type=str, default='data/rendered_midi/lakh_clean', help=help_text)
parser.add_argument('--out-path', type=str, help='path to the output directory')

parser.add_argument('--chunk-duration', type=int, default=16, help='length of the chunks in seconds')

parser.add_argument('--sample-rate', type=int, default=44100, help='sample rate')

help_text = 'will choose random pitches between this and 1'
parser.add_argument('--min-pitch', type=float, default=0.79, help=help_text)
parser.add_argument('--volume', type=float, default=5, help='output level before limiter')

parser.add_argument('--dry-run', default=False, help='dry run')

parser.add_argument('--max-processes', type=int, default=9, help='max processes to use')

# parse arguments
args = parser.parse_args()

chunk_duration = args.chunk_duration
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
    # replace all whitespace with underscores
    string = string.replace(' ', '')
    string = string.replace('-', '')
    string = string.ljust(6, '_')
    return string[0:3] + string[-3:]

def call_ffmpeg(command, outfile): 
    print(' '.join(ffmpeg_call))
    try: 
        subprocess.check_call(command)
    except KeyboardInterrupt:
        print('\nAborting')
        exit(0)
    except:
        print('ffmpeg failed. Rolling back.')
        os.remove(outfile + '.kicks')
        os.remove(outfile + '.snares')
        os.remove(outfile + '.ogg')

global_chunk_count = 0
for artist in os.listdir(args.in_path):
    print('Artist: ', artist)
    for song in os.listdir(args.in_path + '/' + artist):
        songpath = args.in_path + '/' + artist + '/' + song + '/'
        audio_files = []
        for file in os.listdir(songpath):
            if file[-4:] == '.ogg':
               audio_files.append(file)

        print('Found ', str(audio_files))

        if len(audio_files) > 0:
            # load kicks and snares (one line = one time stamp)
            kicks = []
            with open(songpath + 'kicks.txt', 'r') as f:
                for line in f: 
                    kicks.append(float(line))

            snares = []
            with open(songpath + 'snares.txt', 'r') as f:
                for line in f: 
                    snares.append(float(line))

            for audio_file in audio_files:
                data, sample_rate = soundfile.read(songpath + audio_file)
                song_length = len(data) # in samples

                number_of_chunks = int(song_length / chunk_length * 0.66 - 1)
                # selection, we pretend chunks have length 0 and randomly select the spots where they start
                total_sample_duration = number_of_chunks * chunk_duration * sample_rate
                free_area = song_length - total_sample_duration

                spots = []
                for i in range(number_of_chunks):
                    spots.append(int(random.random() * free_area))
                spots.sort()

                chunk_starts = []
                for i, start in enumerate(spots):
                    chunk_starts.append((start + i * chunk_length) / sample_rate)

                infile = songpath + audio_file
                outfileprefix = args.out_path + '/' + '-'.join([abbreviate(artist), abbreviate(song), abbreviate(audio_file[:-4])]) + '-'
                digits = math.floor(math.log10(number_of_chunks)) + 1

                for i, start in enumerate(chunk_starts):
                    global_chunk_count += 1

                    outfile = outfileprefix + str(i).zfill(digits)
                    pitch = args.min_pitch + random.random() * (1 - args.min_pitch)

                    has_kicks_or_snares = False
                    # write relevant kicks and snares to file
                    with open(outfile + '.kicks', 'w') as f:
                        for kick in kicks:
                            time = kick / pitch - start 
                            if time >= 0 and time < chunk_duration:
                                f.write(str(time) + '\n')
                                has_kicks_or_snares = True
                    
                    # same for snares
                    with open(outfile + '.snares', 'w') as f:
                        for snare in snares:
                            time = snare / pitch - start 
                            if time >= 0 and time < chunk_duration:
                                f.write(str(time) + '\n')
                                has_kicks_or_snares = True

                    if not has_kicks_or_snares:
                        # remove the 2 files
                        print('No kicks or snares in this chunk. Rolling back.')
                        os.remove(outfile + '.kicks')
                        os.remove(outfile + '.snares')
                    else: 

                        newrate = args.sample_rate * pitch

                        af = f'asetrate={newrate},aresample={args.sample_rate}'
                        af += f',atrim=start={start}:end={start + chunk_duration + 0.1}'
                        af += f',asetpts=N/SR/TB'
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
                            print('Asynchronous call_ffmpeg failed.')

for thread in threads:
    thread.join()

print('Done. Created', global_chunk_count, 'chunks.')
