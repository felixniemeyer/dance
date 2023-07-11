# this script loads files from the out directory and chops them into 10s long files for the training

import os
import random   

import argparse

import wave 

parser = argparse.ArgumentParser(description='Chop up the songs into chunkgs')
parser.add_argument('--in-path', type=str, default='rendered-midi', help='path to the midi-to-audio-out directory')
parser.add_argument('--out-path', type=str, default='chunks', help='path to the output directory')
parser.add_argument('--chunk-length', type=int, default=10, help='length of the chunks in seconds')
parser.add_argument('--min-pitch', type=float, default=0.9, help='will choose random pitches between this and 1')

parser.add_argument('--dry-run', default=False, help='dry run')


# parse arguments
args = parser.parse_args()

chunk_length = args.chunk_length

# mk out-path if it doesn't exist
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)

def get_wav_duration(wav_path):
    with wave.open(wav_path, 'rb') as f:
        frames = f.getnframes()
        rate = f.getframerate()
        duration = frames / float(rate)
        return duration

def abbreviate(string): 
    # replace all whitespace with underscores
    string = string.replace(' ', '_')
    string = string.replace('-', '_')
    return string.ljust(5, '_')[0:5]

# load the data (folder structure: out/artist/song/*wav)
chunk_count = 0
for artist in os.listdir(args.in_path):
    for song in os.listdir(args.in_path + '/' + artist):
        songpath = args.in_path + '/' + artist + '/' + song + '/'
        audio_files = []
        for file in os.listdir(songpath):
            if file[-4:] == '.wav':
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
                duration = get_wav_duration(songpath + audio_file)
                chunks = int(duration / chunk_length / 2)
                cursor = 0

                infile = songpath + audio_file
                outfileprefix = args.out_path + '/' + '-'.join([abbreviate(artist), abbreviate(song), abbreviate(audio_file)]) + '-'

                for i in range(chunks):
                    cursor += random.random() * chunk_length

                    pitch = args.min_pitch + random.random() * (1 - args.min_pitch)

                    start = cursor # / pitch
                    outfile = outfileprefix + str(chunk_count)

                    has_kicks_or_snares = False
                    # write relevant kicks and snares to file
                    with open(outfile + '.kicks', 'w') as f:
                        for kick in kicks:
                            time = kick / pitch - start 
                            if time >= 0 and time < chunk_length:
                                f.write(str(time) + '\n')
                                has_kicks_or_snares = True
                    
                    # same for snares
                    with open(outfile + '.snares', 'w') as f:
                        for snare in snares:
                            time = snare / pitch - start 
                            if time >= 0 and time < chunk_length:
                                f.write(str(time) + '\n')
                                has_kicks_or_snares = True

                    if not has_kicks_or_snares:
                        # remove the 2 files
                        print('No kicks or snares in this chunk. Rolling back.')
                        os.remove(outfile + '.kicks')
                        os.remove(outfile + '.snares')
                    else: 
                        af = f'atempo={pitch}'
                        ffmpeg_cmd = f'ffmpeg -i "{infile}" -af "{af}" -ss {start} -t {chunk_length} "{outfile + ".opus"}"'
                        print(ffmpeg_cmd)
                        os.system(ffmpeg_cmd)

                    cursor += args.chunk_length
                    chunk_count += 1

            if chunk_count > 100:
                exit(0)
