# this script loads files from the out directory and chops them into 10s long files for the training

import os
import random   

import argparse

import soundfile

import subprocess

parser = argparse.ArgumentParser(description='Chop up the songs into chunkgs')
parser.add_argument('--in-path', type=str, default='rendered-midi', help='path to the midi-to-audio-out directory')
parser.add_argument('--out-path', type=str, default='chunks', help='path to the output directory')
parser.add_argument('--chunk-length', type=int, default=16, help='length of the chunks in seconds')
parser.add_argument('--min-pitch', type=float, default=0.85, help='will choose random pitches between this and 1')

parser.add_argument('--sample-rate', type=int, default=44100, help='sample rate')

parser.add_argument('--min-low-pass', type=float, default=4000, help='will choose random low pass between this and 24000')
parser.add_argument('--max-high-pass', type=float, default=500, help='will choose random low pass between 0 and this')

parser.add_argument('--min-volume', type=float, default=1, help='will choose random gain between this and --max-volume')
parser.add_argument('--max-volume', type=float, default=2, help='will choose random gain between --min-volume and this')

parser.add_argument('--max-noise', type=float, default=0.05, help='will choose random low pass between this and 1')

parser.add_argument('--max-overdrive', type=float, default=0, help='will choose random low pass between this and 1')

parser.add_argument('--dry-run', default=False, help='dry run')

# parse arguments
args = parser.parse_args()

chunk_length = args.chunk_length

# mk out-path if it doesn't exist
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)

def abbreviate(string): 
    # replace all whitespace with underscores
    string = string.replace(' ', '')
    string = string.replace('-', '')
    string = string.ljust(6, '_')
    return string[0:3] + string[-3:]

# load the data (folder structure: out/artist/song/*.ogg)
chunk_count = 0
for artist in os.listdir(args.in_path):
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
                duration = len(data) / sample_rate

                chunks = int(duration / chunk_length / 2)
                cursor = 0

                infile = songpath + audio_file
                outfileprefix = args.out_path + '/' + '-'.join([abbreviate(artist), abbreviate(song), abbreviate(audio_file[:-4])]) + '-'

                for i in range(chunks):
                    cursor += random.random() * chunk_length

                    pitch = args.min_pitch + random.random() * (1 - args.min_pitch)

                    start = cursor # / pitch
                    outfile = outfileprefix + str(i).zfill(2)

                    cursor += args.chunk_length
                    chunk_count += 1

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
                        volume = args.min_volume + random.random() * (args.max_volume - args.min_volume)
                        lowpass = args.min_low_pass + (1 - random.random() ** 2) * (args.sample_rate / 2 - args.min_low_pass)
                        highpass = args.max_high_pass * random.random() ** 2
                        compress_level_in = 1 + random.random() * args.max_overdrive
                        compress_threshold = 0.5 + 0.4 * random.random() 
                        compress_ratio = 20
                        compress_mix = 0.0 + 0.1 * random.random() * args.max_overdrive
                        noise = random.random() ** 2 * args.max_noise

                        prenoise_af = f'atempo={pitch}'
                        prenoise_af += f',atrim=start={start}:end={start + chunk_length}'

                        postnoise_af = f'volume={volume}'
                        prenoise_af += f',lowpass=f={lowpass}'
                        prenoise_af += f',highpass=f={highpass}'
                        prenoise_af += f',acompressor=level_in={compress_level_in}:threshold={compress_threshold}:ratio={compress_ratio}:mix={compress_mix}:attack=0.1:release=0.1'

                        fc = f'[0:a]{prenoise_af}[musik];[1:a]volume={noise}[noise];[musik][noise]amix=inputs=2[master];[master]{postnoise_af}[aout]'

                        ffmpeg_args = []
                        ffmpeg_args += ['-v', 'warning'] 
                        ffmpeg_args += ['-n'] # no overwrite 
                        ffmpeg_args += ['-i', infile] 
                        ffmpeg_args += ['-f', 'lavfi'] 
                        ffmpeg_args += ['-i', f'anoisesrc=d={chunk_length}:c=pink:r={args.sample_rate}']
                        ffmpeg_args += ['-filter_complex', fc]
                        ffmpeg_args += ['-map', '[aout]']
                        ffmpeg_args += ['-ss', str(start)]
                        ffmpeg_args += ['-t', str(chunk_length)]
                        ffmpeg_args += ['-c:a', 'libvorbis']
                        ffmpeg_args += ['-qscale:a', '2']
                        ffmpeg_args += ['-ar', str(args.sample_rate)]
                        ffmpeg_args += [outfile + '.ogg']

                        ffmpeg_call = ['ffmpeg'] + ffmpeg_args
                        print(' '.join(ffmpeg_call))
                        try: 
                            subprocess.check_call(ffmpeg_call)
                        except KeyboardInterrupt:
                            print('\nAborting')
                            exit(0)
                        except:
                            print('ffmpeg failed. Rolling back.')
                            os.remove(outfile + '.kicks')
                            os.remove(outfile + '.snares')
                            os.remove(outfile + '.ogg')

