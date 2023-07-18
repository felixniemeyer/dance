import os
import mido
import subprocess

import threading

import argparse 

import traceback

parser = argparse.ArgumentParser(description='Generate songs with elements')
parser.add_argument('--midi-path', type=str, default='lakh-midi-dataset-clean', help='path to the midi files')
parser.add_argument('--single-file', type=str, default=None, help='path to a single midi file instead of --midi-path')

parser.add_argument('--soundfont-path', type=str, default='soundfonts', help='path to the soundfonts')
parser.add_argument('--out-path', type=str, default='rendered-midi', help='path to the output directory')

parser.add_argument('--sample-rate', type=int, default=44100, help='sample rate')
parser.add_argument('--bits', type=int, default=16, help='bits')
parser.add_argument('--max-song-length', type=int, default=900, help='max song length in seconds')

parser.add_argument('--skip', type=int, default=0, help='process only every nth song when --midi-path is used')
parser.add_argument('--override', default=False, action='store_true', help='override existing files')
parser.add_argument('--max-processes', type=int, default=9, help='max processes to use')

# parse arguments
args = parser.parse_args()

sample_rate = args.sample_rate
bits = args.bits
channels = 2

threads = []

def convert_to_ogg(file_without_extension):
    # convert wav to ogg using ffmpeg in the background
    try: 
        return_code = subprocess.call([
            'ffmpeg',
            '-n', # no overwrite
            '-v', 'warning',
            '-i', file_without_extension + ".wav",
            '-c:a', 'libvorbis',
            '-qscale:a', '6',
            '-ar', str(sample_rate),
            file_without_extension + ".ogg"
        ])
        # remove wav
        if return_code == 0:
            print('Deleting wav after conversion of ' + file_without_extension)
            os.remove(file_without_extension + ".wav")
        else: 
            print('Conversion failed for ' + file_without_extension)
    except Exception as e:
        print('Exception while converting to ogg: ' + str(e))
        traceback.print_exc()

class Soundfont: 
    def __init__(self, name, path):
        self.name = name
        self.path = path

# read all available soundfonts in the soundfonts folder
print('Loading soundfonts from ' + args.soundfont_path)
soundfonts = []
for i, soundfont in enumerate(os.listdir(args.soundfont_path)):
    if soundfont[-4:] == '.sf2':
        name = soundfont[0:-4]
        print(str(i + 1) + ". " + name)
        path = os.path.join(args.soundfont_path, soundfont)
        soundfonts.append(Soundfont(name, path))
print()

# mkdir out-path if it doesn't exist
if not os.path.exists(args.out_path):
    os.makedirs(args.out_path, exist_ok=True)

def generate_song(midifile, outpath, soundfonts): 

    try: 
        mid = mido.MidiFile(midifile)
        duration = mid.length

        print(f'Song length: {duration:.2f} seconds')
        if mid.length > args.max_song_length: # skip songs longer than 15 minutes. 
            print('Skipping song because it is too long: ' + str(mid.length))
            return 

        print(f'Track count: {len(mid.tracks)}')

        # analyze midi file and find all kicks and snares
        kicks = []
        snares = []

        hasRelevantPercussion = False

        ticks_per_beat = mid.ticks_per_beat
        print('ticks per beat: ' + str(ticks_per_beat))

        class TempoChange:
            def __init__(self, tick, tempo):
                self.at_tick = tick
                self.tempo = tempo

        tempo_changes = []

        for i, track in enumerate(mid.tracks):
            # look for program change to drums

            trackKicks = []
            trackSnares = []
            ticks = 0

            for msg in track:
                ticks += msg.time
                if msg.type == 'set_tempo':
                    tempo_changes.append(TempoChange(ticks, msg.tempo))
                if msg.type == 'note_on': 
                    if msg.channel == 9:
                        if msg.note == 36 or msg.note == 35:
                            trackKicks.append(ticks) 
                        if msg.note >= 37 and msg.note <= 40:
                            trackSnares.append(ticks)

            if(len(trackKicks) > 10 or len(trackSnares) > 10):
                hasRelevantPercussion = True
                print('Found drums in track ' + str(i) + ' ' + track.name)
                kicks += trackKicks
                snares += trackSnares

        if not hasRelevantPercussion: 
            print('Skipping ' + midifile + ' because it has no relevant percussion')
        else:
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)

            # sort kicks and snares by time
            kicks.sort()
            snares.sort()
            tempo_changes.sort(key=lambda x: x.at_tick)

            for tickstamps in [kicks, snares]: 
                tempo = 500000
                tcp = 0 # tempo change pointer
                accumulated_time = 0
                accumulated_ticks = 0
                for i in range(len(tickstamps)):
                    tickstamp = tickstamps[i]
                    while tcp < len(tempo_changes) and tempo_changes[tcp].at_tick <= tickstamp:
                        accumulated_time += mido.tick2second(tempo_changes[tcp].at_tick - accumulated_ticks, ticks_per_beat, tempo)
                        accumulated_ticks = tempo_changes[tcp].at_tick
                        tempo = tempo_changes[tcp].tempo
                        tcp += 1
                    tickstamps[i] = accumulated_time + mido.tick2second(tickstamp - accumulated_ticks, ticks_per_beat, tempo)

            # write kick timestamps to a file
            kicksfile = os.path.join(outpath, 'kicks.txt')
            with open(kicksfile, 'w') as f:
                for msg in kicks:
                    f.write(str(msg) + '\n')
            # write snare timestamps to a file
            snaresfile = os.path.join(outpath, 'snares.txt')
            with open(snaresfile, 'w') as f:
                for msg in snares:
                    f.write(str(msg) + '\n')

            for soundfont in soundfonts: 
                outfile_without_ext = os.path.join(outpath, soundfont.name)
                outfile = outfile_without_ext + '.wav'

                # check if file already exists 
                if not args.override and os.path.exists(outfile_without_ext + '.ogg'):
                    print('Skipping ' + outfile + ' because it already exists')
                else: 
                    print('Rendering midi using soundfont ' + soundfont.name)
                    p = subprocess.Popen([
                        'fluidsynth',
                        '-F', outfile, 
                        '-ni', 
                        '-a', 'file'
                        '--gain', '1', 
                        '-O', f's{bits}', 
                        '-r', str(sample_rate),
                        soundfont.path,
                        midifile,
                    ], stderr=subprocess.DEVNULL) # muting errors
                    max_size = (duration + 10) * sample_rate * channels * bits / 8 
                    max_size += 1024 * 1024 # add 1MB to the max size
                    while p.poll() is None:
                        #if outfile exists
                        if os.path.exists(outfile): 
                            if os.stat(outfile).st_size > max_size:
                                print('File size bigger than expected, possibly a corrupt midi file')
                                print('Stopping fluidsynth...')
                                p.kill()
                                p.wait()
                                # remove file
                                print('Removing file')
                                os.remove(outfile)
                                raise Exception('File size bigger than expected, possibly a corrupt midi file')

                    # p.wait() # equivalent of 'while p.poll() is None'?

                    if p.returncode != 0:
                        print('Fluidsynth returned non-zero exit code')
                        os.remove(outfile) # if an error occured, it's better to remove the file
                        raise Exception('Fluidsynth returned non-zero exit code')
                    else: 
                        print('Converting wav to ogg in background')
                        try: 
                            thread = threading.Thread(target=convert_to_ogg, args=(outfile_without_ext,))
                            thread.start()
                            threads.append(thread)
                            if len(threads) >= args.max_processes:
                                threads[0].join()
                                threads.pop(0)
                        except Exception as e:
                            print('Error converting to ogg', str(e))

    except Exception as e:
        print('Error processing ' + midifile)
        print(str(e))

if args.single_file != None:
    print('Rendering single song' + args.single_file)
    generate_song(args.single_file, args.out_path, soundfonts)
    exit(0)

else: 
    print('Rendering songs from ' + args.midi_path)
    # for every song in every artist dir
    count = 0
    processed_count = 0
    for artist in os.listdir(args.midi_path):
        # continue if not a directory
        if not os.path.isdir(args.midi_path + '/' + artist):
            continue
        for song in os.listdir(args.midi_path + '/' + artist):
            if count % (args.skip + 1) == 0: 
                midifile = args.midi_path + '/' + artist + '/' + song
                print('\n\n') 
                print(str(count) + '. processing ' + artist + '/' + song) 

                songname = song[0:-4]
                outpath = args.out_path + '/' + artist + '/' + songname + '/'

                generate_song(midifile, outpath, [soundfonts[processed_count % len(soundfonts)]])
                
                processed_count += 1

            count += 1
            
# wait for all threads to finish
for thread in threads:
    thread.join()
