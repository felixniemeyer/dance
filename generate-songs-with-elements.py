import os
import mido
import subprocess

import argparse 

parser = argparse.ArgumentParser(description='Generate songs with elements')
parser.add_argument('--midi-path', type=str, default='lakh-midi-dataset-clean', help='path to the midi files')
parser.add_argument('--single-file', type=str, default=None, help='path to a single midi file instead of --midi-path')

parser.add_argument('--soundfont-path', type=str, default='soundfonts', help='path to the soundfonts')
parser.add_argument('--out-path', type=str, default='rendered-midi', help='path to the output directory')

parser.add_argument('--sample-rate', type=int, default=44100, help='sample rate')
parser.add_argument('--bits', type=int, default=16, help='bits')
parser.add_argument('--max-song-length', type=int, default=900, help='max song length in seconds')

# parse arguments
args = parser.parse_args()

sample_rate = args.sample_rate
bits = args.bits
channels = 2

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


        for i, track in enumerate(mid.tracks):
            # look for program change to drums
            ticks_per_beat = mid.ticks_per_beat

            trackKicks = []
            trackSnares = []
            tempo = 500000 
            ticks = 0

            accumulated_time = 0

            using_default_tempo = True
            tempo_changed = False

            for msg in track:
                ticks += msg.time
                if msg.type == 'set_tempo':
                    accumulated_time += mido.tick2second(ticks, ticks_per_beat, tempo)
                    if using_default_tempo: 
                        using_default_tempo = False
                    elif tempo != msg.tempo:
                        tempo_changed = True
                    tempo = msg.tempo
                    ticks = 0
                time = accumulated_time + mido.tick2second(ticks, ticks_per_beat, tempo)
                if msg.type == 'note_on': 
                    if msg.channel == 9:
                        if msg.note == 36 or msg.note == 35:
                            trackKicks.append(time) 
                        if msg.note >= 37 and msg.note <= 40:
                            trackSnares.append(time)

            if(len(trackKicks) > 10 or len(trackSnares) > 10):
                hasRelevantPercussion = True
                print('Found drums in track ' + str(i) + ' ' + track.name)
                kicks += trackKicks
                snares += trackSnares

            if tempo_changed:
                print('tempo in this song changes') 

        if not hasRelevantPercussion: 
            print('Skipping ' + artist + '/' + song + ' because it has no relevant percussion')
        else:
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)

            # sort kicks and snares by time
            kicks.sort()
            snares.sort()

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
                outfile = os.path.join(outpath, soundfont.name + '.wav')

                # check if file already exists 
                if os.path.exists(outfile):
                    print('Skipping ' + artist + '/' + song + ' because it already exists')
                else: 
                    print('Rendering midi using soundfont ' + soundfont.name)
                    p = subprocess.Popen([
                        'fluidsynth',
                        '-F', outfile, 
                        '-ni', 
                        '-a', 'file'
                        '-g', '1', 
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

    except Exception as e:
        print('Error processing ' + artist + '/' + song)
        print(str(e))


if args.single_file != None:
    print('Rendering single song' + args.single_file)
    generate_song(args.single_file, args.out_path, soundfonts)
    exit(0)

else: 
    print('Rendering songs from ' + args.midi_path)
    # for every song in every artist dir
    count = 0
    for artist in os.listdir(args.midi_path):
        # continue if not a directory
        if not os.path.isdir(args.midi_path + '/' + artist):
            continue
        for song in os.listdir(args.midi_path + '/' + artist):
            midifile = args.midi_path + '/' + artist + '/' + song
            print('\n\n') 
            print(str(count) + '. processing ' + artist + '/' + song) 

            songname = song[0:-4]
            outpath = args.out_path + '/' + artist + '/' + songname + '/'

            generate_song(midifile, outpath, [soundfonts[count % len(soundfonts)]])

            count += 1
