import os
import mido
import subprocess

import time

sample_rate = 44100
bits = 16
channels = 2

# read all available soundfonts in the soundfonts folder
print('Loading soundfonts from ./soundfonts') 
soundfontnames = []
soundfonts = []
for i, soundfont in enumerate(os.listdir('soundfonts')): 
    if soundfont[-4:] == '.sf2':
        name = soundfont[0:-4]
        print(str(i + 1) + ". " + name)
        soundfontnames.append(name)
        soundfonts.append(soundfont)

# mkdir out if it doesn't exist
if not os.path.exists('out'):
    os.mkdir('out')


print('\nRendering songs')
# for every song in every artist dir
count = 0
skip = 32
for artist in os.listdir('lakh-midi-dataset-clean'):
    # continue if not a directory
    if not os.path.isdir('lakh-midi-dataset-clean/' + artist):
        continue
    for song in os.listdir('lakh-midi-dataset-clean/' + artist):
        # just for reducing the dataset size
        if skip > 0: 
            skip -= 1
            continue
        skip = 100
        count += 1

        def song_log(msg):
            print('\t' + msg) 

        try: 
            midifile = 'lakh-midi-dataset-clean/' + artist + '/' + song
            print(str(count) + '. processing ' + artist + '/' + song) 

            mid = mido.MidiFile(midifile)
            duration = mid.length

            print(f'Song length: {duration:.2f} seconds')
            if mid.length > 900: # skip songs longer than 15 minutes. 
                print('Skipping song because it is too long: ' + str(mid.length))
                continue

            # analyze midi file and find all kicks and snares
            kicks = []
            snares = []

            hasRelevantPercussion = False
            hasBrokenTracks = False

            song_log(str(len(mid.tracks)) + ' tracks') 
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
                    song_log('Found drums in track ' + str(i) + ' ' + track.name)
                    kicks += trackKicks
                    snares += trackSnares

                if tempo_changed:
                    song_log('tempo in this song changes') 

            if not hasRelevantPercussion: 
                song_log('Skipping ' + artist + '/' + song + ' because it has no relevant percussion')
            else:
                songname = song[0:-4]
                outpath = 'out/' + artist + '/' + songname + '/' 
                if not os.path.exists(outpath):
                    os.makedirs(outpath, exist_ok=True)

                # sort kicks and snares by time
                kicks.sort()
                snares.sort()

                # write kick timestamps to a file
                with open('out/' + artist + '/' + songname + '/kicks.txt', 'w') as f:
                    for msg in kicks:
                        f.write(str(msg) + '\n')
                # write snare timestamps to a file
                with open('out/' + artist + '/' + songname + '/snares.txt', 'w') as f:
                    for msg in snares:
                        f.write(str(msg) + '\n')

                # for i in range(len(synths)):
                font = count % len(soundfontnames) # reduce data size
                outfile = outpath + soundfontnames[font] + '.wav'

                # check if file already exists 
                if os.path.exists(outfile):
                    song_log('Skipping ' + artist + '/' + song + ' because it already exists')
                else: 
                    song_log('Rendering midi using soundfont ' + soundfontnames[font])
                    p = subprocess.Popen([
                        'fluidsynth',
                        '-F', outfile, 
                        '-ni', 
                        '-a', 'file'
                        '-g', '1', 
                        '-O', f's{bits}', 
                        '-r', str(sample_rate),
                        f'soundfonts/{soundfonts[font]}', 
                        midifile,
                    ], stderr=subprocess.DEVNULL) # muting errors
                    max_size = (duration + 1) * sample_rate * channels * bits / 8 
                    max_size += 1024 * 1024 # add 1MB to the max size
                    while p.poll() is None:
                        #if outfile exists
                        if os.path.exists(outfile): 
                            if os.stat(outfile).st_size > max_size:
                                song_log('File size bigger than expected, possibly a corrupt midi file')
                                song_log('Stopping fluidsynth...')
                                p.kill()
                                p.wait()
                                # remove file
                                song_log('Removing file')
                                os.remove(outfile)
                                raise Exception('File size bigger than expected, possibly a corrupt midi file')

                    p.wait()

                    if p.returncode != 0:
                        song_log('Fluidsynth returned non-zero exit code')
                        os.remove(outfile) # if an error occured, it's better to remove the file
                        raise Exception('Fluidsynth returned non-zero exit code')

        except Exception as e:
            song_log('Error processing ' + artist + '/' + song)
            song_log(str(e))
            print()

