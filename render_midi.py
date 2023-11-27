"""
takes midifiles from a folder and renders them to wav using fluidsynth while saving midi events of interest to a csv file
"""
import os, sys
import subprocess

import time

import threading

import argparse

import traceback

import mido

from audio_event import AudioEvent

from  config import channels, relevant_notes

parser = argparse.ArgumentParser(description='Renders midi songs with soundfonts.')
parser.add_argument('--midi-path', type=str, default='data/midi/lakh_clean', help='path to the directory containing midi files. The following directory structure is assumed: midi-path/<artist>/<song-name>.mid')
parser.add_argument('--single-file', type=str, default=None, help='path to a single midi file instead of --midi-path')

parser.add_argument('--soundfont-path', type=str, default='soundfonts', help='path to the soundfonts')
parser.add_argument('--out-path', type=str, help='path to the output directory')

parser.add_argument('--sample-rate', type=int, default=44100, help='sample rate')
parser.add_argument('--bits', type=int, default=16, help='bits')
parser.add_argument('--max-song-length', type=int, default=900, help='max song length in seconds')

parser.add_argument('--skip', type=int, default=0, help='process only every nth song when --midi-path is used')
parser.add_argument('--override', default=False, action='store_true', help='override existing files')
parser.add_argument('--max-processes', type=int, default=9, help='max processes to use')

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

sample_rate = args.sample_rate
bits = args.bits

threads = []

def convert_to_ogg(file_without_extension):
    """
    convert wav to ogg using ffmpeg in the background
    """
    try:
        command = [
            'ffmpeg',
            '-n', # no overwrite
            '-i', file_without_extension + ".wav",
            '-ac', str(channels),
            '-af', 'loudnorm',
            '-c:a', 'libvorbis',
            '-qscale:a', '6',
            '-ar', str(sample_rate),
            file_without_extension + ".ogg"
        ]
        with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as process:
            output, error = process.communicate()
            # remove wav
            if process.returncode == 0:
                print('Deleting wav after conversion of ' + file_without_extension)
                os.remove(file_without_extension + ".wav")
            else:
                print('Conversion failed for ' + file_without_extension)
                print('Output: ' + output.decode('utf-8'))
                print('Error: ' + error.decode('utf-8'))
    except Exception as error:
        print('Exception while converting to ogg: ' + str(error))
        traceback.print_exc()

class Soundfont:
    """
    represents a soundfont
    """
    def __init__(self, name, path):
        self.name = name
        self.path = path

# read all available soundfonts in the soundfonts folder
# find all .sf2 and .sf3 files in the soundfonts folder (recursively)
print('Loading soundfonts from ' + args.soundfont_path)
soundfonts = []
for root, dirs, files in os.walk(args.soundfont_path):
    for file in files:
        if file[-4:] == '.sf2' or file[-4:] == '.sf3':
            name = file[0:-4]
            print(name)
            path = os.path.join(root, file)
            soundfonts.append(Soundfont(name, path))


start_time = time.time()
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
        note_on_lists = {}
        for i in relevant_notes:
            note_on_lists[i] = []

        ticks_per_beat = mid.ticks_per_beat
        print('ticks per beat: ' + str(ticks_per_beat))

        class NoteOnEvent:
            def __init__(self, tick, velocity):
                self.tick = tick
                self.velocity = velocity

        class TempoChange:
            def __init__(self, tick, tempo):
                self.tick = tick
                self.tempo = tempo

        class VolumeChange:
            def __init__(self, tick, volume):
                self.tick = tick
                self.volume = volume


        tempo_changes = []
        volume_changes = []

        for i, track in enumerate(mid.tracks):
            # look for program change to drums

            ticks = 0

            for msg in track:
                ticks += msg.time
                if msg.type == 'set_tempo':
                    tempo_changes.append(TempoChange(ticks, msg.tempo))
                if msg.type == 'note_on':
                    if msg.channel == 9:
                        if msg.note in relevant_notes:
                            note_on_lists[msg.note].append(NoteOnEvent(ticks, msg.velocity))
                if msg.type == 'control_change':
                    if msg.channel == 9 and msg.control == 7:
                        volume_changes.append(VolumeChange(ticks, msg.value))
                        print('Volume change: ' + str(msg.value) + ' at ' + str(ticks))

        total_events = 0
        for i in relevant_notes:
            total_events += len(note_on_lists[i])

        if total_events < 20:
            print('Skipping ' + midifile + ' because it has less than 20 relevant percussion events')
        else:
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)

            # sort kicks and snares by time

            tempo_changes.sort(key=lambda x: x.tick)
            volume_changes.sort(key=lambda x: x.tick)

            audio_events = []

            for note in note_on_lists:
                note_on_list = note_on_lists[note]
                note_on_list.sort(key=lambda x: x.tick)

                tcp = 0 # tempo change pointer
                tempo = 500000 # initial tempo
                accumulated_time = 0
                accumulated_ticks = 0

                vcp = 0 # volume change pointer
                volume = 100 # initial volume

                # add noteOns with same tick
                i = 0
                while i < len(note_on_list) - 1:
                    note_on = note_on_list[i]
                    velocity = note_on.velocity
                    tick = note_on.tick
                    i += 1

                    # aggregate noteOns with same tick
                    while i < len(note_on_list) and tick == note_on_list[i].tick:
                        note_on = note_on_list[i]
                        velocity = 1 - ((1 - velocity) * (1 - note_on.velocity))
                        i += 1

                    # respect tempo_changes
                    while tcp < len(tempo_changes) and tempo_changes[tcp].tick <= tick:
                        accumulated_time += mido.tick2second(tempo_changes[tcp].tick - accumulated_ticks, ticks_per_beat, tempo)
                        accumulated_ticks = tempo_changes[tcp].tick
                        tempo = tempo_changes[tcp].tempo
                        tcp += 1

                    while vcp < len(volume_changes) and volume_changes[vcp].tick <= tick:
                        volume = volume_changes[vcp].volume
                        vcp += 1

                    event_time = accumulated_time + mido.tick2second(tick - accumulated_ticks, ticks_per_beat, tempo)
                    event_volume = (velocity / 127 * 0.5 + 0.5) * volume / 127

                    audio_events.append(AudioEvent(event_time, note, event_volume))

            audio_events.sort(key=lambda x: x.time)

            # write audio events to a file
            events_file_path = os.path.join(outpath, 'events.csv')
            with open(events_file_path, 'w', encoding='utf-8') as event_file:
                for event in audio_events:
                    event_file.write(event.to_csv_line() + '\n')

            for soundfont in soundfonts: 
                outfile_without_ext = os.path.join(outpath, soundfont.name)
                outfile = outfile_without_ext + '.wav'

                # check if file already exists
                if not args.override and os.path.exists(outfile_without_ext + '.ogg'):
                    print('Skipping ' + outfile + ' because it already exists')
                else:
                    print('Rendering midi using soundfont ' + soundfont.name)
                    command = [
                        'fluidsynth',
                        '-F', outfile, 
                        '-ni', 
                        '-a', 'file'
                        '--gain', '1', 
                        '-O', f's{bits}', 
                        '-r', str(sample_rate),
                        soundfont.path,
                        midifile,
                    ]
                    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as process:
                        max_size = (duration + 15) * sample_rate * 2 * bits / 8
                        max_size += 1024 * 1024 # add 1MB to the max size
                        while process.poll() is None:
                            #if outfile exists
                            if os.path.exists(outfile):
                                if os.stat(outfile).st_size > max_size:
                                    print('File size bigger than expected, possibly a corrupt midi file')
                                    print('Stopping fluidsynth...')
                                    process.kill()
                                    process.wait()
                                    # remove file
                                    print('Removing file')
                                    os.remove(outfile)
                                    raise Exception('File size bigger than expected, possibly a corrupt midi file')

                        # p.wait() # equivalent of 'while p.poll() is None'?

                        if process.returncode != 0:
                            print('Fluidsynth returned non-zero exit code')
                            os.remove(outfile) # if an error occured, it's better to remove the file
                            raise Exception('Fluidsynth returned non-zero exit code')

                        print('Converting wav to ogg in background')
                        try:
                            thread = threading.Thread(target=convert_to_ogg, args=(outfile_without_ext,))
                            thread.start()
                            threads.append(thread)
                            if len(threads) >= args.max_processes:
                                threads[0].join()
                                threads.pop(0)

                        except Exception as error:
                            print('Error converting to ogg', str(error))

    except Exception as error:
        print('Error processing ' + midifile)
        print(str(error))

if args.single_file is not None:
    print('Rendering single song' + args.single_file)
    generate_song(args.single_file, args.out_path, soundfonts)
    sys.exit(0)
else:
    print('Rendering songs from ' + args.midi_path + '\n')
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
                print('\n' + str(count) + '. processing ' + artist + '/' + song)

                songname = song[0:-4]
                outpath = args.out_path + '/' + artist + '/' + songname + '/'

                soundfont = soundfonts[processed_count % len(soundfonts)]

                generate_song(midifile, outpath, [soundfont])
                processed_count += 1

            count += 1

# wait for all threads to finish
for thread in threads:
    thread.join()

print('Finished in ' + str(time.time() - start_time) + ' seconds')
