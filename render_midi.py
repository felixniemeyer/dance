"""
renders midi files to wav while exporting relevant audio events 
"""
import os
import sys
import subprocess
import time
import threading
import argparse

import mido

from audio_event import AudioEvent

from  config import channels, relevant_notes, samplerate

parser = argparse.ArgumentParser(description='Renders midi songs with soundfonts.')
parser.add_argument('--midi-path', type=str, default='data/midi/lakh_clean',
    help='path to the directory containing midi files. The following directory structure is assumed: midi-path/<artist>/<song-name>.mid')
parser.add_argument('--single-file', type=str, default=None,
    help='path to a single midi file instead of --midi-path')

parser.add_argument('--soundfont-path', type=str, default='soundfonts',
    help='path to the soundfonts')
parser.add_argument('--single-soundfont', type=str, default=None,
    help='path to a single soundfont instead of --soundfont-path')
parser.add_argument('--out-path', type=str, help='path to the output directory')

parser.add_argument('--bits', type=int, default=16,
    help='bits per sample')
parser.add_argument('--max-song-length', type=int, default=60 * 20,
    help='max song length in seconds')

parser.add_argument('--skip', type=int, default=0,
    help='process only every nth song when --midi-path is used')
parser.add_argument('--overwrite', default=False, action='store_true',
    help='overwrite existing files')
parser.add_argument('--max-processes', type=int, default=9,
    help='max processes to use')

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

class NoteOnEvent:
    def __init__(self, tick, note, velocity):
        self.tick = tick
        self.note = note
        self.velocity = velocity

class TempoChange:
    def __init__(self, tick, tempo):
        self.tick = tick
        self.tempo = tempo

    def __str__(self):
        return f'tick: {self.tick}, tempo: {self.tempo}'

    def __repr__(self):
        return str(self)

class VolumeChange:
    def __init__(self, tick, volume):
        self.tick = tick
        self.volume = volume

start_time = time.time()
def generate_song(midifile, outpath, soundfonts, threads):
    try:
        mid = mido.MidiFile(midifile)
        duration = mid.length

        print(f'Song length: {duration:.2f} seconds')
        if duration > args.max_song_length: # skip songs longer than 15 minutes.
            print('Skipping song because it is too long: ' + str(mid.length))
            return

        print(f'Track count: {len(mid.tracks)}')

        ticks_per_beat = mid.ticks_per_beat
        print('ticks per beat: ' + str(ticks_per_beat))

        note_ons, tempo_changes, volume_changes = read_events(mid)

        if len(note_ons) < 20:
            print('Skipping ' + midifile + ' because it has less than 20 relevant percussion events')
        else:
            if not os.path.exists(outpath):
                os.makedirs(outpath, exist_ok=True)

            audio_events = calc_audio_events(note_ons, tempo_changes, volume_changes, ticks_per_beat)
            write_audio_events(audio_events, outpath)

            for soundfont in soundfonts:
                outfile_without_ext = os.path.join(outpath, soundfont.name)
                # call render_and_convert in a new thread
                thread = threading.Thread(target=render_and_convert, args=(midifile, outfile_without_ext, soundfont, duration + 60))
                threads.append(thread)
                thread.start()
                print(f'Starting thread {len(threads)}/{args.max_processes}')

                while len(threads) >= args.max_processes:
                    threads[0].join()
                    threads.pop(0)

    except Exception as error:
        print('Error processing ' + midifile + ': ' + str(error))
        print(str(error))

def read_events(mid):
    note_ons = []
    tempo_changes = []
    volume_changes = []

    for _, track in enumerate(mid.tracks):
        # look for program change to drums

        ticks = 0

        for msg in track:
            ticks += msg.time
            if msg.type == 'set_tempo':
                tempo_changes.append(TempoChange(ticks, msg.tempo))
            if msg.type == 'note_on':
                if msg.channel == 9:
                    if msg.note in relevant_notes:
                        note_ons.append(NoteOnEvent(ticks, msg.note, msg.velocity))
            if msg.type == 'control_change':
                if msg.channel == 9 and msg.control == 7:
                    volume_changes.append(VolumeChange(ticks, msg.value))

    # sort everything because it may come from different tracks
    tempo_changes.sort(key=lambda x: x.tick)
    volume_changes.sort(key=lambda x: x.tick)
    note_ons.sort(key=lambda x: x.tick)

    return note_ons, tempo_changes, volume_changes

def calc_audio_events(note_ons, tempo_changes, volume_changes, ticks_per_beat):
    # initialize results array
    audio_events = []

    # initial tempo
    tempo = 500000
    tcp = 0 # tempo change pointer
    accumulated_time = 0
    accumulated_ticks = 0

    # initial volume
    volume = 100
    vcp = 0 # volume change pointer

    # add noteOns with same tick
    for note_on in note_ons:
        tick = note_on.tick

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
        event_volume = (note_on.velocity / 127 * 0.5 + 0.5) * volume / 127

        audio_events.append(AudioEvent(event_time, note_on.note, event_volume))

    audio_events.sort(key=lambda x: x.time)

    return audio_events

def write_audio_events(audio_events, outpath):
    print('Writing audio events to file')
    # write audio events to a file
    events_file_path = os.path.join(outpath, 'events.csv')
    with open(events_file_path, 'w', encoding='utf-8') as event_file:
        for event in audio_events:
            event_file.write(event.to_csv_line() + '\n')

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
        'timidity', 
        midifile,
        '-Ow', 
        '-o', wavfile,
        '-s', str(samplerate),
        '-A', str(args.bits),
        '-x', f"soundfont '{soundfont.path}'",
        '--preserve-silence'
    ]

    print(f"{' '.join(renderCommand)}")
    with subprocess.Popen(
        renderCommand,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
        ) as renderProcess:

        # timidity_process.stdout.close()  # Close timidity's output stream to indicate it's done
        render_stdout, render_stderr = renderProcess.communicate()  # Get timidity's output if needed
        print(f'\nRendering terminated for {midifile}')
        if renderProcess.returncode != 0:
            print('Process returned non-zero exit code: ' + str(renderProcess.returncode))
        render_output = render_stdout.decode('utf-8')
        if render_output != '':
            print('Render output' + render_output)
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
                    print(f"[{count}] processing {artist}/{song}")

                    songname = song[0:-4]
                    outpath = args.out_path + '/' + artist + '/' + songname + '/'

                    soundfont = soundfonts[processed_count % len(soundfonts)]

                    generate_song(midifile, outpath, [soundfont], threads)
                    processed_count += 1

                count += 1

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
                if file[-4:] == '.sf2': # timidity does not seem to handle sf3 files very well
                    name = file[0:-4]
                    print(name)
                    path = os.path.join(root, file)
                    soundfonts.append(Soundfont(name, path))
    return soundfonts


if __name__ == '__main__':
    main()
