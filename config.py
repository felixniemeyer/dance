# Terminology: 
# chunk = a piece of audio (e.g. 16 seconds long)
# frame = equally sized chunks of audio
# sequence = a sequence of frames

# audio config
bitrate = 16
samplerate = 44100

chunk_duration = 8
channels = 1
frame_size = 512 # 86 FPS


# see https://www.zem-college.de/midi/mc_tabed.htm
note_labels = {
    35: "Acoustic Bass Drum", 
    36: "Bass Drum 1", 
    38: "Acoustic Snare", 
    40: "Electric Snare"
}

# make keys into an array
relevant_notes = [i for i in note_labels.keys()]
