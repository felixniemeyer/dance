import mido

# Load MIDI file into memory
midi_file = mido.MidiFile('lakh-midi-dataset-clean/ABBA/Gimme_Gimme_Gimme.mid')
 
midi_file.print_tracks()

