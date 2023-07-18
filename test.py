from midi2audio import FluidSynth

fs = FluidSynth('soundfonts/FluidR3_GM.sf2') 

# fs.midi_to_audio('lakh-midi-dataset-clean/ABBA/The_Winner_Takes_It_All.1.mid', 'out-test.wav') 
interpret = 'ABBA'
song = 'Gimme_Gimme_Gimme.mid'
fs.midi_to_audio('lakh-midi-dataset-clean/' + interpret + '/' + song, song + '.wav') 


