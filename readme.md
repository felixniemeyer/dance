# dance

This repo trains a neural net to recognize important moments in a live music stream in real time. 
It's primary purpose is to make it easy to make visuals react nicely to music. 

# sources of groundtruth

[wip] from midifiles

[] from 4 to the floor tracks

[] manually

## from midi files

In midi files it is quite clear to determine the kicks and snares. 
They get send on channel 10 (which is represented by number 9) and have a specific note number.

That makes midi files a good source of ground truth.

For generating ground truth from midi data, you will need
- to install FluidSynth on your system, 
- lakh-midi-dataset-clean, the songs in midi format, and
- some soundfonts in a `./soundfonts` directory

## from 4 to the floor tracks
Often, house tracks have 4 kicks per bar and a snare on every second bar.
Doing a BPM detection and then looking for the kicks and snares is a possibility to get some real music ground truth.

## manually
Maybe for finetuning or if someone is bored.



