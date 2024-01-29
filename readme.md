# dance

This repo trains a neural net for recognizing percussion events in a music stream in real time. 
It's original use case is to make it easy to make visuals react nicely to music. 
## most scripts use argparse
You can show a description and list of parameters with 
```
python <script.py> -h
```

## directory structure

- ./checkpoints: contains checkpoints created during training
- ./data: contains training data related data
- ./data/chunks: contains chunks for training
- ./meta: ideas and considerations
- ./models: python code of different nn architectures

## sources of training data
For training, we need small samples of audio along with the timestamps of the relevant audio events. Some sort of note velocity or even a "presence" value that decays after the onset could be helpful. 

[wip] from midifiles

[wip] from audio files

[] from 4 to the floor tracks

[] manually

### from midi files

In midi files it is quite clear to determine the kicks and snares. 
They get send on channel 10 (which is represented by number 9) and have a specific note number.

That makes midi files a good source of ground truth.

For generating ground truth from midi data, you will need
- to install FluidSynth on your system, 
- lakh-midi-dataset-clean, the songs in midi format, and
- some soundfonts in a `./soundfonts` directory

### from audio tracks
We basically create a non-realtime workflow in order to create training data for training the realtime model. 

We use demucs to split tracs horizontally into drum, vocals, bass and the rest. 
Then we use onset detection on the drum track and classify the shreds. 

For classifying the shreds we train another neural network. We train it using percussion sounds from sample packs. Because shreds can be very short depening on then the next onset appears, we artificially shorten the samples from the sample bank (maybe among other augmentation techniques like repitching to increase training data diversity). 

### from 4 to the floor tracks
Often, house tracks have 4 kicks per bar and a snare on every second bar.
Doing a BPM detection and then looking for the kicks and snares is a possibility to get some real music ground truth.

### manually
Manually labelling the percussion events in an audio file is also an option. 
One could use a DAW to create a synchronous midi track for the audio and then convert the exported midi to a timestamp
Maybe interesting for finetuning it to a specific album. 


