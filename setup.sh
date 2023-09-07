echo 'making directories'
if [ ! -d "data" ]; then
  mkdir data
fi
if [ ! -d "soundfonts" ]; then
  mkdir soundfonts
fi

# if /data/midi/lakh_clean doesn't exist, download it
if [ ! -d "data/midi/lakh_clean" ]; then
  echo 'downloading lakh clean midi file dataset'
  cd data
  mkdir midi
  cd midi
  wget 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz'
  tar xf clean_midi.tar.gz
  rm clean_midi.tar.gz
  mv clean_midi lakh_clean
  cd ..
  cd ..
fi

###
echo 'downloading soundfonts'
cd soundfonts

# found a list of soundfonts here: https://github.com/FluidSynth/fluidsynth/wiki/SoundFont
if [ ! -d "GeneralUser GS 1.471" ]; then
  echo 'downloading GeneralUser soundfont'
  wget 'https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_1.471.zip?dl=1' -O 'GeneralUser_GS_1.471.zip'
  unzip 'GeneralUser_GS_1.471.zip'
  rm 'GeneralUser_GS_1.471.zip'
fi

if [ ! -f "FluidR3_GM.sf2" ]; then
  echo 'downloading FluidR3_GM soundfont'
  wget 'https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2' 
fi

# and two more fonts from here: https://musescore.org/en/handbook/3/soundfonts-and-sfz-files#list
if [ ! -d "Arachno" ]; then
  echo 'downloading Arachno soundfont'
  mkdir Arachno
  cd Arachno
  wget 'http://maxime.abbey.free.fr/mirror/arachnosoft/files/soundfonts/arachno-soundfont-10-sf2.zip' 
  unzip 'arachno-soundfont-10-sf2.zip'
  rm 'arachno-soundfont-10-sf2.zip'
  cd ..
fi

if [ ! -f 'MS_Basic.sf3' ]; then
  echo 'downloading MS Basic soundfont'
  wget 'https://github.com/musescore/MuseScore/raw/master/share/sound/MS%20Basic.sf3' -O 'MS_Basic.sf3'
fi

cd ..

###
if [ ! -d 'venv' ]; then
  echo 'setting up venv'
  python -m venv venv
fi

source venv/bin/activate

###
echo 'installing requirements'
pip install -r requirements.txt

###
echo 'done setting up'
echo 'you can now start the virtual env with `source venv/bin/activate`'
echo 'and run the program with `python render_midi.py`'
