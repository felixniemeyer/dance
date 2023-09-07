echo 'making directories'
mkdir data
mkdir soundfonts

echo 'downloading midi files'
cd data
mkdir midi
cd midi
wget 'http://hog.ee.columbia.edu/craffel/lmd/clean_midi.tar.gz'
tar xf clean_midi.tar.gz
rm clean_midi.tar.gz
mv clean_midi lakh_clean
cd ..
cd ..

echo 'downloading soundfonts'
cd soundfonts

# found a list of soundfonts here: https://github.com/FluidSynth/fluidsynth/wiki/SoundFont

wget 'https://www.dropbox.com/s/4x27l49kxcwamp5/GeneralUser_GS_1.471.zip?dl=1' -O 'GeneralUser_GS_1.471.zip'
unzip 'GeneralUser_GS_1.471.zip'
rm 'GeneralUser_GS_1.471.zip'

wget 'https://musical-artifacts.com/artifacts/738/FluidR3_GM.sf2' 

# and two more fonts from here: https://musescore.org/en/handbook/3/soundfonts-and-sfz-files#list

mkdir Arachno
cd Arachno
wget 'http://maxime.abbey.free.fr/mirror/arachnosoft/files/soundfonts/arachno-soundfont-10-sf2.zip' 
unzip 'arachno-soundfont-10-sf2.zip'
rm 'arachno-soundfont-10-sf2.zip'
cd ..

wget 'https://github.com/musescore/MuseScore/raw/master/share/sound/MS%20Basic.sf3' -O 'MS_Basic.sf3'

echo 'setting up venv'
python -m venv venv

echo 'installing requirements'
source venv/bin/activate
pip install -r requirements.txt
