DATA_VERSION=$1
CHUNKS=$2

source venv/bin/activate

python annotate.py --music-path /mnt/datengrab/archive/music/ --out-path chunks/manual/$DATA_VERSION --chunks-per-song 9 & 

python inspector.py --chunks-path chunks/manual/$DATA_VERSION --checkpoints-path checkpoints --port 8051 & 

npm run dev 


