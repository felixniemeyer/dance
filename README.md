# Audio Detection NN

## To start training:

python train.py rnn_only -d 1000 -t snares

## To apply a model:

python apply.py checkpoints/cool/1.pt test2/MS_Basic.ogg
