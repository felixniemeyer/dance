import os
import subprocess

import argparse

parser = argparse.ArgumentParser("make options directly in the code for now")

parser.add_argument('checkpoints_path', type=str) 
parser.add_argument('epoch', type=str) 

args = parser.parse_args()

# list all folders 
names = os.listdir(args.checkpoints_path)

checkpoints = []
for name in names:
    checkpoints.append(os.path.basename(name) + '/' + args.epoch + '.pt')

command = ['python', 'try_model.py', args.checkpoints_path] + checkpoints

print(command)

# run the command
with subprocess.Popen(command, stdout=subprocess.PIPE) as proc:
    proc.communicate()
    proc.wait()
