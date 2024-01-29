import os

import argparse

parser = argparse.ArgumentParser("make options directly in the code for now")

parser.add_argument('checkpoints_folder', type=str) 
parser.add_argument('name', type=str) 
parser.add_argument('epoch', type=str) 
parser.add_argument('input', type=str)

args = parser.parse_args()

# get directory of audio file
input_path = os.path.dirname(args.input)

# list all folders 
runs = os.listdir(args.checkpoints_folder + '/' + args.name)

for run in runs:
    cmd = f'python apply.py {args.checkpoints_folder}/{args.name}/{run}/{args.epoch}.pt {args.input} -o {input_path}/{args.name}-{run}'

    print(cmd)

    # run the command
    os.system(cmd)

