import os
import time

import argparse

parser = argparse.ArgumentParser("make options directly in the code for now")

parser.add_argument('name', type=str) 
parser.add_argument('--train-parameters', type=str)
parser.add_argument('--log-directory', type=str, default='gridlogs')
parser.add_argument('--checkpoints-directory', type=str, default='checkpoints/grids')

args = parser.parse_args()

# remove trailing slashes
if args.log_directory[-1] == '/':
    args.log_directory = args.log_directory[:-1]

if args.checkpoints_directory[-1] == '/':
    args.checkpoints_directory = args.checkpoints_directory[:-1]

# mkdir gridlog if it does not exist
logdir = args.log_directory + '/' + args.name
if not os.path.exists(logdir):
    os.makedirs(logdir)

# parameter sets
learning_rates = [0.007]
loss_functions = ['mse', 'smoothl1', 'bcew']
models = ['cnn_only', 'rnn_only', 'cnn_and_rnn', 'cnn_and_rnn_and_funnel']

variation_count = len(learning_rates) * len(loss_functions) * len(models)

print(f"Training params forwarded to the train script: {args.train_parameters}")
print(f"Running {variation_count} variations. Ok?")

answer = input("y/n: ")

if answer.lower() != 'y':
    print("Aborting.")
    exit(0)

start_time = time.time()

# run
for learning_rate in learning_rates:
    for loss_function in loss_functions:
        for model in models:
            variation = f"r{learning_rate}_l{loss_function}_m{model}"

            cmd = f"python train.py {model} "

            cmd += f"--learning-rate {learning_rate} "
            cmd += f"--loss-function {loss_function} "

            cmd += f"--checkpoints-path '{args.checkpoints_directory}/{args.name}' "
            cmd += f"--tag {variation} "
            
            cmd += f"--summarize "

            cmd += args.train_parameters

            print(cmd)

            # write output of train.py to a file
            cmd += f" > '{logdir}/{variation}.txt'"

            # run the command
            os.system(cmd)

end_time = time.time()
print(f"Total time: {end_time - start_time} seconds")
