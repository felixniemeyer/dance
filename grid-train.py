import os

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

cnn_first_layer_feature_sizes = [32]
cnn_activation_functions = ['relu', 'tanh', 'sigmoid']

rnn_hidden_sizes = [96]
rnn_layer_counts = [3]


variation_count = len(learning_rates) * len(cnn_first_layer_feature_sizes) * len(rnn_hidden_sizes) * len(rnn_layer_counts) * len(loss_functions) * len(cnn_activation_functions)

print(f"Training params forwarded to the train script: {args.train_parameters}")
print(f"Running {variation_count} variations. Ok?")

answer = input("y/n: ")

if answer.lower() != 'y':
    print("Aborting.")
    exit(0)

# run
for learning_rate in learning_rates:
    for cnn_first_layer_feature_size in cnn_first_layer_feature_sizes:
        for rnn_hidden_size in rnn_hidden_sizes:
            for rnn_layer_count in rnn_layer_counts:
                for loss_function in loss_functions:
                    for cnn_activation_function in cnn_activation_functions:
                        variation = f"r{learning_rate}_cnn{cnn_first_layer_feature_size}{cnn_activation_function}_rnn{rnn_layer_count}x{rnn_hidden_size}_{loss_function}"

                        cmd = "python train.py "

                        cmd += f"--learning-rate {learning_rate} "
                        cmd += f"--cnn-first-layer-feature-size {cnn_first_layer_feature_size} "
                        cmd += f"--rnn-hidden-size {rnn_hidden_size} "
                        cmd += f"--rnn-layers {rnn_layer_count} "
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

