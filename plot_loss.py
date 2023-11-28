import matplotlib.pyplot as plt

# read csv 
import csv
import numpy as np

import argparse

parser = argparse.ArgumentParser(description='Plot loss')
parser.add_argument('csv', type=str, default='loss.csv', help='csv file')

args = parser.parse_args()

# read csv
with open(args.csv, 'r') as f:
    reader = csv.reader(f)
    loss = list(reader)
    # transpose 
    loss = np.array(loss).T.astype(np.float32)

# plot 
plt.plot(loss[0], loss[1])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.show()

