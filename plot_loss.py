import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot loss')
parser.add_argument('csv', type=str, default='loss.csv', help='csv file')
args = parser.parse_args()

with open(args.csv, 'r') as f:
    reader = csv.reader(f)
    rows = [row for row in reader if row]

rows = np.array(rows, dtype=np.float32)

epochs = rows[:, 0]

if rows.shape[1] == 3:
    train_loss = rows[:, 1]
    val_loss   = rows[:, 2]
    plt.plot(epochs, train_loss, label='train')
    plt.plot(epochs, val_loss,   label='val')
    plt.legend()
else:
    # legacy: only val loss
    plt.plot(epochs, rows[:, 1], label='val')
    plt.legend()

plt.xlabel('epoch')
plt.ylabel('loss')
plt.title(args.csv)
plt.tight_layout()
plt.show()
