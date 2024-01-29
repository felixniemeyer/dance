import torch.nn as nn
import torch

# create a test tensor with layout [batch, seq_len, frame_size]
batch_size = 2
seq_len = 3
frame_size = 5

test_tensor = (torch.rand(batch_size, seq_len, frame_size) * 10).int()

print('test_tensor shape:', test_tensor.shape)
print('test_tensor:', test_tensor)

only_frames = test_tensor.view(-1, frame_size)

print('only_frames shape:', only_frames.shape)
print('only_frames:', only_frames)

# shape back
back_to_original = only_frames.view(batch_size, seq_len, frame_size)

print('back_to_original shape:', back_to_original.shape)
print('back_to_original:', back_to_original)
