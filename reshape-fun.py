import torch.nn as nn
import torch

# create a test tensor with layout [batch, seq_len, buffer_size]
batch_size = 2
seq_len = 3
buffer_size = 5

test_tensor = (torch.rand(batch_size, seq_len, buffer_size) * 10).int()

print('test_tensor shape:', test_tensor.shape)
print('test_tensor:', test_tensor)

only_buffers = test_tensor.view(-1, buffer_size)

print('only_buffers shape:', only_buffers.shape)
print('only_buffers:', only_buffers)

# shape back
back_to_original = only_buffers.view(batch_size, seq_len, buffer_size)

print('back_to_original shape:', back_to_original.shape)
print('back_to_original:', back_to_original)
