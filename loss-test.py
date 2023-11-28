import torch
import torch.nn as nn

labels = [0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]

# map labels: 
rand = torch.rand(len(labels))
estimationsA = list(map(lambda key, value: 0.25 * (1 * value + 3 * rand[key]), labels, rand))

# zeros
estimationsB = torch.zeros(len(labels))

# 
true_labels = torch.tensor(
    labels, 
    dtype=torch.float32
)

predicted_probsA = torch.tensor(
    estimationsA,
    dtype=torch.float32
)

predicted_probsB = torch.tensor(
    estimationsB,
    dtype=torch.float32
)

# Compute Binary Cross-Entropy Loss for each value independently
lossA = nn.functional.binary_cross_entropy(predicted_probsA, true_labels)
lossB = nn.functional.binary_cross_entropy(predicted_probsB, true_labels, reduction='sum')

# Print the loss for each value
print(f"LossA: {lossA} should be smaller than LossB: {lossB}")
