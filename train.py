import torch
import torch.nn as nn
import torch.optim as optim

from dance_data import DanceDataset
from dancer_model import DancerModel

from torch.utils.data import Dataset, DataLoader, random_split

import config

batch_size = 1
learning_rate = 1e-3

# Assuming you have prepared your dataset and DataLoader
dataset = DanceDataset("./chunks")
print('dataset size: ', len(dataset))

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size)
val_loader = DataLoader(val_dataset)

# Initialize the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model from disk if it exists
model = DancerModel().to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    for batch_inputs, batch_labels in train_loader:
        batch_inputs, batch_labels = batch_inputs.to(device), batch_labels.to(device)

        # Zero the gradients from previous iterations
        optimizer.zero_grad()

        # Forward pass
        outputs = model(batch_inputs)

        # Compute the loss
        loss = criterion(outputs, batch_labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

    # Validation after each epoch
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        total_correct = 0
        total_samples = 0
        for val_inputs, val_labels in val_loader:
            val_inputs, val_labels = val_inputs.to(device), val_labels.to(device)

            # Forward pass
            val_outputs = model(val_inputs)

            # Count correct predictions
            _, predicted_labels = torch.max(val_outputs, 1)
            total_correct += (predicted_labels == val_labels).sum().item()
            total_samples += val_labels.size(0)

        accuracy = total_correct / total_samples
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Accuracy: {accuracy:.4f}")

# Training completed

