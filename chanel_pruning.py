import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Define a simple convolutional layer
conv_layer = nn.Conv2d(in_channels=20, out_channels=60, kernel_size=3)

# Randomly initialize weights
torch.nn.init.xavier_uniform_(conv_layer.weight)

# Print the original weight shape
print("Original weight shape:", conv_layer.weight.shape)

# Apply channel-wise pruning
prune.l1_unstructured(conv_layer, name='weight', amount=0.2)  # Prune 20% of channels

# Print the pruned weight shape
print("Pruned weight shape:", conv_layer.weight.shape)