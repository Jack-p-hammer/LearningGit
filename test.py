import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
import pandas as pd
import os

# Make a single layer neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.fc(x)
    
# Generate some random data
x = np.linspace(0, 15, 100)
# Generate a sine wave with an exponential decay
v0 = 0
v1 = 1
wn = .5
zeta = 0.25
y = v1 + (v0-v1)*np.cos(wn*x)*np.exp(-zeta*x)

# Convert to PyTorch tensors
x_tensor = torch.tensor(x, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Create a dataset and dataloader
dataset = data.TensorDataset(x_tensor, y_tensor)
dataloader = data.DataLoader(dataset, batch_size=10, shuffle=True)

# Initialize the model, loss function and optimizer
model = SimpleNN(input_size=1, output_size=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


