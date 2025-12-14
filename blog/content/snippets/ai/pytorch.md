---
title: "PyTorch Essentials"
date: 2024-12-12
draft: false
category: "ai"
tags: ["ai-knowhow", "pytorch", "deep-learning", "python"]
---


Essential PyTorch operations and patterns for deep learning.

---

## Installation

```bash
# CPU version
pip install torch torchvision torchaudio

# GPU version (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Check installation
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

---

## Basic Tensor Operations

```python
import torch

# Create tensors
scalar = torch.tensor(42)
vector = torch.tensor([1, 2, 3])
matrix = torch.tensor([[1, 2], [3, 4]])

# Random tensors
x = torch.rand(3, 4)
y = torch.randn(3, 4)  # Normal distribution
z = torch.zeros(3, 4)
ones = torch.ones(3, 4)

# Operations
a = torch.tensor([[1., 2.], [3., 4.]])
b = torch.tensor([[5., 6.], [7., 8.]])

c = a + b
d = a * b  # Element-wise
e = torch.matmul(a, b)  # Matrix multiplication
f = a @ b  # Also matrix multiplication

# GPU operations
if torch.cuda.is_available():
    device = torch.device('cuda')
    a = a.to(device)
    b = b.to(device)
    c = a + b
```

---

## Building Models

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(-1, 784)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

model = SimpleNet()
```

---

## Training Loop

```python
import torch.optim as optim

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
for epoch in range(num_epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}')

# Evaluation
model.eval()
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        # ... calculate metrics
```

---

## DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

dataset = CustomDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
```

---

## Saving and Loading

```python
# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model = SimpleNet()
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Save entire model
torch.save(model, 'entire_model.pth')
loaded_model = torch.load('entire_model.pth')

# Save checkpoint
torch.save({
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': loss,
}, 'checkpoint.pth')
```

---

## TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/experiment_1')

for epoch in range(num_epochs):
    # ... training ...
    writer.add_scalar('Loss/train', loss.item(), epoch)
    writer.add_scalar('Accuracy/train', accuracy, epoch)

writer.close()

# View: tensorboard --logdir=runs
```

---