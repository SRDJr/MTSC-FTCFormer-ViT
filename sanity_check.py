import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.dataset import TSDataset # Your dataset class
from configs.config import Config

# 1. Simple Linear Model
class SimpleBaseline(nn.Module):
    def __init__(self, seq_len, num_classes):
        super().__init__()
        self.flat = nn.Flatten()
        self.linear = nn.Linear(seq_len * 61, num_classes) # 61 channels

    def forward(self, x):
        return self.linear(self.flat(x))

# 2. Load Data
train_ds = TSDataset(subset="train", dataset_name="Heartbeat")
train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

# 3. Quick Train
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SimpleBaseline(seq_len=405, num_classes=2).to(device) # Length is 405 for Heartbeat
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

print("🚀 Running Sanity Check...")
for epoch in range(5):
    model.train()
    total_acc = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        
        preds = torch.argmax(out, dim=1)
        total_acc += (preds == y).sum().item()
    
    print(f"Epoch {epoch+1}: Acc = {total_acc / len(train_ds):.4f}")