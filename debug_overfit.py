import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import os

# Import your model and dataset
from src.tcformer import TCFormer
from train import MTSCDataset 

# --- CONFIG ---
DATA_PATH = "/content/work_dir/data/processed/Heartbeat/ReplValChng/train"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def overfit_one_batch():
    print("🧪 STARTING STABILIZED OVERFIT TEST")
    print("-----------------------------------")
    
    # 1. Load Data
    ds = MTSCDataset(DATA_PATH)
    if len(ds) == 0: return

    # 2. Pick 16 Balanced Images
    indices_0 = [i for i, label in enumerate(ds.labels) if label == 0]
    indices_1 = [i for i, label in enumerate(ds.labels) if label == 1]
    batch_indices = indices_0[:8] + indices_1[:8]
    
    tiny_ds = Subset(ds, batch_indices)
    loader = DataLoader(tiny_ds, batch_size=16, shuffle=False)
    
    print(f"🎯 Selected {len(tiny_ds)} images.")

    # 3. Initialize Model (Patch Size 8 = Sweet Spot)
    model = TCFormer(
        img_size=224,
        in_chans=61,
        num_classes=2,
        k=5,
        sample_ratios=[0.5, 0.5, 0.5], # Less aggressive clustering
        patch_size=8,  # Changed from 4 to 8 for stability
        stride=8
    ).to(DEVICE)
    
    # 4. GENTLER Optimizer
    # Lower LR (1e-4) prevents the oscillation you saw
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    print("🚀 Training with Normalize + Lower LR...")
    for epoch in range(1, 201): # Give it more epochs to converge slowly
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            # --- CRITICAL FIX: MANUAL NORMALIZATION ---
            # Shift from [0, 1] to [-0.5, 0.5] approximately
            # This helps gradients flow better
            images = images - 0.5 
            
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            
            # Gradient Clipping (Prevents exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            preds = torch.argmax(output, dim=1)
            acc = (preds == labels).float().mean() * 100
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}: Loss = {loss.item():.5f} | Acc = {acc.item():.1f}%")
                
            if acc.item() == 100.0:
                print(f"\n✅ SUCCESS at Epoch {epoch}! Loss: {loss.item():.5f}")
                print("   Preds: ", preds.tolist())
                print("   Truth: ", labels.tolist())
                return

    print("\n❌ FAILED. Final Acc:", acc.item())
    print("   Preds: ", preds.tolist())

if __name__ == "__main__":
    overfit_one_batch()