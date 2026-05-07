import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import time
from collections import Counter

# Import your custom modules
from src.tcformer import TCFormer
from configs.config import Config

# --- 1. Standard Dataset Loader ---
class MTSCDataset(Dataset):
    def __init__(self, root_dir):
        self.file_paths = []
        self.labels = []
        
        if not os.path.exists(root_dir):
            return

        class_folders = sorted(os.listdir(root_dir))
        for class_id in class_folders:
            class_path = os.path.join(root_dir, class_id)
            if not os.path.isdir(class_path): continue
                
            files = glob.glob(os.path.join(class_path, "*.npy"))
            for f in files:
                self.file_paths.append(f)
                self.labels.append(int(class_id))

    def __len__(self): return len(self.file_paths)
    
    def __getitem__(self, idx):
        img = np.load(self.file_paths[idx])
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        return img_tensor, self.labels[idx]

# --- 2. Training Functions ---
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        
        # 1. Normalize (Center around 0)
        images = images - 0.5 
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # 2. Gradient Clipping (Prevents crashes/oscillations)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return running_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion, device, print_stats=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            images = images - 0.5 # Consistent Normalization
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if print_stats:
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            
    if total == 0: return 0.0, 0.0
    
    # --- DEBUG STATS ---
    if print_stats:
        print(f"\n[DEBUG] Stats:")
        # Calculate Per-Class Accuracy
        classes = sorted(list(set(all_labels)))
        for cls in classes:
            total_c = all_labels.count(cls)
            correct_c = sum(1 for p, t in zip(all_preds, all_labels) if p == cls and t == cls)
            acc_c = (correct_c / total_c * 100) if total_c > 0 else 0.0
            print(f" Class {cls}: {correct_c}/{total_c} ({acc_c:.1f}%)")
        
    return running_loss / len(loader), 100 * correct / total

# --- 3. The Master Loop ---
def main():
    os.makedirs("results", exist_ok=True)
    results_file = "results/final_results.csv"
    
    if not os.path.exists(results_file):
        df = pd.DataFrame(columns=["Dataset", "Variant", "Test_Accuracy", "Best_Epoch", "Training_Time_Sec"])
        df.to_csv(results_file, index=False)
    
    print(f"==================================================")
    print(f"   STARTING UNIVERSAL MASTER TRAINING LOOP")
    print(f"   Device: {Config.DEVICE}")
    print(f"==================================================\n")

    # LOOP 1: Iterate all Datasets in Config
    for dataset_name in Config.DATASETS:
        dataset_info = Config.DATASET_REGISTRY[dataset_name]
        
        # Load Dataset specific params
        in_channels = dataset_info['in_channels']
        num_classes = dataset_info['num_classes']
        patch_size = dataset_info['patch_size'] # <--- AUTO-LOADS 8 for Heartbeat, 16 for others
        stride = dataset_info['stride']
        
        print(f"--> Dataset: {dataset_name} (Chans: {in_channels}, Classes: {num_classes}, Patch: {patch_size})")
        print(f"   Beta: {Config.DATASET_REGISTRY[dataset_name].get('beta', 1.0)}")


        # LOOP 2: Iterate all Variants
        for variant in Config.VARIANTS:
            print(f"    --> Variant: {variant}")
            
            # 1. Prepare Data
            train_path = os.path.join(Config.DATA_PATH, dataset_name, variant, "train")
            test_path = os.path.join(Config.DATA_PATH, dataset_name, variant, "test")
            
            train_ds = MTSCDataset(train_path)
            test_ds = MTSCDataset(test_path)
            
            if len(train_ds) == 0:
                print(f"        [SKIP] No data found at {train_path}")
                continue
                
            train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
            test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
            
            # 2. Initialize Model
            model = TCFormer(
                img_size=Config.IMG_SIZE,
                in_chans=in_channels,
                num_classes=num_classes,
                k=Config.K_NEIGHBORS,
                sample_ratios=Config.SAMPLE_RATIOS,
                patch_size=Config.DATASET_REGISTRY[dataset_name].get('patch_size', 16),
                stride=Config.DATASET_REGISTRY[dataset_name].get('stride', 16),
                
                # --- NEW: Pass Dataset-Specific Beta ---
                beta=Config.DATASET_REGISTRY[dataset_name].get('beta', 1.0),
                use_cross_attention=Config.USE_CROSS_ATTENTION
            ).to(Config.DEVICE)
            
            # 3. Optimizer & Scheduler (The Winning Combo)
            optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.01)
            
            # Gentle Weights: Use weights ONLY if it's Heartbeat (2 classes) to help balance
            if dataset_name == "Heartbeat":
                cw = torch.tensor([1.0, 2.0]).to(Config.DEVICE)
                criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.1)
            else:
                # Standard Loss for balanced datasets
                criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
                
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

            # FOR UVGESTURES ONLY
            # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            #     optimizer, mode='max', factor=0.5, patience=10
            # )
            
            # 4. Train
            best_acc = 0.0
            best_epoch = 0
            start_time = time.time()
            
            pbar = tqdm(range(Config.EPOCHS), desc=f"Training {dataset_name}", leave=False)
            
            for epoch in pbar:
                train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
                val_loss, val_acc = evaluate(model, test_loader, criterion, Config.DEVICE, print_stats=True)
                
                scheduler.step(val_acc)
                
                if val_acc > best_acc:
                    best_acc = val_acc
                    best_epoch = epoch + 1
                
                pbar.set_postfix({"Loss": f"{train_loss:.3f}", "Curr Acc": f"{round(val_acc, 2)}%", "Best Acc": f"{best_acc:.2f}%"})
            
            total_time = time.time() - start_time
            print(f"        [DONE] Best Accuracy: {best_acc:.2f}% (Epoch {best_epoch})")
            
            # Log Results
            new_row = pd.DataFrame([{
                "Dataset": dataset_name,
                "Variant": variant,
                "Test_Accuracy": round(best_acc, 2),
                "Best_Epoch": best_epoch,
                "Training_Time_Sec": round(total_time, 2)
            }])
            new_row.to_csv(results_file, mode='a', header=False, index=False)

    print("\n==================================================")
    print(f"   ALL EXPERIMENTS COMPLETED.")
    print(f"   Results saved to: {results_file}")
    print("==================================================")

if __name__ == "__main__":
    main()