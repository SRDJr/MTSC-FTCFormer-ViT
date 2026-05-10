# generate_data.py
import os
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from aeon.datasets import load_classification
from src.image_converter import convert_to_image
from configs.config import Config
import shutil

def process_single_sample(x, y, idx, variant, save_dir, class_map):
    """
    Worker function to process one sample.
    """
    try:
        # 1. Convert Time Series to Image (H, W, C)
        img_tensor = convert_to_image(x, method=variant, target_size=Config.IMG_SIZE)
        
        # 2. Get Label ID
        label_str = str(y)
        label_idx = class_map.get(label_str)
        
        if label_idx is None:
            return False

        # 3. Save as .npy (Lossless)
        # Structure: /Dataset/Variant/Split/Class/sample_0.npy
        filename = f"sample_{idx}.npy"
        file_path = os.path.join(save_dir, str(label_idx), filename)
        
        np.save(file_path, img_tensor)
        return True
    except Exception as e:
        print(f"Error processing sample {idx}: {e}")
        return False

def process_dataset_variant(dataset_name, variant):
    """
    Process a full dataset for a specific variant.
    """
    print(f"\n[INFO] Starting: {dataset_name} - {variant}")
    
    try:
        # Load Data (Train & Test)
        # FIX: Removed 'return_type="numpy3d"' as it is deprecated in new aeon versions.
        # Aeon now returns numpy arrays by default for equal-length datasets.
        
        X_train, y_train = load_classification(
            dataset_name, 
            split="train", 
            extract_path=Config.RAW_DATA_PATH
        )
        X_test, y_test = load_classification(
            dataset_name, 
            split="test", 
            extract_path=Config.RAW_DATA_PATH
        )
        
        # Merge to get all unique classes for consistent mapping (Train+Test)
        all_labels = np.unique(np.concatenate([y_train, y_test]))
        class_map = {str(label): i for i, label in enumerate(all_labels)}
        
        # Process Splits
        for split_name, X, y in [("train", X_train, y_train), ("test", X_test, y_test)]:
            # Create Target Directory
            # e.g., data/processed/BasicMotions/ReplVal/train
            base_dir = os.path.join(Config.DATA_PATH, dataset_name, variant, split_name)
            
            # Create class subfolders
            for label_id in class_map.values():
                os.makedirs(os.path.join(base_dir, str(label_id)), exist_ok=True)
            
            # Run Parallel Processing (Fast)
            # n_jobs=-1 uses all available CPU cores
            Parallel(n_jobs=-1)(
                delayed(process_single_sample)(
                    X[i], y[i], i, variant, base_dir, class_map
                ) for i in tqdm(range(len(X)), desc=f"{dataset_name}-{variant}-{split_name}")
            )
            
    except Exception as e:
        print(f"[ERROR] Failed processing {dataset_name}: {e}")

if __name__ == "__main__":
    # Ensure Base Data Path Exists
    if not os.path.exists(Config.DATA_PATH):
        os.makedirs(Config.DATA_PATH)
    if not os.path.exists(Config.RAW_DATA_PATH):
        os.makedirs(Config.RAW_DATA_PATH)
    
    # --- THE MASTER LOOP ---
    print(f"Processing {len(Config.DATASETS)} datasets x {len(Config.VARIANTS)} variants...")
    
    for dataset in Config.DATASETS:
        for variant in Config.VARIANTS:
            process_dataset_variant(dataset, variant)
            
    
    # Dynamically extract the folder name from your config path
    target_folder = os.path.basename(os.path.normpath(Config.DATA_PATH))
    print(f"\n[SUCCESS] All datasets and variants processed into '{target_folder}' folder.")
    
    # --- AUTO-ZIP FOR FAST GOOGLE DRIVE SYNC ---
    # To be uncommented if one wants to store the zip files for the data/processed
    # print(f"\n[INFO] Zipping the '{target_folder}' folder for fast Drive sync...")
    
    # # Automatically append '_archive' to your exact config path
    # zip_path = f"{Config.DATA_PATH}_archive"
    
    # # This creates the archive dynamically based on config
    # shutil.make_archive(zip_path, 'zip', Config.DATA_PATH)
    
    # print(f"[SUCCESS] Zipping complete! File saved as: {zip_path}.zip")