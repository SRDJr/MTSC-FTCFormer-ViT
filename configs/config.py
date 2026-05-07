import torch
import os
import sys

class Config:
    # --- 1. ENVIRONMENT DETECTION ---
    IS_COLAB = 'google.colab' in sys.modules

    # --- 2. PATH SETUP ---
    # NEW: We are saving to 'processed_batch2' to keep it separate from the old data
    if IS_COLAB:
        BASE_DIR = '/content/work_dir'
        DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_batch2')
        # DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
        RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')
    else:
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed_batch2')
        # DATA_PATH = os.path.join(BASE_DIR, 'data', 'processed')
        RAW_DATA_PATH = os.path.join(BASE_DIR, 'data', 'raw')

    # --- 3. GLOBAL HYPERPARAMETERS ---
    IMG_SIZE = 224
    BATCH_SIZE = 16
    LEARNING_RATE = 1e-4  
    EPOCHS = 50
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- NEW: FUZZY PIPELINE CONTROLS ---
    USE_FUZZY = True      # Set to False to run standard Crisp ViT baseline
    USE_SOFTMAX = False    # True = Phase 1 (FANTF Style). False = Phase 2 (Softmax-Free)
    USE_CROSS_ATTENTION = False
    # FTCFormer Defaults
    K_NEIGHBORS = 5
    SAMPLE_RATIOS = [0.25, 0.25, 0.25]

    # --- 4. DATASET REGISTRY (The "Brain") ---
    DATASET_REGISTRY = {
        # --- OLD DATASETS (Commented Out) ---
        # "BasicMotions": { 
        #   "in_channels": 6, 
        #   "num_classes": 4, 
        #   "patch_size": 16, 
        #   "stride": 16, 
        #   "beta": 1.0 
        # }
        # "Cricket": { 
        #   "in_channels": 6, 
        #   "num_classes": 12, 
        #   "patch_size": 16, 
        #   "stride": 16, 
        #   "beta": 1.0 
        # }
        # "Epilepsy": { 
        #   "in_channels": 3, 
        #   "num_classes": 4, 
        #   "patch_size": 8, 
        #   "stride": 8, 
        #   "beta": 5.0
        # }
        # "UWaveGestureLibrary": { 
        #   "in_channels": 3, 
        #   "num_classes": 8, 
        #   "patch_size": 14, 
        #   "stride": 14, 
        #   "beta": 1.0 
        # }
        # "Heartbeat": { 
        #   "in_channels": 61, 
        #   "num_classes": 2, 
        #   "patch_size": 8, 
        #   "stride": 8, 
        #   "beta": 1.0
        # }

        # --- NEW DATASETS (Batch 2) ---
        "CharacterTrajectories": { 
            "in_channels": 3, 
            "num_classes": 20,
            "patch_size": 16, "stride": 16,   # Smooth handwriting curves
            "beta": 1.0                       # Relaxed filtering for clean data
        }
        # "EthanolConcentration": { 
        #     "in_channels": 3, 
        #     "num_classes": 4,
        #     "patch_size": 16, "stride": 16,     # Chaotic sensor data
        #     "beta": 1.0                   # Strict filtering for noise
        # }
        # "LSST": { 
        #     "in_channels": 6, 
        #     "num_classes": 14,
        #     "patch_size": 8, "stride": 8,     # Short sequence, small patches needed
        #     "beta": 1.0                       # Balanced filtering
        # }
        # "ArticularyWordRecognition": { 
        #     "in_channels": 9, 
        #     "num_classes": 25,
        #     "patch_size": 16, "stride": 16,   
        #     "beta": 1.0                      
        # }
        # "AtrialFibrillation": { 
        #     "in_channels": 2, 
        #     "num_classes": 3,
        #     "patch_size": 16, "stride": 8,     # Spiky ECG data (like Epilepsy)
        #     "beta": 1.0                       # Strict filtering
        # }
    }

    # --- 5. TRAINING SELECTION ---
    DATASETS = list(DATASET_REGISTRY.keys()) 
    
    # We will process all variants just like before
    # VARIANTS = [
    #    'Val_ValChng', 'ValChng_ChngValChng', 'Values_x_Values', 
    #    'ReplVal', 'ReplValChng', 'TSSI', 'WSI'
    # ]
    VARIANTS = [
       'Values_x_Values'
    ]