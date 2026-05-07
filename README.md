# Transforming Time-Series to Images: A Fuzzy Attention Approach (via FTCFormer)

This repository contains the official codebase for transforming Multivariate Time-Series Classification (MTSC) tasks into computer vision problems, processed by a highly customized, noise-resilient Vision Transformer (ViT). 

This work builds upon the foundational **FTCFormer** (Fuzzy Token Clustering Transformer) architecture, introducing novel **Learnable Gaussian Attention** mechanisms specifically engineered for chaotic sensor data.

## Overview
Deep learning models often struggle with the raw, non-linear, and chaotic nature of multivariate time-series data. 
**Our Approach:** Instead of feeding 1D signals directly into a network, we map the time-series data into 2D spatial representations (images) using various mathematical transformations. This allows us to leverage the powerful pattern-recognition capabilities of Vision Transformers while using Fuzzy Logic to aggressively filter out inherent signal noise.


## 1. Architecture & Methodology

### The Problem: Time-Series to Image Translation
Deep learning models often struggle with the raw, non-linear, and chaotic nature of multivariate time-series data. 
**Our Approach:** Instead of feeding 1D signals directly into a network, we map the time-series data into 2D spatial representations (images) using various mathematical transformations (e.g., `Values_x_Values`). This allows us to leverage the powerful pattern-recognition capabilities of Vision Transformers, treating temporal fluctuations as visual textures.

### The Engine: FTCFormer with Modifications
Standard Vision Transformers (and even the baseline FTCFormer) are optimized for natural images with sharp edges and clear physical boundaries. Because our "images" represent mathematical frequencies and gradients, we fundamentally modified the architecture:

1. **Fuzzy Self-Attention (Phase 1 & Phase 2):** We replaced standard crisp Multi-Head Attention with a custom `LearnableGaussianAttention` module. Each attention head learns its own target ($\mu$) and strictness ($\sigma$) parameters to aggressively filter out noise inherent in time-series data. 
    * *Phase 1:* FANTF-style normalization using softmax.
    * *Phase 2:* Softmax-free, independent feature gating.
2. **Adaptive Density Clustering:** Inside the CTM (Clustering Token Merging) downsampling module, we injected a dataset-specific, hyperparameter $\beta$ to mathematically stabilize the clustering of token distances before merging.
3. **Optimized Spatial Processing:** We explicitly bypassed legacy visual components—such as Spatial Reduction (SR) layers and Convolutional Positional Encodings (CPEG)—forcing the network to group tokens strictly by their raw semantic values rather than superficial 2D grid coordinates.


## 2. Repository Structure

```text
mtsc-ftcformer/
├── configs/
│   └── config.py                # The Master Switchboard
├── data/                        # (Not tracked in Git)
│   ├── raw/                     # Downloaded aeon time-series datasets
│   └── processed/               # Generated image tensor archives
├── results/
│   └── final_results.csv        # Auto-generated training metrics
├── src/
│   ├── image_converter.py       # Mathematical 1D-to-2D transformations
│   ├── tcformer.py              # Top-level model wrapper
│   ├── tcformer_layers.py       # Core logic: CTM, Gaussian Attention
│   ├── tcformer_utils.py        # Baseline clustering math
│   └── transformer_utils.py     # Standard ViT utilities
├── .gitignore
├── debug_overfit.py             # Diagnostics tool
├── generate_data.py             # Data factory pipeline
├── README.md                    
├── requirements.txt             # Exact environment dependencies
├── sanity_check.py              # Architecture validation tool
├── train_fuzzy.py               # Master training loop (Thesis logic)
└── train.py                     # Legacy/Baseline training loop
```

**NOTE:** Due to GitHub's file size restrictions, the data/ directory is intentionally excluded from version control.

- **`data/raw/ \& data/processed/`**: You do not need to create these folders manually. Running the generate_data.py script will automatically generate these folders and populate the `data/raw` folder  with the downloaded aeon time-series datasets and the `data/processed` folder with the 2D image representations.

#### IMPORTANT FILES

* **`configs/config.py` (The Master Switchboard)**: The central control hub of the project. It features automatic environment detection (seamlessly switching paths between Google Colab and local execution) and manages global hyperparameters. Crucially, it houses the **Dataset Registry** (mapping specific channels, classes, and $\beta$ densities to each dataset) and the **Fuzzy Pipeline Controls**. This allows researchers to toggle between the Standard ViT baseline, Phase 1 (FANTF-style), and Phase 2 (Softmax-Free) architectures simply by flipping boolean flags.

* **`generate_data.py` (The Data Factory)**: Automates the entire pipeline of converting raw 1D time-series signals into 2D spatial image tensors. It fetches datasets via the `aeon` library, applies your specified mathematical transformations (e.g., `Values_x_Values`), and utilizes multi-core parallel processing (`joblib`) to rapidly generate lossless `.npy` files. Finally, it automatically compiles the structured output into a single `.zip` archive, optimizing the massive dataset for fast cloud storage synchronization.

* **`src/tcformer_layers.py` (The Mathematical Core)**: Houses the fundamental architectural innovations of this thesis. It implements the custom `LearnableGaussianAttention` and `LearnableGaussianCrossAttention` modules, driving the Phase 1 and Phase 2 fuzzy gating mechanisms via learnable $\mu$ and $\sigma$ parameters. It also contains the deeply modified `CTM` (Clustering Token Merging) downsampling block, featuring the injected dataset-specific $\beta$ density parameter and enhanced WSN (Weighted Shared Neighbor) assignments. Crucially, legacy spatial components from the original ViT (like SR layers and Convolutional FFNs) are explicitly bypassed here to optimize for non-spatial, chaotic signal processing.

* **`src/image_converter.py` (The Spatial Translator)**: Contains the core mathematical logic responsible for mapping 1D chaotic signals into 2D image tensors. It implements 7 distinct encoding variants to capture different temporal properties, ranging from Phase Space Histograms (`Val_ValChng`), to Outer Product Pairwise Correlations (`Values_x_Values`), and Windowed Self-Similarity Matrices (`WSI`). The module processes multi-channel sensor data independently and applies strict min-max normalization, ensuring the output perfectly matches the Vision Transformer's expected $H \times W \times C$ format.

* **train_fuzzy.py (The Master Pipeline)**: The primary execution script orchestrating the end-to-end training process. It dynamically reads from config.py to initialize the appropriate network state (Baseline, Phase 1, or Phase 2) and runs the PyTorch training loop. Crucially, it contains a "Thesis Proof" extraction block that isolates the learned fuzzy parameters ($\mu$ and $\sigma$) and saves the deepest attention weights (final_memberships.pt) for post-training interpretability and heatmap visualization. The *train.py* is a similar script but lacks the Fuzzy phase 1 and phase 2.


## 3. Getting Started

### Environment Setup
To ensure strict reproducibility, please use the provided dependencies list to clone the exact training environment.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YOUR_USERNAME/mtsc-ftcformer.git](https://github.com/YOUR_USERNAME/mtsc-ftcformer.git)
   cd mtsc-ftcformer

2. **Create and activate a virtual environment:**
    * Using standard Python venv: python -m venv env

    * Windows Activation: env\Scripts\activate

    * macOS/Linux Activation: source env/bin/activate

3. **Install exact dependencies:**
    pip install -r requirements.txt


### Dataset Configuration

Because the data factory is fully automated, you do **not** need to manually download or extract any raw time-series datasets. 

1. Simply run the data factory pipeline:
   ```bash
   python generate_data.py
   ```

2. What this script does automatically:

* It uses the aeon library to fetch the raw datasets specified in config.py from the internet and saves them to `data/raw/`.

* It mathematical converts the 1D signals into 2D image tensors using your chosen variants.

* It creates the `data/processed/` directory and saves the lossless .npy files.

* Finally, it compresses the entire processed directory into a .zip archive (processed.zip) for easy backup or transfer.


## 4. Usage: The Master Switchboard
The entire architecture is highly modular and controlled via the `configs/config.py` file. Instead of maintaining separate scripts for different model versions, you can isolate specific mathematical states for ablation studies by simply toggling the boolean flags:

* **Baseline Mode:** `USE_FUZZY = False`, `USE_CROSS_ATTENTION = False`
  *(Runs the standard, crisp FTCFormer ViT baseline without noise filtering).*
* **Phase 1/2 Mode:** `USE_FUZZY = True`
  *(Toggle `USE_SOFTMAX = True` for FANTF-style normalization, or `False` for Softmax-Free independent feature gating).*
* **Ultimate Hybrid:** `USE_FUZZY = True`, `USE_CROSS_ATTENTION = True`
  *(Activates the full Fuzzy cross-attention recovery mechanism to salvage tokens lost during downsampling).*

Once your desired configuration is set, execute the master training loop across all configured datasets and variants:
```bash
python train_fuzzy.py
```


## 5. Results & Interpretability

### Tracking Metrics
As the master loop executes, all training results—including the best-epoch test accuracies and total training times—are automatically appended to `results/final_results.csv`. This allows for seamless, tabular tracking of your ablation studies across different datasets and mathematical variants.

### Visualizing the Fuzzy Brain (Thesis Proof)
It is one thing to state that a network uses Fuzzy Logic; it is another to prove exactly *how* it filters out noise. 

At the end of a successful training run (when `USE_FUZZY = True`), the script automatically executes a dedicated parameter extraction block:
1. **Rule Extraction:** It isolates and prints the exact learned $\mu$ (target) and $\sigma$ (strictness) rules for every single attention head.
2. **Membership Interception:** It pushes a final test batch through the network and intercepts the deepest fuzzy membership tensor from the final TCBlock.
3. **Output Generation:** These raw weights are saved locally to `results/final_memberships.pt`. 

You can load this `.pt` tensor into any standard visualization script to generate attention heatmaps, definitively proving which specific chaotic noise frequencies the network learned to mathematically ignore.


## 6. Acknowledgements & References

This work is built upon the foundational **FTCFormer** architecture. While this repository introduces novel modifications for multivariate time-series data (including Adaptive Density Clustering, Learnable Gaussian Attention, Softmax-Free gating and structural ViT bypasses), the core Clustering Token Merging (CTM) concept and original baseline framework belong to the authors of FTCFormer.

If you use this codebase or build upon our fuzzy time-series innovations, please ensure you also cite the original FTCFormer paper:

* **Original FTCFormer Paper:** Muyi Bao, Changyu Zeng, Yifan Wang, Zhengni Yang, Zimu Wang, Guangliang Cheng, Jun Qi, and Wei Wang. *"FTCFormer: Fuzzy Token Clustering Transformer for Image Classification"*, 2025.
* **Original Repository:** [https://github.com/BaoBao0926/FTCFormer/tree/main](https://github.com/BaoBao0926/FTCFormer/tree/main)

