# src/image_converter.py
import numpy as np
import cv2
from scipy.spatial.distance import cdist

def get_gradients(time_series):
    """Helper: Calculate velocity (1st diff) and acceleration (2nd diff)"""
    # Pad to keep shape consistent
    velocity = np.gradient(time_series)
    acceleration = np.gradient(velocity)
    return velocity, acceleration

def normalize_channel(channel):
    """Helper: Min-Max normalize to [0, 1]"""
    _min, _max = channel.min(), channel.max()
    if _max - _min > 1e-6:
        return (channel - _min) / (_max - _min)
    return np.zeros_like(channel)

# --- The 7 Variants ---

def variant_1_val_valchng(channel, size):
    """Variant 1: 2D Histogram of Value vs Velocity"""
    velocity, _ = get_gradients(channel)
    # Create 2D Histogram
    heatmap, _, _ = np.histogram2d(channel, velocity, bins=size, density=True)
    return heatmap.T  # Transpose to align axes

def variant_2_valchng_chngvalchng(channel, size):
    """Variant 2: 2D Histogram of Velocity vs Acceleration"""
    velocity, acceleration = get_gradients(channel)
    heatmap, _, _ = np.histogram2d(velocity, acceleration, bins=size, density=True)
    return heatmap.T

def variant_3_values_x_values(channel, size):
    """Variant 3: Outer Product Matrix (Pairwise Correlation)"""
    # Outer product: matrix[i, j] = val[i] * val[j]
    matrix = np.outer(channel, channel)
    # Resize to target image size if T != 224
    if matrix.shape != (size, size):
        matrix = cv2.resize(matrix, (size, size), interpolation=cv2.INTER_LINEAR)
    return matrix

def variant_4_replval(channel, size):
    """Variant 4: Replicate Values Vertically"""
    # 1. Resize 1D signal to target width
    signal_resized = cv2.resize(channel.reshape(1, -1), (size, 1), interpolation=cv2.INTER_LINEAR)
    # 2. Replicate vertically
    return np.tile(signal_resized, (size, 1))

def variant_5_replvalchng(channel, size):
    """Variant 5: Replicate 1st Differences Vertically"""
    velocity, _ = get_gradients(channel)
    # Normalize the velocity before creating image
    velocity = normalize_channel(velocity)
    
    signal_resized = cv2.resize(velocity.reshape(1, -1), (size, 1), interpolation=cv2.INTER_LINEAR)
    return np.tile(signal_resized, (size, 1))

def variant_6_tssi(channel, size):
    """Variant 6: Time Series as Screenshot Image (Binary Plot)"""
    # Create blank canvas
    img = np.zeros((size, size), dtype=np.float32)
    
    # Normalize channel to 0 -> (size-1) integers
    norm_vals = normalize_channel(channel)
    y_indices = (norm_vals * (size - 1)).astype(int)
    
    # Invert Y because images start from top-left, plots from bottom-left
    y_indices = (size - 1) - y_indices
    
    # Map time steps to X axis
    x_indices = np.linspace(0, size - 1, len(channel)).astype(int)
    
    # "Draw" the points
    img[y_indices, x_indices] = 1.0
    return img

def variant_7_wsi(channel, size):
    """Variant 7: Windowed Self-Similarity Image (Cosine Similarity)"""
    # Use a sliding window of 10% of length
    T = len(channel)
    window_size = max(5, int(T * 0.1))
    
    # Create windows
    # Shape: (Num_Windows, Window_Size)
    windows = np.lib.stride_tricks.sliding_window_view(channel, window_size)
    
    # Compute Cosine Distance (1 - Cosine Similarity)
    # cdist computes distance between every pair of windows
    dist_matrix = cdist(windows, windows, metric='cosine')
    
    # Convert distance (0=same, 2=opposite) to similarity (1=same, 0=opposite)
    sim_matrix = 1 - (dist_matrix / 2.0)
    
    # Resize to target size
    if sim_matrix.shape != (size, size):
        sim_matrix = cv2.resize(sim_matrix, (size, size), interpolation=cv2.INTER_LINEAR)
        
    return sim_matrix

# --- Master Converter Function ---

def convert_to_image(time_series, method='ReplValChng', target_size=224):
    """
    Main entry point.
    Input: time_series shape (Channels, TimeSteps) e.g., (6, 100)
    Output: Image shape (H, W, Channels) e.g., (224, 224, 6)
    """
    C, T = time_series.shape
    output_img = np.zeros((target_size, target_size, C), dtype=np.float32)
    
    method_map = {
        'Val_ValChng': variant_1_val_valchng,
        'ValChng_ChngValChng': variant_2_valchng_chngvalchng,
        'Values_x_Values': variant_3_values_x_values,
        'ReplVal': variant_4_replval,
        'ReplValChng': variant_5_replvalchng,
        'TSSI': variant_6_tssi,
        'WSI': variant_7_wsi
    }
    
    func = method_map.get(method)
    if not func:
        raise ValueError(f"Unknown variant: {method}")
        
    for c in range(C):
        channel_data = time_series[c]
        # Normalize strictly for Variants 1, 2, 3, 7 to standard range
        # (Variants 4, 5, 6 handle normalization internally)
        if method in ['Val_ValChng', 'ValChng_ChngValChng', 'Values_x_Values', 'WSI']:
            channel_data = normalize_channel(channel_data)
            
        img_channel = func(channel_data, target_size)
        
        # Final safety normalization to [0, 1]
        output_img[:, :, c] = normalize_channel(img_channel)
        
    return output_img