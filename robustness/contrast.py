import numpy as np

def apply_contrast_reduction(img_np, factor=0.5):
    """
    Reduces contrast around mean intensity.
    """
    img_float = img_np.astype(np.float32)
    mean_val = np.mean(img_float)
    reduced = mean_val + factor * (img_float - mean_val)
    return np.clip(reduced, 0, 255).astype(np.uint8)
