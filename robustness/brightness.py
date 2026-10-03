import numpy as np

def apply_brightness_reduction(img_np, factor=0.5):
    """
    Reduces brightness of image array (uint8 [0, 255]).
    """
    img_float = img_np.astype(np.float32) * factor
    return np.clip(img_float, 0, 255).astype(np.uint8)
