import numpy as np

def apply_gaussian_noise(img_np, std=0.05):
    """
    Adds zero-mean Gaussian noise to image array (uint8 [0, 255]).
    """
    noise = np.random.normal(0, std * 255.0, img_np.shape)
    noisy_img = img_np.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)
