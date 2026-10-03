import cv2

def apply_gaussian_blur(img_np, kernel_size=5):
    """
    Applies Gaussian blur to image array (uint8 [0, 255]).
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(img_np, (kernel_size, kernel_size), 0)
