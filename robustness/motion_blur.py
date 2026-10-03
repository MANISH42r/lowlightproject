import cv2
import numpy as np

def apply_motion_blur(img_np, kernel_size=9):
    """
    Applies linear motion blur to image array (uint8 [0, 255]).
    """
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[int((kernel_size-1)/2), :] = np.ones(kernel_size)
    kernel = kernel / kernel_size
    return cv2.filter2D(img_np, -1, kernel)
