import os
import numpy as np
import torch
import cv2
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_func
from skimage.metrics import structural_similarity as ssim_func

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

class ImageQualityEvaluator:
    def __init__(self, device=None, load_lpips=False):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.lpips_fn = None
        if load_lpips and LPIPS_AVAILABLE:
            try:
                self.lpips_fn = lpips.LPIPS(net='alex', verbose=False).to(self.device)
            except Exception as e:
                self.lpips_fn = None

    def calculate_no_reference_metrics(self, img_np):
        """
        Calculates non-reference quality metrics for low-light / enhanced image:
        - Mean Brightness (0 - 255)
        - Contrast (Standard deviation of pixel intensities)
        - Entropy (Information quantity)
        - Spatial Frequency (Detail sharpness measure)
        """
        if img_np.dtype != np.uint8:
            img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)

        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY) if img_np.ndim == 3 else img_np
        
        # Mean brightness
        mean_brightness = float(np.mean(gray))

        # Contrast (std dev)
        contrast = float(np.std(gray))

        # Image entropy
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        hist_norm = hist_norm[hist_norm > 0]
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm)))

        # Spatial Frequency (SF)
        rf = np.diff(gray, axis=0)
        cf = np.diff(gray, axis=1)
        spatial_frequency = float(np.sqrt(np.mean(rf**2) + np.mean(cf**2)))

        return {
            "mean_brightness": round(mean_brightness, 2),
            "contrast": round(contrast, 2),
            "entropy": round(entropy, 4),
            "spatial_frequency": round(spatial_frequency, 2)
        }

    def calculate_reference_metrics(self, ref_np, target_np):
        """
        Calculates full-reference quality metrics between reference (ground truth) and target.
        Only valid if a true ground-truth image exists.
        """
        if ref_np.shape != target_np.shape:
            raise ValueError("Reference and target image dimensions must match.")

        # PSNR
        psnr_val = float(psnr_func(ref_np, target_np, data_range=255))

        # SSIM
        win_size = min(7, min(ref_np.shape[0], ref_np.shape[1]))
        if win_size % 2 == 0:
            win_size -= 1
        ssim_val = float(ssim_func(ref_np, target_np, win_size=win_size, channel_axis=2 if ref_np.ndim == 3 else None, data_range=255))

        res = {
            "psnr": round(psnr_val, 2),
            "ssim": round(ssim_val, 4)
        }

        if self.lpips_fn is not None:
            try:
                t_ref = torch.from_numpy(ref_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                t_tar = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).float() / 127.5 - 1.0
                t_ref, t_tar = t_ref.to(self.device), t_tar.to(self.device)
                with torch.no_grad():
                    lpips_val = float(self.lpips_fn(t_ref, t_tar).item())
                res["lpips"] = round(lpips_val, 4)
            except Exception:
                pass

        return res
