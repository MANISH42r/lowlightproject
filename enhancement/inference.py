import os
import yaml
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from enhancement.model import DCE_Net

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

class ZeroDCEEnhancer:
    def __init__(self, model_path=None, config_path="config.yaml", device=None):
        self.config = load_config(config_path)
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        cfg_enh = self.config["enhancement"]
        self.model = DCE_Net(
            n_iterations=cfg_enh.get("n_iterations", 8),
            scale_factor=cfg_enh.get("scale_factor", 1),
            base_channels=cfg_enh.get("channels", 32)
        ).to(self.device)

        if model_path is None:
            model_path = os.path.join(cfg_enh["save_dir"], "best_model.pth")
            if not os.path.exists(model_path):
                model_path = os.path.join(cfg_enh["save_dir"], "latest_model.pth")

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
            self.model.load_state_dict(state_dict)
            print(f"Loaded Zero-DCE++ weights from {model_path}")
        else:
            print(f"Warning: Model weights not found at {model_path}. Initialized randomly.")

        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def enhance_pil(self, pil_img):
        w, h = pil_img.size
        # Resize to multiple of 4 if needed
        w_mod = (w // 4) * 4
        h_mod = (h // 4) * 4
        if (w_mod, h_mod) != (w, h):
            pil_img = pil_img.resize((w_mod, h_mod), Image.Resampling.BILINEAR)

        img_tensor = self.transform(pil_img.convert('RGB')).unsqueeze(0).to(self.device)

        with torch.no_grad():
            enhanced_tensor, _ = self.model(img_tensor)

        enhanced_np = (enhanced_tensor.squeeze(0).cpu().clamp(0, 1).numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
        enhanced_pil = Image.fromarray(enhanced_np)
        return enhanced_pil

    def enhance_tensor(self, tensor_img):
        """
        tensor_img: [B, 3, H, W] tensor in range [0, 1]
        """
        tensor_img = tensor_img.to(self.device)
        with torch.no_grad():
            enhanced, _ = self.model(tensor_img)
        return enhanced
