import os
import time
import json
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from enhancement.model import DCE_Net
from enhancement.losses import ZeroDCELoss

class LowLightDataset(Dataset):
    def __init__(self, img_dir, img_size=(384, 384)):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        path = self.img_paths[idx]
        try:
            with Image.open(path) as img:
                img = img.convert('RGB')
                return self.transform(img)
        except Exception as e:
            # Fallback tensor if image corrupt
            return torch.zeros((3, 384, 384))

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train_enhancement(quick_test=False, resume=False, epochs=None):
    config = load_config()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Parameters
    cfg_enh = config["enhancement"]
    num_epochs = epochs if epochs is not None else (2 if quick_test else cfg_enh["epochs"])
    batch_size = 4 if quick_test else cfg_enh["batch_size"]
    lr = cfg_enh["lr"]
    save_dir = cfg_enh["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    img_size = tuple(config["dataset"]["image_size"])
    train_dir = os.path.join(config["dataset"]["processed_dir"], "train", "images")
    val_dir = os.path.join(config["dataset"]["processed_dir"], "val", "images")

    print(f"Loading datasets from {train_dir} and {val_dir}...")
    train_dataset = LowLightDataset(train_dir, img_size=img_size)
    val_dataset = LowLightDataset(val_dir, img_size=img_size)

    if quick_test:
        train_dataset.img_paths = train_dataset.img_paths[:100]
        val_dataset.img_paths = val_dataset.img_paths[:20]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = DCE_Net(
        n_iterations=cfg_enh.get("n_iterations", 8),
        scale_factor=cfg_enh.get("scale_factor", 1),
        base_channels=cfg_enh.get("channels", 32)
    ).to(device)

    # Loss weights
    weights = cfg_enh["loss_weights"]
    criterion = ZeroDCELoss(
        w_spa=weights["spatial"],
        w_exp=weights["exposure"],
        w_col=weights["color"],
        w_tv=weights["tv"],
        target_exp=cfg_enh.get("target_exposure", 0.6)
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)

    start_epoch = 0
    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": [], "loss_components": []}

    latest_ckpt_path = os.path.join(save_dir, "latest_model.pth")
    best_ckpt_path = os.path.join(save_dir, "best_model.pth")

    if resume and os.path.exists(latest_ckpt_path):
        print(f"Resuming enhancement training from {latest_ckpt_path}...")
        checkpoint = torch.load(latest_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint.get("best_val_loss", float('inf'))
        history = checkpoint.get("history", history)
        print(f"Resumed at epoch {start_epoch}")

    print("\n" + "=" * 60)
    print("STARTING ZERO-DCE++ ENHANCEMENT TRAINING")
    print("=" * 60)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        comp_sums = {"spatial_loss": 0.0, "exposure_loss": 0.0, "color_loss": 0.0, "tv_loss": 0.0}
        start_time = time.time()

        for batch_idx, img_low in enumerate(train_loader):
            img_low = img_low.to(device)

            optimizer.zero_grad()
            img_enhanced, A = model(img_low)
            loss, loss_comps = criterion(img_low, img_enhanced, A)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            for k in comp_sums:
                comp_sums[k] += loss_comps[k]

        epoch_train_loss = running_loss / max(1, len(train_loader))
        avg_comps = {k: v / max(1, len(train_loader)) for k, v in comp_sums.items()}

        # Validation loop
        model.eval()
        val_running_loss = 0.0
        with torch.no_grad():
            for img_val in val_loader:
                img_val = img_val.to(device)
                val_enhanced, A_val = model(img_val)
                v_loss, _ = criterion(img_val, val_enhanced, A_val)
                val_running_loss += v_loss.item()

        epoch_val_loss = val_running_loss / max(1, len(val_loader))
        elapsed = time.time() - start_time

        history["train_loss"].append(epoch_train_loss)
        history["val_loss"].append(epoch_val_loss)
        history["loss_components"].append(avg_comps)

        print(f"Epoch [{epoch+1}/{num_epochs}] ({elapsed:.1f}s) | Train Loss: {epoch_train_loss:.4f} | Val Loss: {epoch_val_loss:.4f}")
        print(f"  |- Spa: {avg_comps['spatial_loss']:.4f} | Exp: {avg_comps['exposure_loss']:.4f} | Col: {avg_comps['color_loss']:.4f} | TV: {avg_comps['tv_loss']:.4f}")

        # Save latest checkpoint
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "history": history
        }
        torch.save(checkpoint_data, latest_ckpt_path)

        # Save best checkpoint
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(checkpoint_data, best_ckpt_path)
            print(f"  * New best model saved to {best_ckpt_path}")

    # Plot loss history
    os.makedirs("results/figures", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(history["train_loss"])+1), history["train_loss"], label="Train Loss", color="blue")
    plt.plot(range(1, len(history["val_loss"])+1), history["val_loss"], label="Val Loss", color="red")
    plt.title("Zero-DCE++ Enhancement Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/figures/enhancement_loss_curve.png")
    plt.close()

    print("\nTraining completed! Saved final curves to results/figures/enhancement_loss_curve.png")
    return best_ckpt_path

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick-test", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()
    train_enhancement(quick_test=args.quick_test, resume=args.resume, epochs=args.epochs)
