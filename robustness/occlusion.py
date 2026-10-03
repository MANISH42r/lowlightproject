import numpy as np

def apply_partial_occlusion(img_np, occlusion_ratio=0.2):
    """
    Applies random rectangular black patch occlusion.
    """
    h, w, c = img_np.shape
    occ_h = int(h * np.sqrt(occlusion_ratio))
    occ_w = int(w * np.sqrt(occlusion_ratio))

    top = np.random.randint(0, h - occ_h + 1)
    left = np.random.randint(0, w - occ_w + 1)

    occ_img = img_np.copy()
    occ_img[top:top+occ_h, left:left+occ_w, :] = 0
    return occ_img
