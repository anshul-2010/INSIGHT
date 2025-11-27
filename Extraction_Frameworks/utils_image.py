import cv2
import numpy as np
import torch

# Image <-> tensor utilities (assume 0..1 float images)
def read_image_bgr(path, to_rgb=True):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if to_rgb:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0
    return img  # H,W,3 float32 in [0,1]

def to_tensor(img):
    # img H,W,3 float32 [0,1] -> torch tensor 1,3,H,W
    t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    return t.float()

def to_numpy_image(tensor):
    # tensor 1,3,H,W or 3,H,W -> H,W,3 uint8
    if isinstance(tensor, torch.Tensor):
        t = tensor.detach().cpu()
        if t.dim() == 4:
            t = t[0]
        img = t.permute(1, 2, 0).numpy()
    else:
        img = np.array(tensor)
    img = np.clip(img * 255.0, 0, 255).astype("uint8")
    return img

def resize_numpy(img, size):
    # size: (H,W)
    return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR)