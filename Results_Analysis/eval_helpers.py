"""
Evaluation helpers for Results_Analysis: data loading, metric computation, and scorer wrappers.

These helpers wire together the Extraction, Classifier and Generation frameworks and provide a
clean interface for the scripts under Results_Analysis.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable, List, Tuple
import numpy as np
from sklearn.metrics import roc_auc_score
from torchvision import transforms
from PIL import Image
import torch

from Extraction_Frameworks.stage1_pipeline import run_stage1
from Generation_Framworks.clip_scoring import ForensicCLIPScorer
from Classifier_Frameworks.models.resnet_like import ResNetSmall
from Classifier_Frameworks.models.vit_tiny import TinyViT
from Classifier_Frameworks.models.hybrid_backbone import HybridBackbone
from Classifier_Frameworks.insight_dataset import InsightDataset


def load_datasets(root_dir: str, names: Iterable[str] = ("dfdc", "sra")) -> Dict[str, List[Tuple[str, int]]]:
    """Load image paths and labels from dataset directories under `root_dir`.

    Expected folder structure:
    root_dir/<dataset_name>/real/*.png
    root_dir/<dataset_name>/fake/*.png
    Returns: dict name -> list of (image_path, label) where label is 0 for real, 1 for fake
    """
    datasets: Dict[str, List[Tuple[str, int]]] = {}
    for name in names:
        ds_dir = os.path.join(root_dir, name)
        if not os.path.isdir(ds_dir):
            continue
        entries = []
        for label_dir, label in (("real", 0), ("fake", 1)):
            p = os.path.join(ds_dir, label_dir)
            if not os.path.isdir(p):
                continue
            for fn in os.listdir(p):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    entries.append((os.path.join(p, fn), label))
        datasets[name] = entries
    return datasets


class ForensicEvaluator:
    """Wraps models and scoring logic used by the Results_Analysis scripts.

    Provides functions to compute scalar scores for images using different baselines.
    """

    def __init__(self, device: str = "cpu", prompts: Iterable[str] = ("tampered face",), use_clip: bool = True):
        self.device = device
        self.resnet = ResNetSmall().to(device).eval()
        self.vit = TinyViT(img_size=32).to(device).eval()
        self.hybrid = HybridBackbone().to(device).eval()
        self.use_clip = use_clip
        if use_clip:
            try:
                self.clip = ForensicCLIPScorer(device=device)
            except Exception:
                print("Warning: CLIP not available. Falling back to text heuristics for scoring.")
                self.clip = None
        else:
            self.clip = None
        self.prompts = list(prompts)
        self.prompt_emb = self.clip.encode_prompts(self.prompts) if self.clip is not None else None
        self.transform = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

    def _pil_to_tensor(self, pil: Image.Image) -> torch.Tensor:
        t = self.transform(pil).unsqueeze(0).to(self.device)
        return t

    def gradcam_score(self, sr_img: np.ndarray) -> float:
        # sr_img is HxWxC float in [0,1]; compute stage1 gradcam and use mean as score
        outputs = run_stage1(sr_img if isinstance(sr_img, str) else _tmp_img_to_path(sr_img), device=self.device)
        # If run_stage1 accepted path, then it will have gradcam in outputs
        return float(np.mean(outputs.gradcam))

    def clip_global_score(self, sr_img: np.ndarray) -> float:
        # Compute CLIP score by comparing full image embedding to prompts. Feed a numpy array into the checker.
        if isinstance(sr_img, str):
            # read image from disk
            pil = Image.open(sr_img).convert('RGB')
            np_img = np.asarray(pil).astype(np.float32) / 255.0
        else:
            np_img = sr_img.astype(np.float32) / 255.0
        if self.clip is None:
            return 0.0
        emb = self.clip._encode_patch(np_img)
        # ForensicCLIPScorer.encode_prompts returns normalized vectors
        cos_sim = np.array(torch.nn.functional.cosine_similarity(emb.unsqueeze(0), self.prompt_emb, dim=1).cpu())
        # return the max similarity value
        return float(np.max(cos_sim))

    def cnn_score(self, sr_img: np.ndarray) -> float:
        # Run the ResNet backbone; produce probability of class 1
        pil = Image.fromarray((sr_img * 255).astype('uint8')) if isinstance(sr_img, np.ndarray) else Image.open(sr_img).convert('RGB')
        t = self._pil_to_tensor(pil)
        with torch.no_grad():
            logits, _ = self.resnet(t)
            p = torch.softmax(logits, dim=1)[0, 1]
        return float(p)

    def vit_score(self, sr_img: np.ndarray) -> float:
        pil = Image.fromarray((sr_img * 255).astype('uint8')) if isinstance(sr_img, np.ndarray) else Image.open(sr_img).convert('RGB')
        t = self._pil_to_tensor(pil)
        with torch.no_grad():
            logits, _ = self.vit(t)
            p = torch.softmax(logits, dim=1)[0, 1]
        return float(p)

    def insight_score(self, sr_img: np.ndarray) -> float:
        # compute stage1 patches and use CLIP patch scoring to return max patch score
        outputs = run_stage1(sr_img if isinstance(sr_img, str) else _tmp_img_to_path(sr_img), device=self.device)
        patches = outputs.patches
        sims = []
        for p in patches:
            if self.clip is None:
                sims.append(0.0)
                continue
            z = self.clip._encode_patch(p)
            sim = torch.nn.functional.cosine_similarity(z.unsqueeze(0), self.prompt_emb, dim=1)
            sims.append(sim.max().item())
        if len(sims) == 0:
            return 0.0
        return float(max(sims))


def evaluate_auroc(scores: Iterable[float], labels: Iterable[int]) -> float:
    try:
        return float(roc_auc_score(list(labels), list(scores)))
    except Exception:
        return float("nan")


def table_print(dct: dict, title: str = None):
    from tabulate import tabulate
    rows = []
    for k, v in dct.items():
        if isinstance(v, dict):
            rows.append([k] + [f"{v2:.4f}" if isinstance(v2, (float, int)) else str(v2) for v2 in v.values()])
        else:
            rows.append([k, v])
    if title:
        print("==", title, "==")
    print(tabulate(rows, headers=["method"] + (list(list(dct.values())[0].keys()) if isinstance(list(dct.values())[0], dict) else ["value"])))


def _tmp_img_to_path(img: np.ndarray) -> str:
    """Write numpy RGB image in [0,1] to a temporary PNG and return the path for stage1 pipeline.
    This helper ensures stage1 can accept numpy images by passing a path.
    """
    import tempfile
    import cv2

    td = tempfile.gettempdir()
    p = os.path.join(td, f"insight_stage1_tmp_{np.random.randint(1e9)}.png")
    cv2.imwrite(p, cv2.cvtColor((img * 255).astype("uint8"), cv2.COLOR_RGB2BGR))
    return p
