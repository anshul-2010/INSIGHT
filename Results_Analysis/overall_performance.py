"""
Binary detection performance evaluation using pre-built components.

This script evaluates several baselines:
 - GradCAM max activation (Stage-1 GradCAM output)
 - CLIP Global (image-level semantic score)
 - CNN forensic probability (ResNet classifier)
 - ViT forensic probability (TinyViT classifier)
 - INSIGHT (max CLIP patch score across attention-weighted patches)

Usage:
    python overall_performance.py --data_root data --datasets dfdc sra
"""

from __future__ import annotations

import argparse
from eval_helpers import load_datasets, ForensicEvaluator, evaluate_auroc, table_print


def compute_results(data_root: str, datasets: list[str], device: str = "cpu", use_clip: bool = True):
    datasets_map = load_datasets(data_root, names=datasets)
    evaluator = ForensicEvaluator(device=device, use_clip=use_clip)
    results_detection = {}
    for name, ds in datasets_map.items():
        aurocs = []
        for img_path, label in ds:
            outputs = evaluator  # alias
            # use stage1 pipeline and scoring
            gradcam_score = evaluator.gradcam_score(img_path)
            clip_global_score = evaluator.clip_global_score(img_path)
            cnn_score = evaluator.cnn_score(img_path)
            vit_score = evaluator.vit_score(img_path)
            insight_score = evaluator.insight_score(img_path)
            aurocs.append([
                gradcam_score, clip_global_score, cnn_score, vit_score, insight_score, label
            ])
        results_detection[name] = {
            "gradcam": evaluate_auroc([x[0] for x in aurocs], [x[5] for x in aurocs]),
            "clip_global": evaluate_auroc([x[1] for x in aurocs], [x[5] for x in aurocs]),
            "cnn": evaluate_auroc([x[2] for x in aurocs], [x[5] for x in aurocs]),
            "vit": evaluate_auroc([x[3] for x in aurocs], [x[5] for x in aurocs]),
            "insight": evaluate_auroc([x[4] for x in aurocs], [x[5] for x in aurocs]),
        }
    return results_detection


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--datasets", nargs="+", default=["dfdc"])
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--no_clip", action='store_true', help="Disable CLIP scoring (useful for machines without clip installed)")
    args = p.parse_args()
    results = compute_results(args.data_root, args.datasets, device=args.device, use_clip=not args.no_clip)
    table_print(results, title="Binary Real-vs-Synthetic AUROC Table")


if __name__ == "__main__":
    main()
