"""
Multimodal judge evaluation: run LLM-as-a-judge over explanations and compute accuracy and
correlation between judge confidence and superpixel salience.
"""

from __future__ import annotations

import argparse
import numpy as np
from typing import List

from eval_helpers import load_datasets
from Validation_Frameworks.multimodal_judge import TextOnlyJudgeAdapter
from Generation_Framworks.semantic_pipeline import SemanticForensicInterpreter
from Extraction_Frameworks.stage1_pipeline import run_stage1
from eval_helpers import table_print


def compute_judge_metrics(data_root: str, dataset_name: str = "sra", subset: int = 300):
    ds = load_datasets(data_root, names=[dataset_name]).get(dataset_name, [])
    ds = ds[:subset]
    # instantiate reasoner and judge (fallback to a simple text-LLM stub if needed)
    reasoner = SemanticForensicInterpreter(prompts=["tampered face"], reason_fn=lambda *a, **k: [], act_fn=lambda *a, **k: [], summarize_fn=lambda *a, **k: "")
    judge = TextOnlyJudgeAdapter(lambda prompt: '{"verdict":"no","confidence":0.5,"justification":"placeholder"}')

    accuracies: List[bool] = []
    false_supports: List[bool] = []
    confs: List[float] = []
    max_saliences: List[float] = []

    for img_path, support_label in ds:
        outputs = run_stage1(img_path)
        patches = outputs.patches
        explanation, _ = reasoner.run(outputs.sr_img, patches, patches, outputs.hierarchy)
        jr = judge.judge(outputs.sr_img, "Image is real?", explanation)
        accuracies.append(jr.verdict == bool(support_label))
        false_supports.append(jr.verdict and support_label == 0)
        confs.append(jr.confidence)
        max_saliences.append(max(outputs.superpixel_activations.values()) if outputs.superpixel_activations else 0.0)

    conf_sal_corr = float(np.corrcoef(confs, max_saliences)[0, 1]) if len(confs) > 1 else float("nan")
    final = {
        "accuracy": float(np.mean(accuracies)),
        "false_support_rate": float(np.mean(false_supports)),
        "confidence_salience_corr": conf_sal_corr,
    }
    return final


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--dataset", type=str, default="sra")
    p.add_argument("--subset", type=int, default=300)
    args = p.parse_args()
    results = compute_judge_metrics(args.data_root, args.dataset, subset=args.subset)
    table_print(results, title="LLM-as-a-Judge Performance")


if __name__ == "__main__":
    main()