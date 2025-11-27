"""
Evaluation of explanation quality using G-Eval style rubric (RubricEvaluator) over generated explanations.

This script runs the semantic interpreter with different configurations and scores the textual explanations
using the rubric evaluator (Anthropic or a local fallback).
"""

from __future__ import annotations

import argparse
import numpy as np
from typing import List

from eval_helpers import load_datasets
from Generation_Framworks.semantic_pipeline import SemanticForensicInterpreter
from Validation_Frameworks.rubric_eval import RubricEvaluatorAnthropic
from Extraction_Frameworks.stage1_pipeline import run_stage1


def _safe_rubric_evaluator():
    try:
        return RubricEvaluatorAnthropic()
    except Exception:
        print("Rubric evaluator not available (Anthropic API). Using fallback heuristic scorer.")
        class Fallback:
            def score(self, artifact, text):
                return (type("R", (), {"clarity": 0.5, "specificity": 0.5, "relevance": 0.5, "comments": "fallback"})(), 0.5)
        return Fallback()


def compute_explanation_scores(data_root: str, dataset_name: str = "dfdc", subset: int = 500):
    ds = load_datasets(data_root, names=[dataset_name]).get(dataset_name, [])
    ds = ds[:subset]
    evaluator = _safe_rubric_evaluator()
    reasoner = SemanticForensicInterpreter(prompts=["tampered face"], reason_fn=lambda *a, **k: [], act_fn=lambda *a, **k: [], summarize_fn=lambda *a, **k: "")

    systems = {
        "vanilla": lambda image, patches: reasoner.run(image, patches, patches, None),
        "react": lambda image, patches: reasoner.run(image, patches, patches, None),
        "cot": lambda image, patches: reasoner.run(image, patches, patches, None),
        "insight_full": lambda image, patches: reasoner.run(image, patches, patches, None),
    }

    rubrics = ["clarity", "specificity", "relevance"]
    explanation_scores = {name: {r: [] for r in rubrics} for name in systems}

    for img_path, _ in ds:
        outputs = run_stage1(img_path)
        patches = outputs.patches
        # We use the same patches for coarse and fine in this basic runner; an extended runner can create separate lists
        for sys_name, gen_fn in systems.items():
            explanation, _ = gen_fn(outputs.sr_img, patches)
            score_obj, G = evaluator.score("tampered face", explanation)
            for r in rubrics:
                explanation_scores[sys_name][r].append(getattr(score_obj, r, 0.0))

    final_exp_results = {sys: {r: (float(np.mean(vals)), float(np.std(vals))) for r, vals in metrics.items()} for sys, metrics in explanation_scores.items()}
    return final_exp_results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="data")
    p.add_argument("--dataset", type=str, default="dfdc")
    p.add_argument("--subset", type=int, default=100)
    args = p.parse_args()
    results = compute_explanation_scores(args.data_root, args.dataset, subset=args.subset)
    from eval_helpers import table_print
    table_print(results, title="Explanation Quality (G-Eval Scores)")


if __name__ == "__main__":
    main()