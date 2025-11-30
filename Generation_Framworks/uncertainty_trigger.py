"""
Uncertainty heuristics deciding when CLIP evidence is insufficient (Section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch


@dataclass
class UncertaintyDecision:
    need_reasoning: bool
    max_score: float
    variance: float
    reason: str


class CLIPUncertaintyModule:
    def __init__(
        self, threshold_clip: float = 0.24, inconsistency_threshold: float = 0.18
    ):
        self.tau = threshold_clip
        self.inc = inconsistency_threshold

    def evaluate(
        self, semantic_scores: torch.Tensor, region_scores: List[torch.Tensor]
    ) -> UncertaintyDecision:
        max_score = semantic_scores.max().item()
        variance = 0.0
        if len(region_scores) > 1:
            mat = torch.stack(region_scores)
            variance = mat.var(dim=0).mean().item()
        low_confidence = max_score < self.tau
        inconsistent = variance > self.inc
        need_reasoning = low_confidence or inconsistent
        if low_confidence and inconsistent:
            reason = "scores ambiguous and region-level votes disagree"
        elif low_confidence:
            reason = "semantic scores fall below threshold"
        elif inconsistent:
            reason = "region-level similarities conflict"
        else:
            reason = "clip evidence sufficient"
        return UncertaintyDecision(
            need_reasoning=need_reasoning,
            max_score=max_score,
            variance=variance,
            reason=reason,
        )
