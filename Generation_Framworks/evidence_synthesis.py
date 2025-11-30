"""
Ω module (Eq. 21) that fuses DRCT cues, CLIP semantics, ReAct traces, and CoT narratives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


@dataclass
class EvidenceSummary:
    narrative: str
    clip_scores: Dict[str, float]
    reasoning_trace: List[str]
    actions: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]


class EvidenceSynthesizer:
    def assemble(
        self,
        prompts: Sequence[str],
        clip_scores,
        superpixels,
        cot_output: str,
        reasoning_steps: List[str],
        actions: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
    ) -> EvidenceSummary:
        score_map = {
            prompt: float(score) for prompt, score in zip(prompts, clip_scores.tolist())
        }
        return EvidenceSummary(
            narrative=cot_output,
            clip_scores=score_map,
            reasoning_trace=reasoning_steps,
            actions=actions,
            diagnostics=diagnostics,
        )
