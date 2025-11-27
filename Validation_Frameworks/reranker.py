"""
Filtering and ranking logic for Mi tuples (Eq. 26–27).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from multimodal_judge import JudgeResult
from rubric_eval import RubricScore


@dataclass
class ArtifactMetadata:
    artifact_index: int
    explanation: str
    rubric: RubricScore
    quality_score: float
    judge: JudgeResult
    paraphrases: Dict[str, str]


def build_metadata(
    explanations: Sequence[str],
    rubric_scores: Sequence[tuple[RubricScore, float]],
    judge_results: Sequence[JudgeResult],
    paraphrases: Dict[int, Dict[str, str]],
) -> List[ArtifactMetadata]:
    metas: List[ArtifactMetadata] = []
    for i, text in enumerate(explanations):
        rubric, G = rubric_scores[i]
        metas.append(
            ArtifactMetadata(
                artifact_index=i,
                explanation=text,
                rubric=rubric,
                quality_score=float(G),
                judge=judge_results[i],
                paraphrases=paraphrases.get(i, {}),
            )
        )
    return metas


def filter_and_rerank(
    metas: Sequence[ArtifactMetadata],
    tau_confidence: float = 0.5,
    top_k: int = 5,
) -> List[ArtifactMetadata]:
    filtered = [
        m
        for m in metas
        if m.judge.verdict and m.judge.confidence >= tau_confidence
    ]
    ranked = sorted(
        filtered,
        key=lambda m: (-m.quality_score, -m.judge.confidence),
    )
    return ranked[:top_k]
