"""
Structured forensic report assembly.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import List

from reranker import ArtifactMetadata


@dataclass
class ReportEntry:
    artifact_index: int
    text: str
    quality_score_G: float
    judge_confidence: float
    judge_justification: str


def assemble_report(
    metas: List[ArtifactMetadata],
    target_style: str = "technical",
    title: str = "INSIGHT Forensic Report",
):
    report = {
        "title": title,
        "summary": f"Top {len(metas)} artifact hypotheses selected.",
        "artifacts": [],
    }
    for meta in metas:
        parap = meta.paraphrases.get(target_style)
        text = parap if parap else meta.explanation
        entry = ReportEntry(
            artifact_index=meta.artifact_index,
            text=text,
            quality_score_G=meta.quality_score,
            judge_confidence=meta.judge.confidence,
            judge_justification=meta.judge.justification,
        )
        report["artifacts"].append(asdict(entry))
    return report


def save_report_json(report, out_path: str):
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
