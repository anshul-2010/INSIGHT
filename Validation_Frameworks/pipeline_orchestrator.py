"""
Stage-6 orchestrator: rubric → multimodal judge → paraphraser → re-ranker → report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

from multimodal_judge import JudgeBackend, JudgeResult
from paraphraser import Paraphraser
from report_generation import assemble_report
from reranker import ArtifactMetadata, build_metadata, filter_and_rerank
from rubric_eval import RubricEvaluatorAnthropic, RubricScore


@dataclass
class StageOutputs:
    rubric_scores: List[tuple[RubricScore, float]]
    judge_results: List[JudgeResult]
    paraphrases: Dict[int, Dict[str, str]]
    selected: List[ArtifactMetadata]
    report: Dict


class PipelineOrchestrator:
    def __init__(
        self,
        anthropic_api_key: str | None = None,
        judge_backend: JudgeBackend | None = None,
        paraphraser_model: str = "google/flan-t5-large",
        paraphraser_device: str | None = None,
    ):
        self.rubric = RubricEvaluatorAnthropic(api_key=anthropic_api_key)
        self.judge_backend = judge_backend
        self.paraphraser = Paraphraser(
            model_name=paraphraser_model, device=paraphraser_device
        )

    def set_judge_backend(self, judge_backend: JudgeBackend):
        self.judge_backend = judge_backend

    def run(
        self,
        image_np,
        artifacts: Sequence[str],
        explanations: Sequence[str],
        styles: Sequence[str] = ("technical", "summary"),
        tau_conf: float = 0.5,
        top_k: int = 5,
    ) -> StageOutputs:
        if self.judge_backend is None:
            raise RuntimeError(
                "Judge backend not configured. Call set_judge_backend()."
            )
        if len(artifacts) != len(explanations):
            raise ValueError("Artifacts and explanations must align.")

        rubric_scores = [
            self.rubric.score(artifacts[i], explanations[i])
            for i in range(len(artifacts))
        ]

        judge_results: List[JudgeResult] = []
        for artifact, explanation in zip(artifacts, explanations):
            jr = self.judge_backend.judge(image_np, artifact, explanation)
            if not isinstance(jr, JudgeResult):
                verdict, conf, just = jr
                jr = JudgeResult(bool(verdict), float(conf), str(just))
            judge_results.append(jr)

        paraphrases: Dict[int, Dict[str, str]] = {}
        for idx, text in enumerate(explanations):
            paraphrases[idx] = {}
            for style in styles:
                try:
                    paraphrases[idx][style] = self.paraphraser.transform(
                        text, style=style
                    )
                except Exception:
                    paraphrases[idx][style] = text

        metas = build_metadata(explanations, rubric_scores, judge_results, paraphrases)
        selected = filter_and_rerank(metas, tau_confidence=tau_conf, top_k=top_k)
        report = assemble_report(selected, target_style=styles[0])
        return StageOutputs(
            rubric_scores=rubric_scores,
            judge_results=judge_results,
            paraphrases=paraphrases,
            selected=selected,
            report=report,
        )
