"""
Semantic interpretation layer (Section 5) that fuses CLIP scores, ReAct reasoning, and CoT narratives.
"""

from __future__ import annotations

from typing import Optional, Sequence

from clip_scoring import ForensicCLIPScorer
from cot_decoder import ChainOfThoughtDecoder
from evidence_synthesis import EvidenceSynthesizer
from react_policy import ReActForensics
from uncertainty_trigger import CLIPUncertaintyModule


class SemanticForensicInterpreter:
    def __init__(
        self,
        prompts: Sequence[str],
        reason_fn,
        act_fn,
        summarize_fn,
        device: str = "cuda",
        alpha: float = 0.5,
    ):
        self.prompts = list(prompts)
        self.alpha = alpha
        self.scorer = ForensicCLIPScorer(device=device)
        self.prompt_emb = self.scorer.encode_prompts(self.prompts)
        self.uncertainty = CLIPUncertaintyModule()
        self.react = ReActForensics(reason_fn, act_fn)
        self.cot = ChainOfThoughtDecoder(summarize_fn)
        self.synth = EvidenceSynthesizer()

    def run(
        self,
        image,
        patches_coarse,
        patches_fine,
        superpixels,
        artifact_prompts: Optional[Sequence[str]] = None,
    ):
        prompts = list(artifact_prompts or self.prompts)
        score_bundle = self.scorer.compute_dual_granularity_scores(
            patches_coarse, patches_fine, self.prompt_emb, alpha=self.alpha
        )
        decision = self.uncertainty.evaluate(score_bundle.unified, score_bundle.region_scores)

        reasoning_trace, actions = [], []
        if decision.need_reasoning:
            reasoning_trace, actions = self.react.run(
                image=image,
                superpixels=superpixels,
                artifact_prompts=prompts,
                region_features=score_bundle.region_scores,
            )

        cot_out = self.cot.build(reasoning_trace, superpixels, prompts)

        evidence = self.synth.assemble(
            prompts=prompts,
            clip_scores=score_bundle.unified.detach().cpu(),
            superpixels=superpixels,
            cot_output=cot_out,
            reasoning_steps=reasoning_trace,
            actions=actions,
            diagnostics={
                "clip_max": decision.max_score,
                "clip_variance": decision.variance,
                "reason": decision.reason,
                "alpha": self.alpha,
            },
        )
        return evidence, score_bundle
