"""
Chain-of-Thought decoder Ψ consolidating the ReAct reasoning history (Section 5.3).
"""

from __future__ import annotations

from typing import Callable, List, Sequence


class ChainOfThoughtDecoder:
    def __init__(
        self,
        vlm_summarize_fn: Callable[..., str],
        fallback: str = "Artifacts identifiable without deeper reasoning.",
    ):
        self.summarize_fn = vlm_summarize_fn
        self.fallback = fallback

    def build(
        self, reasoning_trace: List[str], S, artifact_prompts: Sequence[str]
    ) -> str:
        if not reasoning_trace:
            return self.fallback
        return self.summarize_fn(
            reasoning_trace=reasoning_trace,
            superpixels=S,
            artifact_prompts=artifact_prompts,
        )
