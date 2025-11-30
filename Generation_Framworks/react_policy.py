"""
Lightweight controller for ReAct-style (Reason + Act) forensic prompting (Section 5.2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence


ReasonFn = Callable[..., str]
ActionFn = Callable[..., Dict[str, Any]]


@dataclass
class ReActStep:
    reasoning: str
    action: Dict[str, Any]


class ReActForensics:
    def __init__(
        self, vlm_reason_fn: ReasonFn, vlm_action_fn: ActionFn, max_steps: int = 8
    ):
        self.reason_fn = vlm_reason_fn
        self.action_fn = vlm_action_fn
        self.max_steps = max_steps

    def run(
        self,
        image,
        superpixels,
        artifact_prompts: Sequence[str],
        region_features,
    ) -> (List[str], List[Dict[str, Any]]):
        reasoning_trace: List[str] = []
        actions: List[Dict[str, Any]] = []

        for _ in range(self.max_steps):
            Rt = self.reason_fn(
                image=image,
                S=superpixels,
                artifact_prompts=artifact_prompts,
                region_features=region_features,
                past_reasoning=reasoning_trace,
                past_actions=actions,
            )
            At = self.action_fn(
                image=image,
                S=superpixels,
                region_features=region_features,
                reasoning_step=Rt,
            )
            reasoning_trace.append(Rt)
            actions.append(At)
            if At.get("stop", False):
                break
        return reasoning_trace, actions
