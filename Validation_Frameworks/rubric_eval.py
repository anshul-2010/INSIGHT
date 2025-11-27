# rubric_eval.py
# Uses Anthropic API to score explanations on a 3-dim rubric (clarity, specificity, relevance)
# Expects environment variable ANTHROPIC_API_KEY to be set.

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Tuple

try:
    from anthropic import AI_PROMPT, HUMAN_PROMPT, Anthropic
except Exception:  # pragma: no cover - optional dependency
    Anthropic = None
    HUMAN_PROMPT = "Human:"
    AI_PROMPT = "Assistant:"


@dataclass
class RubricScore:
    clarity: float
    specificity: float
    relevance: float
    comments: str

    def as_dict(self) -> Dict[str, float]:
        return {
            "clarity": float(self.clarity),
            "specificity": float(self.specificity),
            "relevance": float(self.relevance),
        }


class RubricEvaluatorAnthropic:
    def __init__(self, api_key=None, model: str = "claude-2.1", timeout: int = 30, weights=None):
        api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Anthropic API key not found. Set ANTHROPIC_API_KEY or pass api_key.")
        if Anthropic is None:
            raise RuntimeError("anthropic python package not installed. pip install anthropic")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.timeout = timeout
        self.weights = weights or {"clarity": 0.33, "specificity": 0.33, "relevance": 0.34}

    def _make_prompt(self, artifact_name: str, explanation_text: str) -> str:
        rubric_desc = (
            "You are an automated rubric scorer for forensic explanations. "
            "Score the explanation on three metrics in [0,1]:\n"
            "1. clarity – linguistic coherence.\n"
            "2. specificity – concrete references to regions/textures.\n"
            "3. relevance – alignment with the stated artifact hypothesis.\n"
            "Return JSON with fields {\"clarity\": float, \"specificity\": float, \"relevance\": float, \"comments\": string}.\n"
        )
        prompt = (
            f"{HUMAN_PROMPT}\n"
            f"Artifact hypothesis: {artifact_name}\n\n"
            f"Explanation:\n{explanation_text}\n\n"
            f"{rubric_desc}"
            f"{AI_PROMPT}\n"
        )
        return prompt

    def score(self, artifact_name: str, explanation_text: str) -> Tuple[RubricScore, float]:
        prompt = self._make_prompt(artifact_name, explanation_text)
        resp = self.client.completions.create(
            model=self.model,
            prompt=prompt,
            temperature=0.0,
            max_tokens=300,
        )
        raw = resp.get("completion", "") if isinstance(resp, dict) else getattr(resp, "completion", "")
        parsed, comments = self._parse_json(raw)

        score = RubricScore(
            clarity=float(parsed.get("clarity", 0.0)),
            specificity=float(parsed.get("specificity", 0.0)),
            relevance=float(parsed.get("relevance", 0.0)),
            comments=parsed.get("comments", comments),
        )
        G = self._aggregate(score)
        return score, G

    def _aggregate(self, score: RubricScore) -> float:
        w = self.weights
        total = score.clarity * w["clarity"] + score.specificity * w["specificity"] + score.relevance * w["relevance"]
        return max(0.0, min(1.0, float(total)))

    @staticmethod
    def _parse_json(raw: str) -> Tuple[Dict[str, float], str]:
        txt = raw.strip()
        comments = ""
        try:
            start = txt.find("{")
            end = txt.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(txt[start:end])
            else:
                parsed = json.loads(txt)
        except Exception:
            parsed = {}
            comments = raw
        return parsed, comments