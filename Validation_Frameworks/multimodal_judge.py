# multimodal_judge.py

from __future__ import annotations

import abc
import base64
import json
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


def img_to_b64(img_np: np.ndarray) -> str:
    import cv2

    img = (img_np * 255.0).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    if not ok:
        raise RuntimeError("JPEG encoding failed.")
    return base64.b64encode(buffer).decode("utf-8")


@dataclass
class JudgeResult:
    verdict: bool
    confidence: float
    justification: str

    def to_tuple(self):
        return self.verdict, self.confidence, self.justification

    def to_dict(self):
        return {
            "verdict": bool(self.verdict),
            "confidence": float(self.confidence),
            "justification": self.justification,
        }


class JudgeBackend(abc.ABC):
    @abc.abstractmethod
    def judge(self, image, hypothesis: str, explanation: str) -> JudgeResult:
        raise NotImplementedError


class HTTPJudgeAdapter(JudgeBackend):
    def __init__(self, endpoint_url: str, api_key: str | None = None, timeout: int = 30):
        self.endpoint = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        import requests

        self.requests = requests

    def judge(self, image, hypothesis: str, explanation: str) -> JudgeResult:
        payload: Dict[str, Any] = {"hypothesis": hypothesis, "explanation": explanation}
        if isinstance(image, str):
            payload["image_path"] = image
        else:
            payload["image_b64"] = img_to_b64(image)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        resp = self.requests.post(self.endpoint, json=payload, headers=headers, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        verdict = str(data.get("verdict", "")).lower() in {"yes", "y", "true", "1"}
        confidence = float(data.get("confidence", 0.0))
        justification = data.get("justification", "")
        return JudgeResult(verdict, confidence, justification)


class TextOnlyJudgeAdapter(JudgeBackend):
    def __init__(self, text_llm_fn):
        self.text_llm_fn = text_llm_fn

    def judge(self, image, hypothesis: str, explanation: str) -> JudgeResult:
        img_desc = ""
        if not isinstance(image, str):
            h, w, c = image.shape
            img_desc = f"Image size: {h}x{w}, channels:{c}."

        prompt = (
            "You are a judge ensuring explanations are visually grounded.\n"
            f"Image description: {img_desc}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Explanation: {explanation}\n\n"
            'Respond with JSON: {"verdict":"Yes/No","confidence":0-1,"justification":"text"}'
        )
        resp = self.text_llm_fn(prompt)
        try:
            start = resp.find("{")
            end = resp.rfind("}") + 1
            data = json.loads(resp[start:end])
            verdict = str(data.get("verdict", "")).lower() in {"yes", "y", "true", "1"}
            confidence = float(data.get("confidence", 0.0))
            justification = data.get("justification", "")
        except Exception:
            low = resp.lower()
            verdict = "yes" in low
            confidence = 0.5
            justification = resp.strip()
        return JudgeResult(verdict, confidence, justification)