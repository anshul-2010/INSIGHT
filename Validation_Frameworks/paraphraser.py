"""
Style-conditioned paraphrasing layer Θ (Eq. 25).
"""

from __future__ import annotations

from typing import Dict, Iterable, List

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except Exception as exc:  # pragma: no cover
    AutoModelForCausalLM = None
    AutoTokenizer = None
    torch = None


class Paraphraser:
    def __init__(self, model_name: str = "google/flan-t5-large", device: str | None = None):
        if AutoModelForCausalLM is None or AutoTokenizer is None or torch is None:
            raise RuntimeError("transformers not installed. Run `pip install transformers accelerate safetensors`.")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _prompt(self, text: str, style: str) -> str:
        return (
            "Rephrase the following forensic explanation while preserving every factual claim. "
            f"Style target: {style}. Keep references to regions untouched.\n"
            f"Original: {text}\nParaphrase:"
        )

    def transform(self, text: str, style: str, max_new_tokens: int = 200) -> str:
        prompt = self._prompt(text, style)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
            )
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return decoded.split("Paraphrase:")[-1].strip()

    def transform_multi(self, text: str, styles: Iterable[str]) -> Dict[str, str]:
        return {style: self.transform(text, style) for style in styles}

    def transform_batch(self, texts: List[str], style: str) -> List[str]:
        return [self.transform(t, style) for t in texts]
