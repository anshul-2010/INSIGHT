"""
CLIP-driven semantic scoring for coarse/fine superpixel patches (Section 5.1).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Union

import clip
import torch
import torch.nn.functional as F
from torchvision import transforms


PatchLike = Union[torch.Tensor, "PatchDescriptor", dict]


@dataclass
class SemanticScoreBundle:
    unified: torch.Tensor
    coarse: torch.Tensor
    fine: torch.Tensor
    region_scores: List[torch.Tensor]


class ForensicCLIPScorer:
    def __init__(self, model_name: str = "ViT-B/32", device: str = "cuda"):
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device)
        self.model.eval()

    def encode_prompts(self, prompts: Sequence[str]) -> torch.Tensor:
        with torch.no_grad():
            tokens = clip.tokenize(list(prompts)).to(self.device)
            emb = self.model.encode_text(tokens)
        return F.normalize(emb, p=2, dim=1)

    def _patch_to_tensor(self, patch: PatchLike) -> torch.Tensor:
        if isinstance(patch, dict) and "patch_img" in patch:
            data = patch["patch_img"]
        elif hasattr(patch, "patch_img"):
            data = patch.patch_img  # type: ignore[attr-defined]
        else:
            data = patch
        if isinstance(data, torch.Tensor):
            tensor = data
        else:
            tensor = torch.from_numpy(data)
        if tensor.dim() == 3:
            pass
        elif tensor.dim() == 4:
            tensor = tensor[0]
        else:
            raise ValueError("Patch tensor must be HxWxC or CxHxW.")
        if tensor.shape[0] != 3:
            tensor = tensor.permute(2, 0, 1)
        return tensor.float().clamp(0, 1)

    def _encode_patch(self, patch: PatchLike) -> torch.Tensor:
        tensor = self._patch_to_tensor(patch)
        pil = transforms.ToPILImage()(tensor.cpu())
        prep = self.preprocess(pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model.encode_image(prep)[0]
        return F.normalize(emb, dim=0)

    def _batch_similarity(self, patches: Sequence[PatchLike], prompt_emb: torch.Tensor) -> List[torch.Tensor]:
        sims = []
        for patch in patches:
            zi = self._encode_patch(patch)
            sims.append(F.cosine_similarity(zi.unsqueeze(0), prompt_emb).squeeze(0))
        return sims

    @staticmethod
    def _aggregate(sim_vectors: List[torch.Tensor], weights: Iterable[float] | None = None) -> torch.Tensor:
        mat = torch.stack(sim_vectors)
        if weights is None:
            return mat.mean(dim=0)
        weights = torch.tensor(list(weights), dtype=mat.dtype, device=mat.device)
        weights = weights / (weights.sum() + 1e-9)
        return (weights.unsqueeze(1) * mat).sum(dim=0)

    def compute_dual_granularity_scores(
        self,
        patches_coarse: Sequence[PatchLike],
        patches_fine: Sequence[PatchLike],
        prompt_emb: torch.Tensor,
        alpha: float = 0.5,
    ) -> SemanticScoreBundle:
        sims_coarse = self._batch_similarity(patches_coarse, prompt_emb)
        sims_fine = self._batch_similarity(patches_fine, prompt_emb)

        weights_coarse = [getattr(p, "weight", 1.0) if not isinstance(p, dict) else p.get("weight", 1.0) for p in patches_coarse]
        weights_fine = [getattr(p, "weight", 1.0) if not isinstance(p, dict) else p.get("weight", 1.0) for p in patches_fine]

        agg_coarse = self._aggregate(sims_coarse, weights_coarse)
        agg_fine = self._aggregate(sims_fine, weights_fine)

        unified = alpha * agg_coarse + (1 - alpha) * agg_fine
        region_scores = sims_coarse + sims_fine
        return SemanticScoreBundle(unified=unified, coarse=agg_coarse, fine=agg_fine, region_scores=region_scores)