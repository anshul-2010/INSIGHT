# Validation Frameworks

Implements Section 6: rubric-based evaluation, multimodal verification, audience-conditioned paraphrasing, reranking, and final report assembly.

## Components

| File | Purpose |
| --- | --- |
| `rubric_eval.py` | G-Eval style rubric scorer producing clarity/specificity/relevance vectors and weighted G(Dᵢ). |
| `multimodal_judge.py` | Model-agnostic VLM judge wrapper returning `(verdict, confidence, justification)`; includes HTTP and text-only adapters. |
| `paraphraser.py` | Style-conditioned paraphraser Θ(Dᵢ, s) preserving factual claims. |
| `reranker.py` | Filters and ranks artifacts according to Eq. 27, outputting `ArtifactMetadata`. |
| `report_generation.py` | Builds structured reports with user-selected styles and judge metadata. |
| `pipeline_orchestrator.py` | Ties everything together: rubric → judge → paraphraser → rerank → report, returning intermediate stage outputs for auditing. |

## Usage

```python
from Validation_Frameworks.pipeline_orchestrator import PipelineOrchestrator
from Validation_Frameworks.multimodal_judge import HTTPJudgeAdapter

orchestrator = PipelineOrchestrator(
    anthropic_api_key="...",
    judge_backend=HTTPJudgeAdapter(endpoint_url="https://judge.api"),
)
outputs = orchestrator.run(
    image_np=sr_image,
    artifacts=artifact_hypotheses,
    explanations=forensic_texts,
    styles=("technical", "summary"),
    tau_conf=0.6,
    top_k=5,
)
report = outputs.report
```

The returned `StageOutputs` also exposes rubric vectors, raw judge decisions, and paraphrases to facilitate ablations or manual QA.

