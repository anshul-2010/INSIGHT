# INSIGHT Generation Frameworks

Implements Section 5: semantic scoring, uncertainty detection, ReAct reasoning, and Chain-of-Thought synthesis.

## Modules

| File | Description |
| --- | --- |
| `clip_scoring.py` | Dual-granularity CLIP scorer (Eq. 17–18) returning coarse/fine similarities and per-patch diagnostics. |
| `uncertainty_trigger.py` | Heuristics that fire ReAct when CLIP evidence is weak or inconsistent. |
| `react_policy.py` | Lightweight controller that alternates VLM reasoning and actions per Eq. 19. |
| `cot_decoder.py` | Chain-of-Thought summarizer Ψ (Eq. 20) with fallback messaging. |
| `evidence_synthesis.py` | Ω head (Eq. 21) producing structured narratives with clip scores + trace metadata. |
| `semantic_pipeline.py` | High-level orchestrator wiring CLIP → uncertainty → ReAct → CoT → synthesis. |
| `example.py` | Minimal usage snippet that plugs in mock VLM callbacks. |

## Usage

```python
interpreter = SemanticForensicInterpreter(
    prompts=artifact_prompts,
    reason_fn=my_vlm.reason,
    act_fn=my_vlm.act,
    summarize_fn=my_vlm.summarize,
    device="cuda",
    alpha=0.6,
)
evidence, clip_scores = interpreter.run(
    image=sr_image,
    patches_coarse=coarse_patches,
    patches_fine=fine_patches,
    superpixels=hierarchy,
)
```

`evidence` contains the CoT narrative, decision diagnostics, ReAct trace, and CLIP score table, ready for downstream verification or report formatting.***

