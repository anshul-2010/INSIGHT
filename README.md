<!--
INSIGHT - Integrated Neural Semantics for Interpretable Generative-forensic Hallucination Tracing
Root README: a detailed, reproducible, and research-oriented overview of the repository, code and experiments.
-->

# INSIGHT — Integrated Neural Semantics for Interpretable Generative-Forensic Hallucination Tracing

Authors: Anshul Bagaria

Last updated: 2025-11-27

## Abstract

INSIGHT is an end-to-end research framework that (1) localizes likely generative artifacts in low-resolution images with a super-resolution and attention pipeline; (2) attributes semantic meaning to these local regions using CLIP-driven scoring and LLM-based reasoning; and (3) evaluates the generated explanations using robust metrics and an LLM-as-a-judge. The repository provides all code and tooling necessary for developing, reproducing, ablating, and extending the pipeline for academic and forensic research.

This README documents the high-level method, software modules, example commands for experiments, reproducibility guidance, and evaluation scripts used to produce the results.

---

## Contributions (Summary)

- Stage 0: Lightweight forensic classifier stacks including spatial, spectral and hybrid backbones (ResNet-style, TinyViT, DCT-based spectral streams), training scripts, adversarial training support, and contrastive pretraining utilities.
- Stage 1: DRCT super-resolution to recover fine-grained details from low-resolution images; GradCAM attention to localize discriminative regions; superpixel-aware patch extraction that weights patches by GradCAM activation.
- Semantic & Explanation Stage: CLIP-based semantic scoring for patches; ReAct-style reasoning + CoT narrative synthesis; evidence synthesis module that merges visual and semantic cues to produce human-readable forensic explanations.
- Validation & Evaluation: G-Eval style rubric scoring for textual explanations (Anthropic-based or heuristics fallback), LLM-as-a-judge multimodal evaluation, and binary detection baselines for final performance metrics (AUROC, accuracy, false-support rate).

---

## Directory structure & module responsibilities

- `Classifier_Frameworks/` — classifiers, models, training scripts, adversarial utilities, and evaluation helpers for spatial/spectral/hybrid backbones.
- `Extraction_Frameworks/` — stage-1 pipeline: DRCT super-resolution, GradCAM utilities, superpixel hierarchy builder, patch extractor.
- `Generation_Framworks/` — CLIP scoring, ReAct & Chain-of-Thought modules, evidence synthesis, and other prompt-driven grounding modules.
- `Validation_Frameworks/` — LLM-as-a-judge adapters, rubric evaluation wrappers, paraphraser, re-ranker and report generation utilities.
- `Results_Analysis/` — top-level analysis scripts that run full experimental evaluations and produce tables/metrics for the paper.
- `LICENSE` — license for distribution.

Each module has a README describing details (Inspect `Classifier_Frameworks/README.md`, `Extraction_Frameworks/README.md`, `Generation_Framworks/README.md`, `Validation_Frameworks/README.md`).

---

## Pipeline overview (method)

The INSIGHT pipeline is implemented as a staged architecture:

1. Stage 0 (Classifier): Train or load a lightweight forensic classifier on low-resolution images (32×32). Classifiers include architectures like ResNetSmall, TinyViT, and hybrid spatial–frequency models. Training supports supervised and adversarial training, optionally using SimCLR pretraining for the spatial encoder.

2. Stage 1 (Extraction): Recover structure and localize evidence:
   - Use DRCT (Degradation-Robust Convolutional Transformer) to super-resolve the input image.
   - Apply GradCAM using a classifier to obtain attention/activation maps at the SR resolution.
   - Convert GradCAM activations to a superpixel hierarchy and derive attention-weighted patches using `patch_extractor` and `superpixel` modules.

3. Stage 2 (Semantic Reasoning & Evidence Synthesis): For extracted patches:
   - CLIP-based scoring at patch and image levels to compute similarities to artifact prompts.
   - ReAct-style reasoning that runs an LLM-guided reasoner and action policy to gather evidence and decide when to stop.
   - CoT-based summarization of reasoning traces into a human-readable explanation
   - Evidence synthesis aggregates the CLIP scores, DRCT cues, ReAct traces, and CoT narrative into final forensic evidence.

4. Validation and scoring: Evaluate the method using measures like AUROC (binary detection), G-Eval rubric scores (clarity, specificity, relevance), and a multimodal LLM judge that evaluates the generated explanation for groundedness and verdict correctness.

Figure (optional text):
SR -> GradCAM -> Superpixel masks -> Attention-weighted patches -> CLIP scoring -> ReAct + CoT -> Evidence synthesis -> Judge & Metrics

---

## Installation & environment

We recommend using Python 3.10+ and a virtualenv or conda environment. The repo is cross-platform (Linux/Windows/macOS), but many research experiments expect Linux with CUDA for speed.

Install core dependencies (CPU-only test):

```bash
python -m venv .venv
source .venv/bin/activate  # Unix
# or on Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r Classifier_Frameworks/requirements.txt
pip install -r Results_Analysis/requirements.txt
```

Optional packages for advanced features:
- CLIP: pip install git+https://github.com/openai/CLIP.git
- Anthropic: pip install anthropic (for G-Eval)
- OpenAI / HuggingFace's `transformers` for LLM pipelines used in Validation/Generation frameworks.

GPU instructions (recommended for experiments): install `torch` and `torchvision` for your CUDA version per the official PyTorch docs.

---

## Quickstart & example commands

The repository includes runnable scripts for key steps. The examples below assume you are in the repository root and the `data` folder contains datasets prepared as below.

1) Run Stage-1 pipeline on a single image and save outputs (SR, GradCAM, patches):

```bash
python Extraction_Frameworks/stage1_pipeline.py --image /path/to/img.png --device cpu --patch_size 32 --sp_levels 150
```

2) Train a classifier on a dataset:

```bash
python -m Classifier_Frameworks.train --data /path/to/data --arch resnet --batch 128 --epochs 40 --device cuda --adv
```

3) Run SimCLR pretraining (optional):

```bash
python -m Classifier_Frameworks.run_simclr --data /path/to/data
```

4) Run analysis: AUROC computation for baselines and INSIGHT method:

```bash
python Results_Analysis/overall_performance.py --data_root ./data --datasets dfdc sra --device cpu --no_clip
```

5) Evaluate explanations using G-Eval rubric (requires Anthropic API for real experiments):

```bash
export ANTHROPIC_API_KEY="sk-..."
python Results_Analysis/geval_scoring.py --data_root ./data --dataset dfdc --subset 100
```

6) Run multimodal judge evaluation:

```bash
python Results_Analysis/judge_evaluation.py --data_root ./data --dataset sra --subset 200
```

7) Run test smoke checks and unit tests:

```bash
python Results_Analysis/test_smoke.py
pytest Results_Analysis/tests
```

---

## Dataset expectations & recommended evaluation protocol

The repo expects datasets to be structured as:

```
<data_root>/<dataset_name>/real/*.png
<data_root>/<dataset_name>/fake/*.png
```

Supported dataset names in the experiments include `dfdc` and `sra`. The smoke-test dataset can contain a few synthetic images for debugging.

**Metrics used in the paper experiments**:
- AUROC (Area Under the ROC curve) for binary detection baselines.
- G-Eval / rubric scores (clarity, specificity, relevance) averaged across explanations.
- Judge accuracy / false-support rate for LLM-as-a-judge evaluation.

---

## Reproducibility & configuration

To reproduce experiments reliably:

1. Use fixed seeds: `seed = 0; torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)`.
2. Run the same command from the repository root so imports resolve correctly and stage-1 outputs are saved to the same path.
3. Use a fixed `--subset` seed or a list of predetermined test IDs for toggling evaluation subsets.
4. Configure optional components (CLIP, Anthropic) via environment variables or explicit CLI flags.

Example: reproduce AUROC and explanation score evaluation on DFDC:

```bash
python Results_Analysis/overall_performance.py --data_root /path/to/data --datasets dfdc --device cuda --no_clip
python Results_Analysis/geval_scoring.py --data_root /path/to/data --dataset dfdc --subset 500
```

If you want to publish exact numbers, include the git commit hash, the environment spec (pip freeze), and the dataset sub-splitting details (indices or seed) with each experimental run.

---

## Developer notes & extensibility

- To add a new baseline, implement a method under `Results_Analysis/eval_helpers.py` and update `overall_performance.py` to call it.
- Replace DRCT or any other module with a checkpointed model via the `Extraction_Frameworks` API. `load_drct` returns an `nn.Module` that performs super-resolution on a LR Tensor containing an image.
- Add new LLM adapters in `Validation_Frameworks` by subclassing `JudgeBackend` and adding a new adapter class that implements the `judge` method.
- Add tests under `Results_Analysis/tests` for new code and CI integration.

---

## Results and Expected Outputs

The experiments in the paper present AUROC comparisons across multiple datasets and baselines, explanation quality (G-Eval) and judge evaluation metrics. Re-run scripts with `--subset` or using full dataset sizes to reproduce the reported numbers. The `Results_Analysis` scripts produce tables and console prints by default; you can easily add `--save-json` or `--out` flags to persist experiment outputs.

---

## Citation and License

If you use INSIGHT in your research, please cite the project and specific modules used. We provide `LICENSE` at the root (choose MIT/Apache/Proprietary depending on the included license). If you would like a bibtex entry, add it here.

---

## Contact & Acknowledgements

Please open issues for bugs, feature requests, or to request datasets. Acknowledgements: OpenAI/CLIP, Anthropic, and other research code projects that inspired components used in this work.
