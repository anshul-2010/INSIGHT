# Results_Analysis

This folder contains scripts and utilities used to evaluate the INSIGHT pipeline for research purposes. The scripts are designed to be reproducible and modular: they leverage the Extraction, Classifier and Generation frameworks in this repo (Stage-1 extraction, CLIP-based semantic scoring, classifier backbones, and LLM-based scoring/raters).

## Project layout

- `eval_helpers.py` — small family of helpers and a `ForensicEvaluator` wrapper that connects Stage-1, CLIP, and classifier backbones. Includes `load_datasets`, `evaluate_auroc`, and `table_print` helpers.
- `overall_performance.py` — entry point to compute AUROC for the binary detection task (real vs synthetic) across multiple baselines and the INSIGHT method.
- `geval_scoring.py` — G-Eval-style evaluation for textual explanations. Uses a rubric scorer (Anthropic by default, but with a fallback) to evaluate explanation quality on rubrics such as clarity, specificity and relevance.
- `judge_evaluation.py` — evaluates `LLM-as-a-judge` performance on multimodal verification tasks, computing accuracy and correlations between judge confidence and salience.
- `test_smoke.py` — quick smoke tests for the pipeline, including model forward-shape checks and a tiny CLIP dry-run (skipped if CLIP not available).
- `requirements.txt` — minimal dependencies for running the analysis scripts and tests.

## Getting started

1. Install dependencies (recommended in a virtualenv/conda):

```bash
python -m pip install -r Results_Analysis/requirements.txt
python -m pip install scikit-learn tabulate
# Optional: pip install clip anthropic opencv-python
```

2. Make sure the repo root is on `PYTHONPATH` (or run the scripts from the workspace root). For example:

```bash
cd /path/to/INSIGHT
python Results_Analysis/overall_performance.py --data_root data --datasets dfdc sra
```

3. Datasets should be arranged as:

```
<data_root>/dfdc/real/*.png
<data_root>/dfdc/fake/*.png
<data_root>/sra/real/*.png
<data_root>/sra/fake/*.png
```

## Scripts and usage

All scripts accept a `--data_root` argument pointing to a folder containing dataset(s) arranged as above. Additional flags are documented below.

### overall_performance.py

Computes AUROC for multiple baselines and INSIGHT. It uses `ForensicEvaluator` to score each image.

Usage:

```bash
python Results_Analysis/overall_performance.py --data_root path/to/data --datasets dfdc sra --device cpu
```

Options (main ones):
- `--data_root` — directory containing dataset subfolders
- `--datasets` — dataset names to evaluate (`dfdc`, `sra`, ...)
- `--device` — `cpu` or `cuda` (default: `cpu`)
- `--no_clip` — disable CLIP scoring

Outputs: prints a table of AUROC per dataset and method by default. You can modify the script to save JSON/CSV artifacts.

### geval_scoring.py

Evaluate textual explanation quality using an LLM rubric (Anthropic recommended). If Anthropic is not available, the script falls back to a simple heuristic.

Usage:

```bash
python Results_Analysis/geval_scoring.py --data_root path/to/data --dataset dfdc --subset 100
```

Options:
- `--dataset` — which dataset to use (default: `dfdc`)
- `--subset` — number of images to run on for quick tests

If you want to use Anthropic for robust G-Eval metrics, set `ANTHROPIC_API_KEY`:

```bash
export ANTHROPIC_API_KEY="sk-..."
```

### judge_evaluation.py

Evaluates an LLM-as-a-judge on the `sra` dataset using outputs from the semantic interpreter. It computes accuracy (verdict vs label), false-support rate, and correlation between judge confidence and superpixel salience.

Usage:

```bash
python Results_Analysis/judge_evaluation.py --data_root path/to/data --dataset sra --subset 200
```

### test_smoke.py

A small script that sanity checks the forward pass of models and the ForensicEvaluator, and demonstrates a CLIP dry-run. Run it from the repo root:

```bash
python Results_Analysis/test_smoke.py
```

## Notes on runtime and dependencies

- CLIP is optional but required for scoring patches with the Forensic CLIP scorer. If unavailable, the `ForensicEvaluator` will still run but will return fallback values where CLIP is used.
- The results scripts often rely on `Extraction_Frameworks/run_stage1.py` (Stage-1 pipeline) to produce superpixels, patches and GradCAMs. Make sure that `Extraction_Frameworks` is in your `PYTHONPATH` or run scripts from repo root.
- For GPU-based experiments, install the appropriate `torch` and `torchvision` versions for your GPU environment, and pass `--device cuda`.

## Reproducibility and extensions

- To make the analysis deterministic, set consistent seeds (e.g., `torch.manual_seed`, `np.random.seed`) inside the scripts or in your driver script.
- The `ForensicEvaluator` class is intended to be a convenience wrapper; you can extend it by adding new baselines or saving per-image diagnostic reports (e.g., per-patch CLIP scores).

## Example end-to-end demo (suggested)

1. Prepare dataset under `./data` as specified.
2. Run stage1 on a single image to generate patches and GradCAM, or invoke the `Extraction_Frameworks/stage1_pipeline.py` CLI:

```bash
python Extraction_Frameworks/stage1_pipeline.py --image path/to/img.png --device cpu
```

3. Run `overall_performance.py` for a quick AUROC summary across datasets:

```bash
python Results_Analysis/overall_performance.py --data_root data --datasets dfdc sra --device cpu --no_clip
```

## Troubleshooting

- Import errors: run scripts from repo root or set `PYTHONPATH` to include the repository root so subpackages import properly.
- Missing optional packages: the scripts attempt to degrade gracefully, but some reports will be incomplete. Install optional packages (`clip`, `anthropic`, etc.) if you need those features.
- Memory / GPU: reduce `--subset` to small numbers for quick dev cycles, and use `--device cpu` if GPU isn’t available.

## Contributing and extending

- Add new baselines by extending `ForensicEvaluator` with a method `new_baseline_score(img)` and include it in `overall_performance.py`.
- Save results by modifying the top-level scripts to write JSON/CSV logs (I recommend `jsonlines` or `pandas.DataFrame.to_csv`).

If you'd like, I can:
- Add a demonstration Jupyter notebook that runs a small end-to-end pipeline with visual outputs (SR, GradCAM, patches),
- Add more tests that exercise the `run_stage1` pipeline end-to-end using a small synthetic dataset,
- Implement `--save-json` or `--out` flags for scripts to store results.

---

Last updated: 2025-11-27
