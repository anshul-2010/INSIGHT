# INSIGHT Classifier Frameworks

This directory hosts the forensic classifier stack described in Section 3 of the INSIGHT paper.  
It implements the research workflows for low-resolution (32×32) real vs. synthetic recognition, covering
hybrid spatial–frequency modeling, ensembling, contrastive pretraining, autoencoder probes, and adversarial stress tests.

## Folder Guide

| Path | Purpose |
| --- | --- |
| `train.py` | Unified supervised trainer with optional adversarial augmentation, AMP, checkpointing, and ensembles. |
| `run_simclr.py` | Minimal SimCLR loop for self-supervised pretraining of spatial encoders. |
| `models/` | Backbone zoo (CNN, ResNet, ViT, frequency-only, BNN stub, autoencoder, and the default hybrid backbone). |
| `ensemble.py` | Logit-level ensemble and stacking helpers. |
| `insight_dataset.py` | Dataset wrapper with dual-view augmentation for contrastive learning. |
| `attacks.py` / `adv_train.py` | FGSM/PGD attacks and utilities for robustness sweeps. |
| `utils.py` | Shared helpers, including a fast DCT implementation for spectral modeling. |

An additional `models/README.md` provides per-architecture notes.

## Default Hybrid Backbone

`models/hybrid_backbone.py` fuses three complementary cues:

1. **Spatial stream** — a ResNet-style encoder with channel/spatial attention and a learnable high-pass filter to enhance weak forensic cues.
2. **Frequency stream** — an efficient DCT-driven encoder that isolates aliasing, checkerboard, and spectral residual artifacts.
3. **Projection head** — merges both embeddings into a 256-d latent representation whose gradients remain stable for GradCAM and downstream scoring modules.

The forward API returns `(logits, embedding, aux)` where `aux` exposes activation maps for interpretability.

## Usage

### Supervised training

```bash
cd Classifier_Frameworks
python -m train \
  --data /path/to/dataset \
  --arch hybrid \
  --batch 256 \
  --epochs 100 \
  --adv --adv-eps 0.02 --adv-iters 7 \
  --amp \
  --save-dir runs/hybrid_exp01
```

- Set `--ensemble-archs resnet cnn dct` to build a deep ensemble instead of a single backbone.
- Enable the autoencoder detector via `--arch autoencoder` to inject reconstruction-error features as an auxiliary signal.
- All runs log metrics to `history.json` and store the best checkpoint at `<save-dir>/best.pt`.

### Contrastive warm start

```bash
python -m run_simclr --data /path/to/dataset
```

This pretrains a spatial encoder using dual augmentations (`augment_twice=True`).  
Fine-tune the resulting checkpoint through `train.py` via `--resume`.

### Robustness evaluation

Use `attacks.py` utilities (FGSM, PGD) or wrap the training loop with `adv_train.py` for dedicated adversarial sweeps.  
`train.py` exposes the same PGD inner loop via `--adv` flags to keep GradCAM gradients stable under perturbations.

## Dataset Expectations

The dataset directory should contain:

```
root/
  real/*.png|jpg
  fake/*.png|jpg
```

All imagery is resized to 32×32 inside the dataset class.  
Set `augment_twice=True` when building loaders for contrastive/self-supervised objectives.

## Extending the Framework

- **New backbones** — drop a file in `models/` implementing `forward -> (logits, embedding, *extras)` and register it in `models/__init__.py`.
- **Artifact-specific cues** — use `SimpleEnsemble` to blend diverse inductive biases (CNN, ViT, frequency models, BNN).
- **Autoencoder probes** — `AutoencoderDetector` mixes latent codes with mean absolute reconstruction errors for subtle manipulations.
- **Adversarial attacks** — plug in additional attacks inside `attacks.py` and call them from `maybe_adv_examples` in `train.py`.

This structure keeps the classifier faithful to the research pipeline: every component emits interpretable activations, supports ensembling, and remains robust under severe degradations.

