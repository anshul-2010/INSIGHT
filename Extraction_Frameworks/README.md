# INSIGHT Extraction Frameworks

Implements Stage 1 of the INSIGHT pipeline (Section 4): recovering structure from 32×32 inputs and localizing forensic artifacts before semantic scoring.

## Modules

| File | Purpose |
| --- | --- |
| `drct_sr.py` | Degradation-Robust Convolutional Transformer super-resolution backbone. Provides `load_drct()` for fast instantiation. |
| `gradcam.py` | GradCAM utilities with auto layer selection and heatmap overlays for attention-guided localization. |
| `superpixel.py` | Generates a multi-level SLIC hierarchy, region masks, and parent-child maps for semantic patch voting. |
| `patch_extractor.py` | Implements attention-weighted superpixel-to-patch decomposition with Eq. 13–16 weighting. |
| `stage1_pipeline.py` | Orchestrates DRCT → GradCAM → superpixels → weighted patches and exposes a CLI for experimentation. |
| `utils_image.py` | Shared I/O helpers for tensor/image conversion. |

## Running Stage 1

```bash
cd Extraction_Frameworks
python stage1_pipeline.py \
  --image ./samples/lr.png \
  --device cuda \
  --scale 4 \
  --patch_size 48 \
  --tau 10.0 \
  --sp_levels 80 180
```

Outputs (LR input, DRCT SR, GradCAM heatmap, and top-N weighted patches) are written to `stage1_out/`. By default the script uses a lightweight dummy classifier for GradCAM; pass your own forensic backbone by calling `run_stage1(..., classifier=my_model, target_layer=my_layer)`.

## Integration Notes

- **Classifier coupling**: the GradCAM stage expects the classifier from Stage 0. Wrap your trained model and pass its last convolutional block as `target_layer` to align attention with the classifier’s discriminative evidence.
- **Superpixel hierarchy**: use `build_hierarchy(image, levels=(50,150,300))` to obtain coarse-to-fine maps and parent relationships for semantic patch voting (Section 4.2).
- **Patch descriptors**: `extract_attention_weighted_patches` returns `PatchDescriptor` objects containing patch crops, weights, and spatial coordinates ready for CLIP scoring or multimodal explanations.
- **Extending DRCT**: swap `load_drct` with a checkpointed model if you fine-tune DRCT; the interface stays `forward(lr_tensor) -> sr_tensor` in `[0,1]`.

