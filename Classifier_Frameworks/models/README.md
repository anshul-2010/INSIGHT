# Model Zoo Cheatsheet

| File | Summary | Notes |
| --- | --- | --- |
| `hybrid_backbone.py` | Default INSIGHT backbone combining spatial (ResNet) and spectral (DCT CNN) streams with attention-guided feature enhancement. | Returns GradCAM-friendly activation maps in the auxiliary output. |
| `resnet_like.py` | Lightweight ResNet baseline used for ablations and SimCLR pretraining. | Shares building blocks with the hybrid spatial stream. |
| `cnn_backbone.py` | Channel/spatial-attention CNN emphasizing micro-textures and high-frequency cues. | Includes a learnable high-pass residual module (`SimpleHighPass`). |
| `vit_tiny.py` | Tiny ViT with small patch embeddings for hierarchically upsampled inputs. | Works best when preceded by DRCT upsampling stages. |
| `frequency_module.py` | Operates purely on DCT-transformed inputs to capture spectral aliasing artifacts. | Uses the optimized `compute_dct_batch` implementation from `utils.py`. |
| `autoencoder.py` | Contains both the base convolutional autoencoder and `AutoencoderDetector`, which augments classification with reconstruction error. | Useful for subtle manipulations where texture cues vanish. |
| `bnn_stub.py` | Binary neural network prototype exploring representational limits under binarization. | Uses sign-based activations/weights (STE). |
| `contrastive_simclr.py` | Encoder + projection head utilities for self-supervised contrastive learning. | Interoperable with any backbone exposing latent features. |

When adding new architectures, keep the forward signature `(logits, embedding, *extras)` so that ensembling and downstream interpretability modules remain plug-and-play.

