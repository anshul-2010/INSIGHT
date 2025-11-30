import torch
from Classifier_Frameworks.models.resnet_like import ResNetSmall
from Classifier_Frameworks.models.cnn_backbone import SmallCNN
from Classifier_Frameworks.models.vit_tiny import TinyViT
from Classifier_Frameworks.models.frequency_module import DCTBackbone
from Results_Analysis.eval_helpers import load_datasets, ForensicEvaluator


def test_forward_shapes():
    B = 4
    x = torch.randn(B, 3, 32, 32)
    for Model in [ResNetSmall, SmallCNN, TinyViT, DCTBackbone]:
        if Model is TinyViT:
            m = Model(img_size=32)
        else:
            m = Model()
        logits, feat = m(x)
        assert logits.shape[0] == B
        assert feat.shape[0] == B


if __name__ == "__main__":
    test_forward_shapes()
    print("All smoke tests passed")
    # Test our evaluation helpers on a small synthetic dataset
    fe = ForensicEvaluator(device="cpu")
    print("Forensic evaluator instanced. Running quick CLIP global dry-run...")
    # Create synthetic image
    import numpy as np

    img = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    try:
        g = fe.clip_global_score(img)
        print("CLIP global score:", g)
    except Exception as e:
        print("CLIP scoring requires clip package; skipping:", e)
