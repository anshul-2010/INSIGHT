import os
import tempfile
import numpy as np
from PIL import Image
from Results_Analysis.eval_helpers import load_datasets, evaluate_auroc


def write_dummy_image(path):
    arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def test_load_datasets_and_auroc():
    td = tempfile.mkdtemp()
    for dname in ("dfdc", "sra"):
        real_dir = os.path.join(td, dname, "real")
        fake_dir = os.path.join(td, dname, "fake")
        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        for i in range(3):
            write_dummy_image(os.path.join(real_dir, f"r{i}.png"))
            write_dummy_image(os.path.join(fake_dir, f"f{i}.png"))

    datasets = load_datasets(td)
    assert "dfdc" in datasets or "sra" in datasets

    # test auroc trivial case
    scores = [0.1, 0.4, 0.9]
    labels = [0, 0, 1]
    auc = evaluate_auroc(scores, labels)
    assert isinstance(auc, float)
