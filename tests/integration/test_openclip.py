import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.skipif(
    os.environ.get("MVR_RUN_OPENCLIP_TESTS") != "1",
    reason="set MVR_RUN_OPENCLIP_TESTS=1 to run the real model integration",
)
def test_openclip_encodes_images_and_text_with_lab3_model(tmp_path: Path) -> None:
    from mvr.enrichment.embeddings.openclip import OpenClipEncoder
    from PIL import Image

    red = tmp_path / "red.jpg"
    blue = tmp_path / "blue.jpg"
    Image.new("RGB", (32, 32), "red").save(red)
    Image.new("RGB", (32, 32), "blue").save(blue)

    encoder = OpenClipEncoder(device="cpu")
    image_vectors = encoder.encode_images([red, blue], batch_size=2)
    text_vectors = encoder.encode_text(["a red image", "a blue image"], batch_size=2)

    assert encoder.model_name == "ViT-B-32"
    assert encoder.pretrained == "laion2b_s34b_b79k"
    assert image_vectors.shape == (2, 512)
    assert text_vectors.shape == (2, 512)
    assert image_vectors.dtype == np.float32
    assert text_vectors.dtype == np.float32
    assert np.linalg.norm(image_vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
    assert np.linalg.norm(text_vectors, axis=1) == pytest.approx([1.0, 1.0], abs=1e-5)
