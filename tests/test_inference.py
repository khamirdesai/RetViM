import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from retvim.inference import Predictor, preprocess
from retvim.model import CLASS_NAMES, build_model


def test_grayscale_rgb_and_deterministic_preprocessing():
    image = Image.new("L", (320, 180), 127)
    actual = preprocess(image)
    assert actual.shape == (1, 3, 224, 224)
    expected = torch.tensor([(127 / 255 - mean) / std for mean, std in zip(
        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    torch.testing.assert_close(actual[0, :, 0, 0], expected)
    torch.testing.assert_close(actual, preprocess(image.convert("RGB")))


def test_missing_or_unsafe_weights_never_return_random_model(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_model(tmp_path / "missing.safetensors")
    malformed = tmp_path / "metadata.pt"
    torch.save({"metadata": "no tensors"}, malformed)
    with pytest.raises(ValueError):
        build_model(malformed)


@pytest.mark.weights
def test_release_weights_inference():
    weights = os.environ.get("RETVIM_TEST_WEIGHTS")
    if not weights:
        pytest.skip("Set RETVIM_TEST_WEIGHTS to enable real-weight smoke inference.")
    torch.set_num_threads(2)
    predictor = Predictor(weights)
    for path in sorted(Path("examples").glob("*.jpeg")):
        with Image.open(path) as image:
            result = predictor.predict(image)
        assert result["prediction"] in CLASS_NAMES
        assert [p["label"] for p in result["probabilities"]] == list(CLASS_NAMES)
        assert sum(p["probability"] for p in result["probabilities"]) == pytest.approx(1.0, abs=1e-6)
        assert all(0 <= p["probability"] <= 1 for p in result["probabilities"])
