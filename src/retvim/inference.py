"""Single-view preprocessing and inference for the supplied native-Mamba checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from time import perf_counter

import torch
import cv2
import numpy as np
from PIL import Image

from ._release import MODEL_REPO, MODEL_REVISION, WEIGHTS_FILENAME, WEIGHTS_SHA256
from .model import CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model

DISCLAIMER = (
    "Research demonstration only. Outputs are uncalibrated model probabilities; "
    "they are not a diagnosis or a measure of clinical certainty."
)


def download_weights(*, revision: str = MODEL_REVISION) -> Path:
    """Download the versioned artifact, verifying its release digest when applicable."""
    from huggingface_hub import hf_hub_download

    path = Path(hf_hub_download(MODEL_REPO, WEIGHTS_FILENAME, revision=revision))
    if revision == MODEL_REVISION and WEIGHTS_SHA256:
        digest = _sha256(path)
        if digest != WEIGHTS_SHA256:
            raise ValueError("Downloaded weights do not match the release SHA-256.")
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def preprocess(image: Image.Image) -> torch.Tensor:
    """Match notebook evaluation: RGB, OpenCV INTER_LINEAR resize, ImageNet normalization."""
    pixels = np.asarray(image.convert("RGB"))
    pixels = cv2.resize(pixels, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32) * 255.0
    denominator = np.asarray(IMAGENET_STD, dtype=np.float32) * 255.0
    normalized = (pixels.astype(np.float32) - mean) / denominator
    return torch.from_numpy(normalized.transpose(2, 0, 1).copy()).unsqueeze(0)


class Predictor:
    """Reusable strictly loaded model. The caller serializes concurrent CPU requests."""

    def __init__(self, weights: str | Path, *, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = build_model(weights, device=device)

    @torch.inference_mode()
    def predict(self, image: Image.Image) -> dict:
        started = perf_counter()
        tensor = preprocess(image).to(self.device)
        logits = self.model(tensor)
        probabilities = torch.softmax(logits.float(), dim=-1)[0].cpu()
        if not torch.isfinite(probabilities).all():
            raise RuntimeError("Model returned non-finite probabilities.")
        index = int(probabilities.argmax())
        return {
            "prediction": CLASS_NAMES[index],
            "probabilities": [
                {"label": label, "probability": float(probabilities[i])}
                for i, label in enumerate(CLASS_NAMES)
            ],
            "elapsed_seconds": round(perf_counter() - started, 3),
            "backend": "native-mamba-reference",
            "disclaimer": DISCLAIMER,
        }
