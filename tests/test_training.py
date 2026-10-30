"""Data-integrity and optimization checks for the standalone training path."""

import copy

import numpy as np
import pytest
import torch
from PIL import Image

from retvim.data import prepare_manifest, read_manifest, validate_splits
from retvim.model import CLASS_NAMES, NativeMamba
from retvim.training import (
    configure_trainable, initialize_training_model, lr_multiplier, mix_batch, training_transform,
)


def create_dataset(root):
    rng = np.random.default_rng(7)
    for split, patients in (("train", (10, 11, 12, 13)), ("val", (20,)), ("test", (30,))):
        for class_index, label in enumerate(CLASS_NAMES):
            directory = root / split / label
            directory.mkdir(parents=True)
            for patient in patients:
                pixels = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(directory / f"{label}-{class_index * 100 + patient}-1.png")


def test_manifest_patient_split_is_repeatable_and_detects_modified_data(tmp_path):
    root = tmp_path / "OCT2017"
    create_dataset(root)
    path = tmp_path / "manifest.json"
    first = prepare_manifest(root, path, val_fraction=0.4)
    second = prepare_manifest(root, tmp_path / "repeat.json", val_fraction=0.4)
    assert first == second
    assert len(first["splits"]["test"]) == 4
    assert read_manifest(path) == first
    record = first["splits"]["train"][0]
    (root / record["path"]).write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        read_manifest(path)


@pytest.mark.parametrize("key", ["patient", "sha256", "path"])
def test_split_overlap_is_rejected(tmp_path, key):
    create_dataset(tmp_path / "data")
    manifest = prepare_manifest(tmp_path / "data", tmp_path / "manifest.json", val_fraction=0.4)
    splits = copy.deepcopy(manifest["splits"])
    splits["test"][0][key] = splits["train"][0][key]
    with pytest.raises(ValueError, match="overlap"):
        validate_splits(splits)


def test_cutmix_targets_match_actual_pixel_area():
    np.random.seed(3)
    torch.manual_seed(4)
    images = torch.stack([torch.zeros(3, 20, 30), torch.ones(3, 20, 30)])
    labels = torch.tensor([0, 1])
    mixed, y_a, y_b, lam = mix_batch(images, labels, probability=1, mode="cutmix")
    # Derive expected mean from target mixture, independent of rectangle geometry.
    expected = y_a.float() * lam + y_b.float() * (1 - lam)
    torch.testing.assert_close(mixed.mean((1, 2, 3)), expected)


def test_train_transform_has_finite_normalized_rgb_output():
    transform = training_transform()
    transform.set_random_seed(42)
    tensor = transform(image=np.zeros((28, 32, 3), dtype=np.uint8))["image"]
    assert tensor.shape == (3, 224, 224)
    assert tensor.dtype == torch.float32
    assert torch.isfinite(tensor).all()


def test_learning_rate_warmup_and_decay():
    def rate(step):
        return lr_multiplier(step, total=100, warmup=10, minimum=0.001)
    assert rate(0) == 0
    assert rate(5) == 0.5
    assert rate(10) == 1
    assert rate(100) == 0.001
    assert rate(500) == 0.001


def test_selective_recurrence_backpropagates_and_updates_parameters():
    torch.manual_seed(42)
    model = NativeMamba(dim=8, d_state=3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    before = model.in_proj.weight.detach().clone()
    loss = model(torch.randn(2, 35, 8)).square().mean()
    loss.backward()
    for name in ("A_log", "D", "x_proj.weight", "dt_proj.weight", "in_proj.weight"):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0
    optimizer.step()
    assert not torch.equal(before, model.in_proj.weight)


def test_fresh_training_initialization_and_backbone_freezing():
    model = initialize_training_model(pretrained=False)
    configure_trainable(model)
    assert not model.vit.patch_embed.proj.weight.requires_grad
    assert not model.vit.cls_token.requires_grad
    assert not model.vit.blocks[5].attn.qkv.weight.requires_grad
    assert model.vit.blocks[6].attn.qkv.weight.requires_grad
    assert model.vit.pos_embed.requires_grad
    assert model.classifier[6].weight.requires_grad
    for block in model.medmamba_blocks:
        time_steps = torch.nn.functional.softplus(block.ssm.dt_proj.bias.detach())
        assert float(time_steps.min()) >= 0.001 - 1e-6
        assert float(time_steps.max()) <= 0.1 + 1e-6
