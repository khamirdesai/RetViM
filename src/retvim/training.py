"""Standalone training for the native-Mamba architecture; no benchmark substitution."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file
from torch import nn
from torch.utils.data import DataLoader

from .data import OCTDataset, manifest_digest, read_manifest
from .model import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD, RetViM, build_model


def training_transform():
    """Explicit Albumentations 2 settings; historical unpinned APIs are not reused."""
    import albumentations as A
    import cv2
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(224, 224, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=0.5),
        A.Affine(translate_percent=(-0.1, 0.1), scale=(0.85, 1.15), rotate=(-15, 15), p=0.5),
        A.OneOf([A.GaussNoise(std_range=(0.01, 0.025), p=1),
                 A.GaussianBlur(blur_limit=(3, 5), p=1)], p=0.3),
        A.OneOf([A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1),
                 A.CLAHE(clip_limit=2, p=1)], p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(1, 20),
                        hole_width_range=(1, 20), p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2(),
    ])


def configure_trainable(model: RetViM, frozen_blocks: int = 6) -> None:
    if not 0 <= frozen_blocks <= len(model.vit.blocks):
        raise ValueError("Invalid number of frozen ViT blocks.")
    model.requires_grad_(True)
    if frozen_blocks:
        model.vit.patch_embed.requires_grad_(False)
        model.vit.cls_token.requires_grad_(False)
        for block in model.vit.blocks[:frozen_blocks]:
            block.requires_grad_(False)


def initialize_training_model(*, pretrained: bool) -> RetViM:
    model = RetViM(pretrained=pretrained)
    for block in model.medmamba_blocks:
        for layer in block.conv_branch.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode="fan_out")
    groups = [model.pool_fusion, model.classifier, model.attn_pool]
    groups += [block.fusion for block in model.medmamba_blocks]
    for group in groups:
        for layer in group.modules():
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, std=0.02)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    return model


def mix_batch(images, labels, *, probability=0.3, mode=None):
    """Mixup/CutMix; CutMix targets use the actual clipped rectangle area."""
    if images.shape[0] < 2 or np.random.random() >= probability:
        return images, labels, labels, 1.0
    permutation = torch.randperm(len(labels), device=images.device)
    cutmix = mode == "cutmix" or (mode is None and np.random.random() < 0.5)
    alpha = 1.0 if cutmix else 0.2
    lam = float(np.random.beta(alpha, alpha))
    if not cutmix:
        return images * lam + images[permutation] * (1 - lam), labels, labels[permutation], lam
    height, width = images.shape[-2:]
    box_h, box_w = int(height * math.sqrt(1 - lam)), int(width * math.sqrt(1 - lam))
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = max(0, cy - box_h // 2), min(height, cy + box_h // 2)
    x1, x2 = max(0, cx - box_w // 2), min(width, cx + box_w // 2)
    mixed = images.clone()
    mixed[:, :, y1:y2, x1:x2] = images[permutation, :, y1:y2, x1:x2]
    lam = 1 - (y2 - y1) * (x2 - x1) / (height * width)
    return mixed, labels, labels[permutation], lam


def lr_multiplier(step: int, *, total: int, warmup: int, minimum: float) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return max(minimum, 0.5 * (1 + math.cos(math.pi * progress)))


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    np.random.seed(seed)
    random.seed(seed)


def run_training(args) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.patience < 1:
        raise ValueError("Epochs, batch size and patience must be positive; workers must be nonnegative.")
    if args.lr < 1e-7 or args.weight_decay < 0 or args.warmup_epochs < 0:
        raise ValueError("Learning rate must be >= 1e-7; decay and warmup must be nonnegative.")
    if args.precision == "bf16" and (not args.device.startswith("cuda") or
                                     not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 training requires a supported CUDA device; use float32 otherwise.")
    # A fresh output directory prevents silently overwriting an earlier experiment.
    output = Path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Training output must be empty; choose a new run directory.")
    manifest = read_manifest(Path(args.manifest))
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
    transform = training_transform()
    transform.set_random_seed(args.seed)
    root = Path(manifest["root"])
    train = OCTDataset(root, manifest["splits"]["train"], transform)
    val = OCTDataset(root, manifest["splits"]["val"])
    loader_options = dict(num_workers=args.workers, pin_memory=args.device.startswith("cuda"),
                          worker_init_fn=seed_worker, persistent_workers=args.workers > 0)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, generator=generator,
                              drop_last=len(train) >= args.batch_size, **loader_options)
    val_loader = DataLoader(val, batch_size=args.batch_size, shuffle=False, **loader_options)
    model = (build_model(args.weights) if args.weights else
             initialize_training_model(pretrained=not args.no_pretrained))
    configure_trainable(model)
    # Recompute ViT activations during backward to limit training memory.
    model.vit.set_grad_checkpointing(True)
    model.to(args.device)
    counts = torch.tensor([sum(r["label"] == label for r in train.records)
                           for label in CLASS_NAMES], dtype=torch.float32, device=args.device)
    weights = counts.reciprocal()
    weights = weights / weights.sum() * len(CLASS_NAMES)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999))
    total = args.epochs * len(train_loader)
    warmup = min(args.warmup_epochs, args.epochs) * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_multiplier(step, total=total, warmup=warmup, minimum=1e-7 / args.lr))
    output.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "manifest_sha256": manifest_digest(Path(args.manifest)),
              "class_order": list(CLASS_NAMES), "class_weights": weights.cpu().tolist(),
              "torch_version": torch.__version__, "scope": "New native-Mamba training run; not paper reproduction."}
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    best_accuracy, best_loss, stale = -1.0, math.inf, 0
    for epoch in range(args.epochs):
        model.train()
        loss_total, train_count = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(args.device), labels.to(args.device)
            images, y_a, y_b, lam = mix_batch(images, labels)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=torch.bfloat16,
                                enabled=args.precision == "bf16"):
                logits = model(images)
                loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; run stopped.")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            loss_total += float(loss.detach()) * len(labels)
            train_count += len(labels)
        model.eval()
        val_numerator, val_denominator, correct, val_count = 0.0, 0.0, 0, 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                logits = model(images)
                if not torch.isfinite(logits).all():
                    raise RuntimeError("Non-finite validation logits; run stopped.")
                numerator = nn.functional.cross_entropy(logits, labels, weight=weights,
                                                        label_smoothing=0.1, reduction="sum")
                val_numerator += float(numerator)
                val_denominator += float(weights[labels].sum())
                correct += int((logits.argmax(1) == labels).sum())
                val_count += len(labels)
        val_loss, accuracy = val_numerator / val_denominator, correct / val_count
        state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
        save_file(state, str(output / "last.safetensors"))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_file(state, str(output / "best.safetensors"))
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
        else:
            stale += 1
        record = {"epoch": epoch + 1, "train_loss": loss_total / train_count,
                  "val_loss": val_loss, "val_accuracy": accuracy, "val_n": val_count,
                  "learning_rate": optimizer.param_groups[0]["lr"], "best_val_accuracy": best_accuracy}
        with (output / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            break
