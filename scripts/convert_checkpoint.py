#!/usr/bin/env python3
"""Convert the audited original RetViM checkpoint to tensor-only safetensors.

Only the known source SHA-256 is accepted. The restricted PyTorch loader receives
two narrowly allowlisted metadata types; arbitrary pickle loading is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PosixPath

import torch
from safetensors.torch import save_file

from retvim.model import RetViM

SOURCE_SHA256 = "e2e3a17b1cb0fc1feac71bac67d53709119be7ce8aaa6211c774d1b0b3d38c1a"


class _NotebookConfig:
    """Inert container for the original notebook's configuration metadata."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert(source: Path, destination: Path) -> dict:
    actual = sha256_file(source)
    if actual != SOURCE_SHA256:
        raise ValueError("Source SHA-256 does not match the audited RetViM checkpoint.")
    allowed = [(_NotebookConfig, "__main__.Config"), PosixPath]
    with torch.serialization.safe_globals(allowed):
        checkpoint = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    state = {
        key.removeprefix("model."): value.detach().contiguous()
        for key, value in checkpoint["state_dict"].items()
        if key.startswith("model.")
    }
    model = RetViM()
    model.load_state_dict(state, strict=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file(state, str(destination), metadata={
        "format": "pt",
        "architecture": "RetViM-native-mamba1",
        "source_checkpoint": source.name,
    })
    return {
        "source_sha256": actual,
        "weights_sha256": sha256_file(destination),
        "weights_bytes": destination.stat().st_size,
        "model_tensors": len(state),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "epoch": checkpoint["epoch"],
        "global_step": checkpoint["global_step"],
        "strict_load": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(convert(args.checkpoint, args.output), indent=2))


if __name__ == "__main__":
    main()
