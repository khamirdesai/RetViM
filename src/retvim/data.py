"""Explicit, hashed Kermany image manifests with patient-disjoint splits."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset

from .inference import _sha256, preprocess
from .model import CLASS_NAMES

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def collect_images(directory: Path, root: Path) -> list[dict]:
    if not directory.is_dir():
        raise ValueError(f"Missing class-folder directory: {directory}")
    folders = {p.name for p in directory.iterdir() if p.is_dir() and not p.name.startswith(".")}
    if folders != set(CLASS_NAMES):
        raise ValueError(f"{directory} must contain exactly {CLASS_NAMES}.")
    records = []
    for label in CLASS_NAMES:
        files = sorted(p for p in (directory / label).rglob("*")
                       if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not files:
            raise ValueError(f"No images in {directory / label}")
        for path in files:
            match = re.fullmatch(r"(CNV|DME|DRUSEN|NORMAL)-(\d+)-(\d+)", path.stem)
            if not match or match[1] != label:
                raise ValueError(f"Expected Kermany CLASS-patient-frame filename: {path}")
            records.append({"path": path.relative_to(root).as_posix(), "label": label,
                            "patient": match[2], "sha256": _sha256(path)})
    return records


def validate_splits(splits: dict[str, list[dict]]) -> None:
    """Reject patient, exact-content, and path overlap across all named splits."""
    seen = {key: {} for key in ("path", "patient", "sha256")}
    for split, records in splits.items():
        if not records or {r["label"] for r in records} != set(CLASS_NAMES):
            raise ValueError(f"{split} must have images from all four classes.")
        paths = set()
        for record in records:
            if record["path"] in paths:
                raise ValueError(f"Repeated image path within {split}: {record['path']}")
            paths.add(record["path"])
            for key, values in seen.items():
                value = record[key]
                if value in values and values[value] != split:
                    raise ValueError(f"{key} overlap between {values[value]} and {split}: {record['path']}")
                values[value] = split


def prepare_manifest(data: Path, output: Path, *, val_fraction: float = 0.1, seed: int = 42) -> dict:
    """Repartition train+val by patient; preserve the provided test images."""
    from sklearn.model_selection import train_test_split

    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be strictly between zero and one.")
    root = data.resolve()
    pool = collect_images(root / "train", root)
    if (root / "val").exists():
        pool += collect_images(root / "val", root)
    test = collect_images(root / "test", root)
    patients = {}
    for record in pool:
        patient = record["patient"]
        if patient in patients and patients[patient] != record["label"]:
            raise ValueError(f"Patient {patient} has conflicting class labels; review before splitting.")
        patients[patient] = record["label"]
    ids = sorted(patients)
    _, val_ids = train_test_split(ids, test_size=val_fraction, random_state=seed,
                                 stratify=[patients[p] for p in ids])
    val_ids = set(val_ids)
    splits = {"train": [r for r in pool if r["patient"] not in val_ids],
              "val": [r for r in pool if r["patient"] in val_ids], "test": test}
    validate_splits(splits)
    manifest = {"schema_version": 1, "dataset": "paultimothymooney/kermany2018",
                "root": str(root), "seed": seed, "validation_patient_fraction": val_fraction,
                "protocol": "Stratified patient split of train+val; supplied test held fixed.",
                "class_order": list(CLASS_NAMES), "splits": splits}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("class_order") != list(CLASS_NAMES):
        raise ValueError("Unsupported manifest schema or class order.")
    if set(manifest["splits"]) != {"train", "val", "test"}:
        raise ValueError("Manifest requires train, val, and test splits.")
    validate_splits(manifest["splits"])
    root = Path(manifest["root"]).resolve()
    for records in manifest["splits"].values():
        for record in records:
            image = (root / record["path"]).resolve()
            if not image.is_relative_to(root):
                raise ValueError("Manifest image must be inside the dataset root.")
            match = re.fullmatch(r"(CNV|DME|DRUSEN|NORMAL)-(\d+)-(\d+)", image.stem)
            if not match or match[1] != record["label"] or match[2] != record["patient"]:
                raise ValueError(f"Manifest identity mismatch: {image}")
            if _sha256(image) != record["sha256"]:
                raise ValueError(f"Dataset image changed since manifest creation: {image}")
    return manifest


def manifest_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class OCTDataset(Dataset):
    def __init__(self, root: Path, records: list[dict], transform=None):
        self.root, self.records, self.transform = root, records, transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        import numpy as np

        record = self.records[index]
        with Image.open(self.root / record["path"]) as image:
            if self.transform is None:
                tensor = preprocess(image)[0]
            else:
                tensor = self.transform(image=np.asarray(image.convert("RGB")))["image"]
        return tensor, CLASS_NAMES.index(record["label"])
