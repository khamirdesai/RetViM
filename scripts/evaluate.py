"""Evaluate available class-folder images; never substitute manuscript metrics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    cohen_kappa_score, confusion_matrix, roc_auc_score,
)

from retvim.inference import Predictor, _sha256
from retvim.data import read_manifest, manifest_digest
from retvim.model import CLASS_NAMES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--data", type=Path, help="Class-folder test directory")
    inputs.add_argument("--manifest", type=Path, help="Evaluate only the held-out test entries")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    manifest = read_manifest(args.manifest) if args.manifest else None
    if manifest:
        args.data = Path(manifest["root"])
    elif not args.data.is_dir() or set(p.name for p in args.data.iterdir() if p.is_dir()) != set(CLASS_NAMES):
        parser.error("Data must contain exactly CNV, DME, DRUSEN, NORMAL class directories.")
    predictor = Predictor(args.weights, device=args.device)
    records, targets, probabilities = [], [], []
    for target, label in enumerate(CLASS_NAMES):
        files = ([args.data / r["path"] for r in manifest["splits"]["test"] if r["label"] == label]
                 if manifest else sorted(p for p in (args.data / label).rglob("*")
                                         if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}))
        if not files:
            parser.error(f"No images found for {label}.")
        for path in files:
            with Image.open(path) as image:
                result = predictor.predict(image)
            targets.append(target)
            probabilities.append([p["probability"] for p in result["probabilities"]])
            records.append({"path": str(path.relative_to(args.data)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "true_label": label, **result})
    scores = np.asarray(probabilities)
    predicted = scores.argmax(axis=1)
    weight_digest = _sha256(args.weights)
    report = {
        "kind": "measured_evaluation", "scope": "Only images provided to this invocation; no patient-level independence inferred.",
        "class_order": list(CLASS_NAMES), "n": len(targets), "weights_sha256": weight_digest,
        "manifest_sha256": manifest_digest(args.manifest) if args.manifest else None,
        "accuracy": accuracy_score(targets, predicted),
        "balanced_accuracy": balanced_accuracy_score(targets, predicted),
        "roc_auc_macro_ovr": roc_auc_score(targets, scores, multi_class="ovr", average="macro"),
        "cohens_kappa": cohen_kappa_score(targets, predicted),
        "classification_report": classification_report(targets, predicted, target_names=CLASS_NAMES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(targets, predicted, labels=range(4)).tolist(),
        "predictions": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measured {len(targets)} images; accuracy {report['accuracy']:.6f}; wrote {args.output}")


if __name__ == "__main__":
    main()
