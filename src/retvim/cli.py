"""Command-line research inference."""

import argparse
import json
from pathlib import Path

from PIL import Image

from ._release import MODEL_REVISION
from .inference import Predictor, download_weights


def main():
    parser = argparse.ArgumentParser(description="RetViM training and research inference")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Classify one retinal OCT image")
    predict.add_argument("--image", required=True)
    predict.add_argument("--weights", help="Local .safetensors path; otherwise download published weights")
    predict.add_argument("--revision", default=MODEL_REVISION)
    predict.add_argument("--device", default="cpu")
    prepare = commands.add_parser("prepare-data", help="Create a hashed, patient-disjoint Kermany manifest")
    prepare.add_argument("--data", type=Path, required=True, help="OCT2017 root containing train, val, test")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--val-fraction", type=float, default=0.1)
    prepare.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Train the native-Mamba architecture using an audited manifest")
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--weights", help="Optional initial safetensors (fine-tuning, not optimizer resume)")
    train.add_argument("--no-pretrained", action="store_true", help="Initialize ViT randomly; useful for smoke tests")
    train.add_argument("--device", default="cpu")
    train.add_argument("--precision", choices=("float32", "bf16"), default="float32")
    train.add_argument("--epochs", type=int, default=50)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--weight-decay", type=float, default=0.05)
    train.add_argument("--warmup-epochs", type=int, default=3)
    train.add_argument("--patience", type=int, default=12)
    train.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.command == "prepare-data":
        from .data import prepare_manifest

        manifest = prepare_manifest(args.data, args.output, val_fraction=args.val_fraction, seed=args.seed)
        print(json.dumps({split: len(records) for split, records in manifest["splits"].items()}, indent=2))
        return
    if args.command == "train":
        from .training import run_training

        run_training(args)
        return
    weights = args.weights or download_weights(revision=args.revision)
    with Image.open(args.image) as image:
        result = Predictor(weights, device=args.device).predict(image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
