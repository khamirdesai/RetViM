"""Command-line research inference."""

import argparse
import json

from PIL import Image

from ._release import MODEL_REVISION
from .inference import Predictor, download_weights


def main():
    parser = argparse.ArgumentParser(description="RetViM research inference")
    commands = parser.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Classify one retinal OCT image")
    predict.add_argument("--image", required=True)
    predict.add_argument("--weights", help="Local .safetensors path; otherwise download published weights")
    predict.add_argument("--revision", default=MODEL_REVISION)
    predict.add_argument("--device", default="cpu")
    args = parser.parse_args()
    weights = args.weights or download_weights(revision=args.revision)
    with Image.open(args.image) as image:
        result = Predictor(weights, device=args.device).predict(image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
