# RetViM

**Sequential Hybrid Vision Transformer with MedMamba for Retinal Disease Classification**

Accepted for presentation at **IEEE GSCon 2027**.

Khamir Desai · Mayuri A. Mehta · Sree Saicharan Vadapalli

RetViM combines a Vision Transformer with convolution/state-space blocks for four-class retinal OCT classification: **CNV, DME, DRUSEN, NORMAL**.

[Model weights](https://huggingface.co/khamirdesai/RetViM) · [Checkpoint details](docs/checkpoint.md) · [Citation](CITATION.cff)

## Installation

Python 3.10+; Python 3.12 is recommended.

```bash
git clone https://github.com/khamirdesai/RetViM.git
cd RetViM
pip install -e .
```

## Inference

```bash
retvim predict --image /path/to/oct.png
```

The first run downloads the pinned Hugging Face checkpoint. For local weights:

```bash
retvim predict --image /path/to/oct.png --weights /path/to/model.safetensors
```

## Dataset and reported result

[Kermany retinal OCT dataset (Kaggle)](https://www.kaggle.com/datasets/paultimothymooney/kermany2018): CNV, DME, DRUSEN and NORMAL. [Original dataset record](https://data.mendeley.com/datasets/rscbjbr9sj/2).

| Model | Test images | Accuracy (paper-reported) |
| --- | ---: | ---: |
| RetViM | 968 | **99.90% (967/968)** |

This is the highest accuracy among the methods compared in our manuscript. It is not an independently verified overall leaderboard ranking; this release has not rerun the full benchmark.

## Training and evaluation

```bash
pip install -e '.[train,eval]'
retvim prepare-data --data /path/to/OCT2017 --output runs/data/manifest.json
retvim train --manifest runs/data/manifest.json --output runs/retvim --device cuda
python scripts/evaluate.py --manifest runs/data/manifest.json \
  --weights runs/retvim/best.safetensors --output runs/retvim/test.json --device cuda
```

Training exports `best.safetensors` and `last.safetensors`, both compatible with `retvim predict --weights`. Data preparation records hashes, keeps the supplied test set fixed, and rejects patient/file overlap. See [training settings and validation scope](docs/training.md). Download dataset files separately under their own terms.

## Checkpoint

| Backbone | Hybrid blocks | Parameters | Input | Format |
| --- | ---: | ---: | --- | --- |
| ViT-B/16 | 2 | 101.22M | RGB, 224 × 224 | safetensors |

The supplied checkpoint uses **native 1D Mamba**. Its scan backend and classifier differ from the manuscript's summarized SS2D configuration; this implementation preserves the trained tensors. See [checkpoint details](docs/checkpoint.md). The full paper benchmark has not been independently rerun by this release.

## Citation

```bibtex
@unpublished{desai2027retvim,
  title  = {RetViM: Sequential Hybrid Vision Transformer with
            MedMamba for Retinal Disease Classification},
  author = {Desai, Khamir and Mehta, Mayuri A. and
            Vadapalli, Sree Saicharan},
  year   = {2027},
  note   = {Accepted for presentation at IEEE GSCon 2027}
}
```

## Acknowledgments

Mentor: **Prof. Mayuri A. Mehta**. We acknowledge [MedMamba](https://github.com/YubiaoYue/MedMamba), [VMamba](https://github.com/MzeroMiko/VMamba), [Mamba](https://github.com/state-spaces/mamba), and [timm](https://github.com/huggingface/pytorch-image-models).

## License

Original implementation: [MIT](LICENSE). Model weights: license pending; rights reserved. Intended for research; no prospective clinical validation is claimed.

The manuscript, paper figures, and publisher PDF are not distributed here. See [copyright scope](docs/publication.md) and [license scope](docs/license-scope.md).
