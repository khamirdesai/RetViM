# RetViM

**Sequential Hybrid Vision Transformer with MedMamba for Retinal Disease Classification**

Accepted for presentation at **IEEE GSCon 2027**, Paper 533.

Khamir Desai · Mayuri A. Mehta · Sree Saicharan Vadapalli

RetViM combines a Vision Transformer with convolution/state-space blocks for four-class retinal OCT classification: **CNV, DME, DRUSEN, NORMAL**.

[Model weights](https://huggingface.co/khamirdesai/RetViM) · [Checkpoint details](CHECKPOINT.md) · [Citation](CITATION.cff)

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

## Checkpoint

| Backbone | Hybrid blocks | Parameters | Input | Format |
| --- | ---: | ---: | --- | --- |
| ViT-B/16 | 2 | 101.22M | RGB, 224 × 224 | safetensors |

The supplied checkpoint uses **native 1D Mamba**. Its scan backend and classifier differ from the manuscript's summarized SS2D configuration; this implementation preserves the trained tensors. See [CHECKPOINT.md](CHECKPOINT.md). The full paper benchmark has not been independently rerun by this release.

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

The manuscript, paper figures, and publisher PDF are not distributed here. See [copyright scope](COPYRIGHT.md) and [license scope](LICENSES.md).
