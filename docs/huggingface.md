---
language: en
license: other
license_name: rights-reserved-pending
license_link: https://huggingface.co/khamirdesai/RetViM/blob/main/LICENSE_WEIGHTS
library_name: pytorch
pipeline_tag: image-classification
tags:
  - retinal-oct
  - vision-transformer
  - mamba
  - research
base_model: timm/vit_base_patch16_224.augreg_in21k_ft_in1k
---

# RetViM

**Sequential Hybrid Vision Transformer with MedMamba for Retinal Disease Classification**

Accepted for presentation at **IEEE GSCon 2027**, Paper 533.

Khamir Desai · Mayuri A. Mehta · Sree Saicharan Vadapalli

[Code](https://github.com/khamirdesai/RetViM) · [Checkpoint details](https://github.com/khamirdesai/RetViM/blob/main/docs/checkpoint.md) · [Citation](https://github.com/khamirdesai/RetViM/blob/main/CITATION.cff)

## Model

| Property | Value |
| --- | --- |
| Task | Retinal OCT image classification |
| Labels, in order | CNV, DME, DRUSEN, NORMAL |
| Architecture | ViT-B/16 + 2 convolution/native-Mamba blocks |
| Parameters | 101,223,557 |
| Input | RGB, 224 × 224; OpenCV linear resize; ImageNet normalization |
| Weights | Float32 safetensors, 404.95 MB |

The supplied checkpoint uses **native 1D Mamba**, with pooling fusion and classifier dimensions that differ from the manuscript's summarized SS2D configuration. The linked implementation preserves the trained tensors and loads them strictly. It is a custom PyTorch model, not a Transformers `AutoModel` checkpoint.

## Inference

```bash
git clone https://github.com/khamirdesai/RetViM.git
cd RetViM
pip install -e .
retvim predict --image /path/to/oct.png
```

The first run downloads the pinned checkpoint. For offline inference, add `--weights /path/to/model.safetensors`.

The package verifies the pinned checkpoint's SHA-256:

```text
45c2042229b00b8ac882722c092b5e99708b6ff1d8ab900ae7082972b01480fd
```

## Scope and license

Strict tensor loading, scan equations, and example inference were checked. The full paper benchmark was not independently rerun. Intended for research; scores are uncalibrated and no prospective clinical validation is claimed.

Original code: MIT. **Weights: license pending; rights reserved** under `LICENSE_WEIGHTS`. The manuscript, publisher PDF, and paper figures are not included. Final publication metadata will be added when available.

Mentor: **Prof. Mayuri A. Mehta**. Upstream research: [MedMamba](https://github.com/YubiaoYue/MedMamba), [VMamba](https://github.com/MzeroMiko/VMamba), and [Mamba](https://github.com/state-spaces/mamba).
