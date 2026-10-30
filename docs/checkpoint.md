# Checkpoint

This implementation loads the author-supplied epoch-19 checkpoint with every tensor matched strictly. It executes the complete native Mamba selective-state recurrence in PyTorch, including on CPU.

| Property | Value |
| --- | --- |
| Backbone | `vit_base_patch16_224.augreg_in21k_ft_in1k` |
| Hybrid blocks | 2 parallel convolution/Mamba blocks |
| Mamba | Native Mamba-1; causal 1D patch sequence; state dimension 16 |
| Pooling | CLS, average, maximum, learned attention |
| Pool fusion | 3072 → 768 |
| Classifier | 768 → 512 → 256 → 4; GELU |
| Parameters | 101,223,557 |
| Model tensors | 230 |
| File size | 404,953,996 bytes |
| SHA-256 | `45c2042229b00b8ac882722c092b5e99708b6ff1d8ab900ae7082972b01480fd` |

The manuscript describes four-direction SS2D and a summarized classifier with different dimensions. The supplied tensors implement the native-Mamba architecture above. This release documents that distinction and does not claim an exact reproduction of every manuscript detail.

## Input and output

Images are converted to RGB, resized to 224 × 224 with OpenCV `INTER_LINEAR`, and normalized using ImageNet mean `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`. Evaluation is single-view, float32, without test-time augmentation or adaptation.

Output order is `CNV, DME, DRUSEN, NORMAL`. The model returns logits; the prediction helper applies softmax. Scores are uncalibrated and do not establish clinical certainty.

## Validation

Strict loading, numerical scan checks, and example inference were verified. The complete manuscript benchmark has not been independently rerun. The repository's tests use synthetic input for the inference contract, without redistributing patient or dataset images.

```bash
pip install -e '.[dev,train,eval]'
pytest
RETVIM_TEST_WEIGHTS=/path/to/model.safetensors pytest -m weights
```

To measure performance on your own class-folder dataset:

```bash
pip install -e '.[eval]'
python scripts/evaluate.py --data /path/to/test \
  --weights /path/to/model.safetensors --output evaluation.json
```

Each of `CNV`, `DME`, `DRUSEN`, and `NORMAL` must be a subdirectory. The script reports measured results for those inputs; it does not substitute manuscript metrics.
