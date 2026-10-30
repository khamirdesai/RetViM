# Training and evaluation

The package provides a standalone PyTorch training implementation for the **native 1D Mamba checkpoint architecture**. It does not switch to SS2D or silently load incompatible weights.

## Dataset

Download [paultimothymooney/kermany2018 on Kaggle](https://www.kaggle.com/datasets/paultimothymooney/kermany2018) separately. The [original dataset record](https://data.mendeley.com/datasets/rscbjbr9sj/2) describes four classes and filenames as `CLASS-randomized_patient_ID-frame.jpeg`. Observe the terms of the dataset version you obtain; the code's MIT license does not license the dataset.

Pass the directory that directly contains `train/`, `val/` and `test/`, with `CNV`, `DME`, `DRUSEN`, `NORMAL` inside each. `val/` may be absent. No images are bundled here.

```bash
pip install -e '.[train,eval]'
retvim prepare-data --data /path/to/OCT2017 --output runs/data/manifest.json
```

This pools the supplied train and validation images and creates a seed-42 stratified split by the numeric patient ID, using 10% of patients for validation. The supplied test images remain fixed. It records relative paths, labels, patient IDs and SHA-256 hashes. Patient IDs are compared across classes, without prefixing them with a class label. Conflicting labels, patient overlap and identical file bytes across splits cause an error. Resolve source-data conflicts explicitly rather than removing test cases to improve a score. Exact hashing does not detect transformed or near-duplicate images.

Training rechecks the manifest identities and every image hash before starting. Keep the manifest with your local experiment. Its root is an absolute local path; regenerate it if the dataset moves.

## Run training

```bash
retvim train --manifest runs/data/manifest.json --output runs/retvim \
  --device cuda --batch-size 4 --workers 4
```

The pretrained ViT backbone downloads on the first training run. The released RetViM weights are not used unless you explicitly supply `--weights`. For supported CUDA hardware, `--precision bf16` enables mixed precision. The default is float32. `--no-pretrained` initializes the ViT randomly for an offline smoke test; it is not the transfer-learning setup.

| Setting | Value |
| --- | --- |
| Optimizer | AdamW, learning rate 1e-4, weight decay 0.05, betas 0.9/0.999 |
| Schedule | 3 warmup epochs, then cosine decay with a 1e-7 floor |
| Loss | Inverse-training-frequency weighted cross entropy, label smoothing 0.1 |
| Regularization | Mixup alpha 0.2 / CutMix alpha 1.0, combined probability 0.3 |
| Frozen backbone | Patch embedding, CLS token and first 6 ViT blocks |
| Maximum epochs | 50; early stopping on validation loss, patience 12 |
| Checkpoint selection | Highest validation accuracy; test set never selects weights |
| Gradient clipping | Global norm 1.0 |
| Seed | 42 |

Batch size defaults to 4 to limit memory; the historical notebook used 128 with BF16 on an A100. The portable recurrence prioritizes correctness and can be slow, especially during backward. ViT gradient checkpointing reduces activation memory. Choose batch size for your hardware; a changed batch size changes optimization.

The run directory stores `config.json`, `history.jsonl`, `best.safetensors` and `last.safetensors`. Checkpoints are inference-ready model tensors. `--weights` starts a new fine-tuning run; it does not resume optimizer/scheduler/RNG state. A nonempty output directory is rejected to protect earlier experiments.

```bash
retvim predict --image /path/to/oct.jpeg --weights runs/retvim/best.safetensors
python scripts/evaluate.py --manifest runs/data/manifest.json \
  --weights runs/retvim/best.safetensors --output runs/retvim/test.json --device cuda
```

Evaluation uses only manifest test entries and reports accuracy, balanced accuracy, per-class precision/recall/F1, macro one-vs-rest AUROC, Cohen's kappa, a confusion matrix and per-image predictions. Its output includes manifest and weights hashes. It never substitutes paper metrics.

## Reproduction limits

The historical notebook's exact split manifests and unpinned augmentation-library versions were not supplied. This implementation uses explicit Albumentations 2 operations and parameters; it does not claim bitwise equivalence to the notebook's older augmentation APIs or fused CUDA/BF16 kernels. It omits the notebook's late-run stochastic weight averaging and test-time augmentation. The saved epoch-19 artifact preceded the configured SWA phase.

This training recipe creates a new, auditable experiment. It does not guarantee the manuscript's 99.90% result, recreate its exact training split, or establish a new leaderboard ranking. See [checkpoint differences](checkpoint.md).

## Code validation

Validated on 31 October 2026 with Python 3.12, PyTorch 2.9.1, timm 1.0.30 and Albumentations 2.0.8 on CPU:

- 18 automated tests passed, including the released-weight inference test, scan forward equations, gradient propagation, fresh training initialization/freezing, CutMix area correction, split overlap rejection and changed-image hash rejection.
- A full 101.22M-parameter model initialized from the released weights completed one synthetic-data epoch with 12 optimizer steps. Exported tensors were strictly reloaded for prediction and a four-image synthetic evaluation. The classifier changed and the first frozen ViT block stayed unchanged.
- These are implementation checks on synthetic inputs, not a medical benchmark or validation of the 99.90% paper result. CUDA/BF16 training and full-dataset convergence were not tested in this environment.

```bash
pip install -e '.[dev,train,eval]'
ruff check src scripts tests
pytest
RETVIM_TEST_WEIGHTS=/path/to/model.safetensors pytest -m weights
```
