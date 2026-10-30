# Contributing

Please open an issue for reproducible bugs, model/metadata discrepancies, and documentation improvements. Include the package versions, device, command, and complete error message. Do not attach private patient images, credentials, or copyrighted datasets.

Use Python 3.12 for local validation:

```bash
pip install -e '.[dev]'
ruff check src scripts tests
pytest
```

Changes to inference must preserve strict checkpoint loading and the full Mamba recurrence. Any new measured result must identify its dataset manifest, class mapping, preprocessing, checkpoint digest, and evaluation method. Label reported paper values and new measurements separately.
