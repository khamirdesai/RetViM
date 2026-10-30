"""Fail if a tracked release file contains a credential or forbidden runtime artifact."""

import re
import subprocess
from pathlib import Path


def main():
    paths = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    token_pattern = re.compile(rb"(?:hf_[A-Za-z0-9]{25,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})")
    failures = []
    for item in paths:
        if not item:
            continue
        path = Path(item)
        excluded_roots = {"paper", "assets", "examples", "notebooks", "results", "web", ".release-work", "output"}
        forbidden_suffixes = {".ckpt", ".pyc", ".pdf", ".png", ".jpg", ".jpeg", ".webp", ".ipynb"}
        if path.name.startswith(".env") or path.suffix in forbidden_suffixes or path.parts[0] in excluded_roots:
            failures.append(f"forbidden artifact: {item}")
        if path.is_file() and path.stat().st_size < 5_000_000 and token_pattern.search(path.read_bytes()):
            failures.append(f"possible credential in {item}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(paths) - 1} tracked files; code-only scope and credential checks passed.")


if __name__ == "__main__":
    main()
