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
        if path.name.startswith(".env") or path.suffix in {".ckpt", ".pyc"} or ".release-work" in path.parts:
            failures.append(f"forbidden artifact: {item}")
        if path.is_file() and path.stat().st_size < 5_000_000 and token_pattern.search(path.read_bytes()):
            failures.append(f"possible credential in {item}")
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"Validated {len(paths) - 1} tracked files; no credential matches or training checkpoints.")


if __name__ == "__main__":
    main()
