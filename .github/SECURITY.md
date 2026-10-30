# Security

Please use GitHub's private vulnerability reporting on this repository for credential or dependency issues. Never include tokens or private medical images in a public issue.

Public inference uses safetensors. The offline checkpoint converter uses PyTorch's restricted loader and explicitly allowed inert metadata types; do not add an unrestricted pickle fallback. Keep model repositories and dependency versions pinned for a deployed release.
