"""Offline check of the transferred course files (not model weights or GPU readiness)."""
import hashlib
import json
from pathlib import Path


def verify(root):
    manifest = json.loads((root / "bundle-manifest.json").read_text(encoding="utf-8"))
    errors = []
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()):
            errors.append(f"Unsafe manifest path: {name}")
        elif not path.is_file():
            errors.append(f"Missing: {name}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            errors.append(f"Hash mismatch: {name}")
    return errors


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    errors = verify(root)
    if errors:
        raise SystemExit("\n".join(errors))
    print("Course bundle verified. This does not verify installed environments, models, or GPU execution.")
