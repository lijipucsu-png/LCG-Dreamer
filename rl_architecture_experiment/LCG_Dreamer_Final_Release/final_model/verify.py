"""Integrity, import, model-loading, and optional SUMO checks for the release."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from .demo import run_episode
from .runtime import ReleasePaths, load_final_policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_manifest(root: Path) -> list[dict[str, Any]]:
    manifest = json.loads(
        (root / "RELEASE_MANIFEST.json").read_text(encoding="utf-8-sig")
    )
    checks = []
    for item in manifest["critical_files"]:
        path = root / item["path"]
        actual = _sha256(path) if path.is_file() else None
        checks.append(
            {
                "path": item["path"],
                "exists": path.is_file(),
                "sha256_matches": actual == item["sha256"],
            }
        )
    failures = [row for row in checks if not row["exists"] or not row["sha256_matches"]]
    if failures:
        raise RuntimeError(f"release integrity check failed: {failures}")
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sumo-smoke-steps", type=int, default=0)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    checks = verify_manifest(root)
    paths = ReleasePaths.discover(root)
    policy = load_final_policy(arguments.device, root=root)
    report: dict[str, Any] = {
        "integrity_files": len(checks),
        "model_loaded": True,
        "architecture": policy.controller.architecture_id,
        "device": str(policy.device),
        "cuda_available": torch.cuda.is_available(),
        "predictors_loaded": policy.controller.adapter.teachers is not None,
        "calibration": str(paths.calibration),
    }
    del policy
    if arguments.sumo_smoke_steps > 0:
        report["sumo_smoke"] = run_episode(
            "following",
            seed=42,
            device=arguments.device,
            max_steps=arguments.sumo_smoke_steps,
            trace_path=root / "outputs" / "verification" / "smoke_trace.csv",
        )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
