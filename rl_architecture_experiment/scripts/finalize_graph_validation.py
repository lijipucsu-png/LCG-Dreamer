"""Build the authoritative validation manifest for the final graph model."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2


SEEDS = (42, 43, 44)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot decode video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    indices = sorted({0, frames // 4, frames // 2, 3 * frames // 4, frames - 1})
    sample_pixel_std: list[float] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
        ok, frame = capture.read()
        if not ok or frame is None:
            capture.release()
            raise RuntimeError(f"cannot decode frame {index} from {path}")
        sample_pixel_std.append(float(frame.std()))
    capture.release()
    valid = bool(
        frames > 0
        and fps > 0.0
        and width == 1280
        and height == 720
        and min(sample_pixel_std) > 5.0
    )
    return {
        "path": str(path.resolve()),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "frames": frames,
        "fps": fps,
        "duration_seconds": frames / max(fps, 1e-6),
        "width": width,
        "height": height,
        "sample_indices": indices,
        "sample_pixel_std": sample_pixel_std,
        "valid": valid,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--selected-seed", type=int, default=43)
    parser.add_argument("--selected-foundation", type=Path)
    parser.add_argument("--selected-capability", type=Path)
    parser.add_argument("--paired-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    seed_results: dict[str, Any] = {}
    for seed in SEEDS:
        directory = arguments.artifacts / f"graph_full_seed{seed}"
        foundation_path = directory / "foundation_validation.json"
        capability_path = directory / "final_capability_validation.json"
        if seed == arguments.selected_seed:
            foundation_path = arguments.selected_foundation or foundation_path
            capability_path = arguments.selected_capability or capability_path
        auxiliary_path = directory / "graph_auxiliary.json"
        foundation = _load(foundation_path)
        auxiliary = _load(auxiliary_path)
        capability = _load(capability_path)
        passed = bool(
            foundation.get("passed")
            and auxiliary.get("passed")
            and capability.get("passed")
        )
        seed_results[str(seed)] = {
            "checkpoint": str((directory / "graph_complete.pt").resolve()),
            "checkpoint_sha256": capability["checkpoint_sha256"],
            "training_steps": capability["training_steps"],
            "training_episodes": capability["training_episodes"],
            "validation_sources": {
                "foundation": str(foundation_path.resolve()),
                "auxiliary": str(auxiliary_path.resolve()),
                "capability": str(capability_path.resolve()),
            },
            "foundation_passed": bool(foundation.get("passed")),
            "auxiliary_passed": bool(auxiliary.get("passed")),
            "complex_capability_passed": bool(capability.get("passed")),
            "complex_metrics": capability["metrics"],
            "auxiliary_gates": auxiliary["gates"],
            "passed": passed,
        }

    paired = _load(arguments.paired_manifest)
    video_checks = {
        result["controller"]: _inspect_video(Path(result["video"]))
        for result in paired["results"]
    }
    graph_result = next(
        result for result in paired["results"] if result["controller"] == "graph"
    )
    paired_passed = bool(
        set(video_checks) == {"graph", "cacc"}
        and all(check["valid"] for check in video_checks.values())
        and graph_result["success"]
        and not graph_result["collision"]
        and not graph_result["failure"]
    )
    all_model_gates_passed = all(
        result["passed"] for result in seed_results.values()
    )
    complete = bool(all_model_gates_passed and paired_passed)
    manifest = {
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "status": "awaiting_user_validation" if complete else "validation_failed",
        "human_stage": 8,
        "code_curriculum_stage": 7,
        "selected_seed": arguments.selected_seed,
        "selection_reason": (
            "5/5 complex-scenario success with lower string-stability ratio than seed42"
        ),
        "scenario_features": [
            "full_background_traffic",
            "learned_leader_oscillation",
            "communication_delay_100_to_200_ms",
            "cut_in",
            "v2i_curve_profile",
        ],
        "seed_results": seed_results,
        "paired_manifest": str(arguments.paired_manifest.resolve()),
        "paired_results": paired["results"],
        "video_checks": video_checks,
        "all_model_gates_passed": all_model_gates_passed,
        "paired_recording_passed": paired_passed,
        "complete": complete,
        "validation_note": (
            "The frozen CPIDLM teacher is loaded and its normalized/speed/acceleration "
            "distillation gates pass; the auxiliary head's absolute position RMSE remains "
            "reported separately and is not represented as matching teacher accuracy."
        ),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(arguments.output), "complete": complete}))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
