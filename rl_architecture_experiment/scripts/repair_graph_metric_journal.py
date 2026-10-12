"""Remove an abandoned post-checkpoint tail before a resumed graph run."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import torch


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_rows(path: Path) -> list[tuple[str, dict[str, Any]]]:
    rows: list[tuple[str, dict[str, Any]]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            rows.append((line, row))
    if not rows:
        raise ValueError(f"metric journal is empty: {path}")
    return rows


def repair(
    metric_path: Path,
    checkpoint_path: Path,
    audit_path: Path,
    truncate_at_checkpoint: bool = False,
) -> dict[str, Any]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    checkpoint_step = int(checkpoint["total_steps"])
    checkpoint_episode = int(checkpoint["episodes"])
    rows = _load_rows(metric_path)
    steps = [int(row.get("step", -1)) for _, row in rows]
    anchor_candidates = [
        index
        for index, (_, row) in enumerate(rows)
        if int(row.get("step", -1)) == checkpoint_step
        and int(row.get("training/episode", -1)) == checkpoint_episode
    ]
    if not anchor_candidates:
        raise ValueError(
            "checkpoint episode anchor not found for "
            f"episode {checkpoint_episode}, step {checkpoint_step}"
        )
    anchor_index = anchor_candidates[-1]
    if truncate_at_checkpoint:
        resume_index = len(rows)
    else:
        regressions = [
            index
            for index in range(anchor_index + 1, len(steps))
            if steps[index] < steps[index - 1]
        ]
        candidates = [
            index
            for index in regressions
            if checkpoint_step < steps[index] <= checkpoint_step + 1000
        ]
        if not candidates:
            raise ValueError(
                "no post-checkpoint resume boundary found for "
                f"step {checkpoint_step} in {metric_path}"
            )
        resume_index = candidates[-1]
    removed = rows[anchor_index + 1 : resume_index]
    if not removed:
        raise ValueError("resume boundary does not contain an abandoned tail")

    checkpoint_hash = _sha256(checkpoint_path)
    repaired_at = datetime.now(timezone.utc).isoformat()
    marker = {
        "step": checkpoint_step,
        "training/journal_repair": 1.0,
        "training/journal_checkpoint_episode": float(checkpoint_episode),
        "training/journal_removed_lines": float(len(removed)),
        "training/journal_removed_max_step": float(
            max(int(row.get("step", -1)) for _, row in removed)
        ),
        "training/journal_checkpoint_sha256": checkpoint_hash,
        "training/journal_repaired_at": repaired_at,
    }
    repaired_lines = [line for line, _ in rows[: anchor_index + 1]]
    repaired_lines.append(json.dumps(marker, sort_keys=True) + "\n")
    if not truncate_at_checkpoint:
        repaired_lines.extend(line for line, _ in rows[resume_index:])

    suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = metric_path.with_name(
        f"{metric_path.name}.pre_repair_{suffix}.bak"
    )
    shutil.copy2(metric_path, backup_path)
    temporary = metric_path.with_suffix(metric_path.suffix + ".repair.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        stream.writelines(repaired_lines)
    temporary.replace(metric_path)

    audit = {
        "repaired_at": repaired_at,
        "metric_path": str(metric_path.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_step": checkpoint_step,
        "checkpoint_episode": checkpoint_episode,
        "source_lines": len(rows),
        "repaired_lines": len(repaired_lines),
        "anchor_line": anchor_index + 1,
        "resume_line": None if truncate_at_checkpoint else resume_index + 1,
        "truncate_at_checkpoint": truncate_at_checkpoint,
        "removed_lines": len(removed),
        "removed_step_min": min(int(row.get("step", -1)) for _, row in removed),
        "removed_step_max": max(int(row.get("step", -1)) for _, row in removed),
        "backup_path": str(backup_path.resolve()),
        "backup_sha256": _sha256(backup_path),
        "repaired_sha256": _sha256(metric_path),
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metric_path", type=Path)
    parser.add_argument("checkpoint_path", type=Path)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--truncate-at-checkpoint", action="store_true")
    arguments = parser.parse_args()
    print(
        json.dumps(
            repair(
                arguments.metric_path,
                arguments.checkpoint_path,
                arguments.audit,
                truncate_at_checkpoint=arguments.truncate_at_checkpoint,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
