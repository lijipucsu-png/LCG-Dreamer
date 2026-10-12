"""Evaluate a non-overlapping formal episode range for parallel scheduling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .evaluate import _episode, _load_policy, _write_csv
from .protocol import METHODS, MixedLaneProtocol


def evaluate_shard(
    method: str,
    episode_start: int,
    episode_end: int,
    device: str,
) -> Path:
    protocol = MixedLaneProtocol()
    if not 0 <= episode_start < episode_end <= protocol.evaluation_episodes:
        raise ValueError("invalid formal evaluation shard")
    policy = _load_policy(method, protocol, device)
    rows = []
    trace_root = Path(protocol.output_root) / "evaluation" / "traces"
    for episode in range(episode_start, episode_end):
        result, trace = _episode(method, policy, protocol, episode)
        rows.append(result)
        _write_csv(
            trace_root / f"{method}_episode_{episode:03d}.csv", trace
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    output = (
        Path(protocol.output_root)
        / "evaluation"
        / "shards"
        / f"{method}_{episode_start:02d}_{episode_end - 1:02d}.csv"
    )
    _write_csv(output, rows)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("episode_start", type=int)
    parser.add_argument("episode_end", type=int)
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()
    output = evaluate_shard(
        arguments.method,
        arguments.episode_start,
        arguments.episode_end,
        arguments.device,
    )
    print(json.dumps({"complete": True, "output": str(output)}))


if __name__ == "__main__":
    main()
