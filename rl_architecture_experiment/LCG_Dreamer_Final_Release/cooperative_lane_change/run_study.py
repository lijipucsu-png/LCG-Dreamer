"""Run focused transfer, matched evaluation, and manuscript plotting."""

from __future__ import annotations

import argparse
import json

from .evaluate import evaluate
from .plot import _style, plot_dynamics, plot_lane_trajectories, plot_twenty_scenarios
from .protocol import METHODS, LaneChangeProtocol
from .report import build_report
from .train import train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    arguments = parser.parse_args()
    protocol = LaneChangeProtocol()
    if not arguments.skip_training:
        for method in METHODS:
            train(method, protocol, arguments.device)
    if not arguments.skip_evaluation:
        for method in METHODS:
            evaluate(method, protocol, arguments.device)
    build_report(protocol)
    _style()
    from pathlib import Path

    output = Path("manuscript_trc/figures/lane_change")
    plot_lane_trajectories(protocol, output)
    plot_dynamics(protocol, output)
    plot_twenty_scenarios(protocol, output)
    print(json.dumps({"complete": True, "output": str(output)}))


if __name__ == "__main__":
    main()
