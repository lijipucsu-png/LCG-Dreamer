"""Template curriculum hook for release-local scenario adaptation.

Keep the function signature unchanged. Modify only the returned SUMO episode
options and, when necessary, the release copy of platoon_sumo/scenario.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from platoon_control.rl import _curriculum_options as _base_curriculum_options


def curriculum_options(
    stage: int,
    rng: np.random.Generator,
    config: Any,
    total_steps: int,
) -> tuple[dict[str, Any], str]:
    """Return one randomized episode while preserving the trained I/O schema."""

    options, base_name = _base_curriculum_options(
        stage, rng, config, total_steps
    )
    # Replace these values with the target domain. Preserve the observation
    # keys, six-node ordering, 0.1 s control period, and acceleration limits.
    options.update(
        {
            "speed_reference": float(rng.uniform(22.0, 27.0)),
            "communication_delay_probability": float(
                rng.uniform(0.05, 0.15) if stage >= 2 else 0.0
            ),
            "communication_delay_min": 0.1,
            "communication_delay_max": 0.2,
            "background_demand_scale": float(
                rng.uniform(0.8, 1.2) if stage >= 5 else 0.0
            ),
        }
    )
    return options, f"new_scene/{base_name}"
