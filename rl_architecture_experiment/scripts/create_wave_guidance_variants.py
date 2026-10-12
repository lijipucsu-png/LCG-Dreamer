"""Create isolated, parameter-only wave-guidance checkpoint variants."""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path

import torch


VARIANTS = {
    "selected_s40": (30.0, 0.002, 0.00, 0.40, 0.00, 1.00),
    "negative_bias30": (30.0, 0.002, 0.00, 0.40, -0.30, 1.00),
    "negative_bias60": (30.0, 0.002, 0.00, 0.40, -0.60, 1.00),
    "negative_bias90": (30.0, 0.002, 0.00, 0.40, -0.90, 1.00),
    "negative_bias120": (30.0, 0.002, 0.00, 0.40, -1.20, 1.00),
    "relative_gain02": (30.0, 0.002, 0.02, 0.40, 0.00, 1.00),
    "relative_gain04": (30.0, 0.002, 0.04, 0.40, 0.00, 1.00),
    "relative_gain06": (30.0, 0.002, 0.06, 0.40, 0.00, 1.00),
    "relative_gain08": (30.0, 0.002, 0.08, 0.40, 0.00, 1.00),
    "relative_gain10": (30.0, 0.002, 0.10, 0.40, 0.00, 1.00),
    "relative_gain12": (30.0, 0.002, 0.12, 0.40, 0.00, 1.00),
    "relative_gain14": (30.0, 0.002, 0.14, 0.40, 0.00, 1.00),
    "relative_gain16": (30.0, 0.002, 0.16, 0.40, 0.00, 1.00),
    "relative_gain12_speed35": (30.0, 0.002, 0.12, 0.35, 0.00, 1.00),
    "relative_gain12_speed45": (30.0, 0.002, 0.12, 0.45, 0.00, 1.00),
    "relative_gain12_speed50": (30.0, 0.002, 0.12, 0.50, 0.00, 1.00),
    "relative_gain12_speed55": (30.0, 0.002, 0.12, 0.55, 0.00, 1.00),
    "relative_gain12_gap00": (30.0, 0.000, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap01": (30.0, 0.001, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap03": (30.0, 0.003, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap04": (30.0, 0.004, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap05": (30.0, 0.005, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap06": (30.0, 0.006, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap08": (30.0, 0.008, 0.12, 0.40, 0.00, 1.00),
    "relative_gain12_gap10": (30.0, 0.010, 0.12, 0.40, 0.00, 1.00),
    "relative_gain08_gap05": (30.0, 0.005, 0.08, 0.40, 0.00, 1.00),
    "relative_gain10_gap05": (30.0, 0.005, 0.10, 0.40, 0.00, 1.00),
    "relative_gain14_gap05": (30.0, 0.005, 0.14, 0.40, 0.00, 1.00),
    "relative_gain16_gap05": (30.0, 0.005, 0.16, 0.40, 0.00, 1.00),
    "relative_gain18_gap05": (30.0, 0.005, 0.18, 0.40, 0.00, 1.00),
    "relative_gain20_gap05": (30.0, 0.005, 0.20, 0.40, 0.00, 1.00),
    "relative_gain24_gap05": (30.0, 0.005, 0.24, 0.40, 0.00, 1.00),
    "relative_gain28_gap05": (30.0, 0.005, 0.28, 0.40, 0.00, 1.00),
    "relative_gain32_gap05": (30.0, 0.005, 0.32, 0.40, 0.00, 1.00),
    "relative_gain36_gap05": (30.0, 0.005, 0.36, 0.40, 0.00, 1.00),
    "relative_gain40_gap05": (30.0, 0.005, 0.40, 0.40, 0.00, 1.00),
    "relative_gain50_gap05": (30.0, 0.005, 0.50, 0.40, 0.00, 1.00),
    "bias10": (30.0, 0.002, 0.00, 0.40, 0.10, 1.00),
    "bias20": (30.0, 0.002, 0.00, 0.40, 0.20, 1.00),
    "bias30": (30.0, 0.002, 0.00, 0.40, 0.30, 1.00),
    "bias40": (30.0, 0.002, 0.00, 0.40, 0.40, 1.00),
    "stress_t45_g10_r04_f05": (45.0, 0.010, 0.04, 0.40, 0.00, 1.00),
    "stress_t50_g12_r06_f08": (50.0, 0.012, 0.06, 0.40, 0.00, 1.00),
    "stress_t60_g15_r04_f05": (60.0, 0.015, 0.04, 0.40, 0.00, 1.00),
    "stress_t60_g15_r08_f05": (60.0, 0.015, 0.08, 0.40, 0.00, 1.00),
}

FRONT_ACCELERATION_GAIN_OVERRIDES = {
    "stress_t45_g10_r04_f05": 0.05,
    "stress_t50_g12_r06_f08": 0.08,
    "stress_t60_g15_r04_f05": 0.05,
    "stress_t60_g15_r08_f05": 0.05,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS))
    arguments = parser.parse_args()
    source = torch.load(arguments.checkpoint, map_location="cpu", weights_only=False)
    arguments.output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "source_checkpoint": str(arguments.checkpoint.resolve()),
        "reward_changed": False,
        "weight_parameters_changed": False,
        "variants": {},
    }
    selected_variants = arguments.variants or list(VARIANTS)
    for name in selected_variants:
        (
            target_gap,
            gap_gain,
            relative_gain,
            speed_gain,
            acceleration_bias,
            blend,
        ) = VARIANTS[name]
        front_acceleration_gain = FRONT_ACCELERATION_GAIN_OVERRIDES.get(name, 0.10)
        payload = copy.deepcopy(source)
        model_config = dict(payload["graph_model_config"])
        model_config.update(
            {
                "wave_guidance_target_gap": target_gap,
                "wave_guidance_gap_gain": gap_gain,
                "wave_guidance_relative_speed_gain": relative_gain,
                "wave_guidance_front_acceleration_gain": front_acceleration_gain,
                "wave_guidance_speed_gain": speed_gain,
                "wave_guidance_acceleration_bias": acceleration_bias,
                "wave_guidance_blend": blend,
            }
        )
        payload["graph_model_config"] = model_config
        payload["created_at"] = datetime.now(timezone.utc).isoformat()
        payload["complete"] = False
        payload["wave_guidance_variant"] = name
        output_path = arguments.output / f"graph_wave_{name}.pt"
        torch.save(payload, output_path)
        manifest["variants"][name] = {
            "checkpoint": str(output_path.resolve()),
            "target_gap_m": target_gap,
            "gap_gain": gap_gain,
            "relative_speed_gain": relative_gain,
            "front_acceleration_gain": front_acceleration_gain,
            "speed_gain": speed_gain,
            "acceleration_bias": acceleration_bias,
            "blend": blend,
        }
    (arguments.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
