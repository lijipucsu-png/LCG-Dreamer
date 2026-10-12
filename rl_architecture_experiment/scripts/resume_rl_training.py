from __future__ import annotations

import argparse

from platoon_control.rl import RLConfig, train_rl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--predictor-directory", default="artifacts/checkpoints")
    parser.add_argument("--device", default="cuda")
    arguments = parser.parse_args()

    train_rl(
        RLConfig(
            device=arguments.device,
            checkpoint_dir=arguments.checkpoint_dir,
        ),
        predictor_directory=arguments.predictor_directory,
        resume=arguments.checkpoint,
    )


if __name__ == "__main__":
    main()
