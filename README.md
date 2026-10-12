# LCG-Dreamer: Six-Vehicle Platoon Control

Source code and usage instructions for the six-truck platoon-control study. The project implements longitudinal control with a latent communication graph world model, cooperative platoon lane changes, comparison controllers, and ablation experiments in SUMO.

This repository contains source code only. The unpublished manuscript, manuscript figures, datasets, trained weights, calibration assets, simulation maps, evaluation results, and logs are excluded. Running training or simulation requires preparing the corresponding assets locally.

## Project layout

All project code is under [`rl_architecture_experiment/`](rl_architecture_experiment/README.md).

| Directory | Purpose |
| --- | --- |
| `platoon_control/` | Graph world model, controllers, predictors, training, and evaluation |
| `platoon_sumo/` | Six-truck SUMO environment and scenario construction |
| `cooperative_lane_change/`, `mixed_traffic_lane_change/` | Cooperative platoon lane-change policies and evaluation |
| `classical_comparison/`, `predictive_comparison/` | Comparison controllers and experiment protocols |
| `dreamerv3_ablation/` | DreamerV3 ablation training and evaluation |
| `scripts/`, `configs/`, `experiments/` | Experiment utilities and configuration |
| `tests/` | Controller, model, and environment tests |
| `LCG_Dreamer_Final_Release/` | Final six-vehicle policy interface and adaptation scripts |
| `third_party/` | Vendored dependencies with their original licenses |

## Installation and usage

Use Python 3.11 and install the experiment environment from its directory:

```powershell
cd rl_architecture_experiment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-control.txt
python -m pip install -e . --no-deps
python -m platoon_control --help
```

Follow the [experiment guide](rl_architecture_experiment/README.md) for data preparation, training, and evaluation. For inference with locally supplied final checkpoints, use the [final-model guide](rl_architecture_experiment/LCG_Dreamer_Final_Release/README.md).

The experiment and final-model packages contain separate versions of the same Python modules. Run each from its own directory and use separate virtual environments when installing both.

## Repository scope

Only the six-vehicle platoon-control project and its supporting code are included. Separate traffic-management projects and three-vehicle deployment packages are outside this repository's scope.

The `.gitignore` excludes unpublished documents, generated assets, data, weights, and credentials. Review staged changes before subsequent uploads. Third-party components retain their own licenses; no additional license is granted for first-party code.
