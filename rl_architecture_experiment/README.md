# Six-Truck Platoon-Control Experiments

This project contains the LCG-Dreamer graph world model, the six-truck SUMO environment, cooperative lane-change policies, baseline controllers, and ablation experiments. Run the commands below from this directory.

This is a source-only repository. Datasets, road-network assets, calibration files, pretrained checkpoints, and experiment results must be prepared locally. No manuscript or manuscript figures are included.

## Environment

Python 3.11 is recommended. Dependency versions are defined in `requirements-control.txt` and `environment-control.yml`; the SUMO Python packages are pinned to 1.27.1.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-control.txt
python -m pip install -e . --no-deps
python -m platoon_control --help
sumo --version
```

Use the dependency files to select compatible PyTorch and CUDA installations. SUMO executables must be available on `PATH`; some setups also require `SUMO_HOME`.

## Components

- `platoon_control/models/graph_dreamer.py` implements the graph world model.
- `platoon_control/graph_agents.py` and `graph_rl.py` provide controller integration and training.
- `platoon_sumo/` supplies the PettingZoo/Gymnasium environment and SUMO scenario code.
- `cooperative_lane_change/` and `mixed_traffic_lane_change/` implement platoon lane-change policies.
- `classical_comparison/` and `predictive_comparison/` contain comparison protocols and controllers.
- `dreamerv3_ablation/` contains the short-budget DreamerV3 ablation protocol.
- `LCG_Dreamer_Final_Release/` provides the final policy interface; see its [usage guide](LCG_Dreamer_Final_Release/README.md).

The older `train-rl` path implements a DreamerV3 leader with shared longitudinal SAC followers. The graph-controller training path is `train-graph-rl`. These are distinct experiment implementations.

## Local inputs and generated assets

Prepare the CPIDLM source dataset (`data_30.mat`) and NGSIM trajectory data locally. Check each command's `--help` for input-path overrides. The original setup uses `NGSIM/trajectories-0500-0515.csv`.

The supervised pipeline uses chronological 60/20/20 splits. CPIDLM splits retain a 50-sample separation; NGSIM splits retain a 5-second separation. NGSIM trajectories are converted to SI units and smoothed with an 11-step, third-order Savitzky-Golay filter.

Typical generated paths are:

```text
artifacts/ngsim/clean_tracks.npz
artifacts/ngsim/manifest.json
artifacts/calibration/calibrated_parameters.json
artifacts/calibration/ngsim_vtypes.add.xml
artifacts/checkpoints/{cpidlm,trajectory,leader_front,wave}_best.pt
artifacts/graph_rl_seed42/
artifacts/evaluation/
```

The nominal scenario uses six 25-tonne trucks on a 3.5 km highway with curves, a merging ramp, and a traffic-wave section. Supply the corresponding road-network and route assets expected by `platoon_sumo/scenario/` before running SUMO. Scenario code and configuration are included; generated map files are excluded.

To generate the nominal network locally with SUMO installed:

```powershell
python -m platoon_sumo.build_network
```

## Data preparation and predictor training

```powershell
python -m platoon_control prepare-ngsim
python -m platoon_control calibrate-sumo --trials 400 --seeds 3 --duration 90
python -m platoon_control train-supervised --epochs 100 --patience 12 --device cuda
python -m platoon_control evaluate-supervised --device cuda
```

Inspect `artifacts/checkpoints/predictor_test_report.json` before reinforcement learning. Predictor evaluation compares CPIDLM with packet-age constant-acceleration extrapolation and trajectory predictors with constant-velocity extrapolation. The traffic-wave predictor is evaluated against a zero-acceleration reference and must generate the intended deceleration behavior.

## Graph-controller training

Inspect configuration options before launching a long training job:

```powershell
python -m platoon_control train-graph-rl --help
python -m platoon_control train-graph-rl --device cuda --output artifacts/graph_rl_seed42
python -m platoon_control validate-graph-foundation --help
python -m platoon_control evaluate-graph-auxiliary --help
```

Training requires the prepared predictors, calibration, and SUMO assets. Use command-line options to select checkpoints, variants, budgets, and output directories. Study-management scripts under `scripts/` may reference local checkpoints and prior runs; inspect their path settings before use.

For the earlier SAC-follower baseline:

```powershell
python -m platoon_control train-rl --max-steps 1000000 --device cuda
python -m platoon_control evaluate artifacts/rl/rl_best.pt --device cuda
```

Comparison and ablation implementations maintain their own protocols. Their entry points include:

```powershell
python -m dreamerv3_ablation.run_study --help
python -m cooperative_lane_change.run_study --help
```

Reproducing a particular experiment requires matching local data, checkpoints, seeds, and evaluation configuration.

## Inference and tests

Use the [final-model interface](LCG_Dreamer_Final_Release/README.md) to load the final longitudinal and lateral checkpoints together. The interface controls `leader` and `follower_1` through `follower_5` at a 0.1-second control period.

```powershell
python -m pytest tests -q
```

Model-loading, dataset, and SUMO integration tests require their corresponding local assets. Unit tests that exercise model structure or controller logic can be selected independently.

Vendored DreamerV3 RSSM code is under `third_party/r2dreamer`; its original license is retained.
