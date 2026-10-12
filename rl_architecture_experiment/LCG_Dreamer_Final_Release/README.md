# LCG-Dreamer Final Six-Vehicle Policy

This directory contains the six-vehicle policy interface, controller and environment source, cooperative lane-change head, and scripts for inference and new-scenario adaptation.

This GitHub copy contains source code only. The weights, calibration assets, SUMO maps, release manifest, and validation results referenced below are not included. Supply the matching assets locally before running inference or release verification.

## Required local assets

| Asset | Expected path |
| --- | --- |
| Longitudinal graph controller | `weights/graph_longitudinal_final.pt` |
| Matching cooperative lane-change head | `weights/graph_lane_change_final.pt` |
| CPIDLM predictor | `weights/predictors/cpidlm_best.pt` |
| Neighbor trajectory predictor | `weights/predictors/trajectory_best.pt` |
| Leader-front predictor | `weights/predictors/leader_front_best.pt` |
| Traffic-wave predictor | `weights/predictors/wave_best.pt` |
| Traffic calibration | `weights/calibration/calibrated_parameters.json` |
| Calibrated vehicle types | `weights/calibration/ngsim_vtypes.add.xml` |
| SUMO network and routes | Files expected under `platoon_sumo/scenario/` |

The longitudinal and lateral checkpoints must be a matching pair. The runtime checks their compatibility. `RELEASE_MANIFEST.json`, when supplied with the complete local release, records hashes used by the verification script.

## Installation

From this directory, create a separate environment:

```powershell
conda env create -f environment.yml
conda activate lcg-dreamer-release
python -m pip install -e . --no-deps
sumo --version
python -c "import torch; print(torch.cuda.is_available())"
```

The original release environment uses Windows, Python 3.11, SUMO 1.27.1, and CUDA 12.1. See `environment.yml` and `requirements.txt` for dependency specifications. CPU inference is supported; `--device auto` selects an available GPU first.

## Control interface

The policy controls six fixed vehicle IDs: `leader`, then `follower_1` through `follower_5`. At each 0.1-second control step, it receives a dictionary of observations and returns one action per vehicle:

```python
{
    "leader": {"accel": np.array([a0], dtype=np.float32), "lane_change": 0},
    "follower_1": {"accel": np.array([a1], dtype=np.float32), "lane_change": 0},
    # The same structure applies to follower_2 through follower_5.
}
```

The graph controller generates longitudinal actions. When lane changes are enabled, the lateral head also supplies cooperative lane-change decisions and dynamic target gaps. SUMO/TraCI applies the actions and vehicle constraints.

```python
from final_model import load_final_policy

policy = load_final_policy(device="cuda", enable_lane_change=True)
policy.reset()
actions = policy.act(observations)
observations, rewards, terms, truncs, infos = env.step(actions)
policy.record_applied(infos)
```

Call `record_applied(infos)` after every environment step. The recurrent state and communication-delay compensation need the acceleration actually applied by the environment.

## Verification and inference

After supplying the full local release assets and manifest:

```powershell
python scripts/verify_release.py --device cuda
python scripts/verify_release.py --device cuda --sumo-smoke-steps 30
```

Verification checks asset hashes, controller and predictor loading, and optionally a SUMO rollout. Outputs are written to `outputs/verification/`. The source-only checkout cannot pass full release verification without the omitted assets.

Run the desired scenario:

```powershell
python scripts/run_final_model.py --scenario following --device cuda
python scripts/run_final_model.py --scenario full --device cuda --output outputs/full_test
python scripts/run_final_model.py --scenario lane-change --device cuda --output outputs/lane_change_test
python scripts/run_final_model.py --scenario lane-change --device cuda --gui
```

The full scenario combines communication delays, cut-ins, traffic waves, V2I information, and background traffic. The lane-change scenario adds repeated cooperative lane changes. Each run writes a summary and per-step trajectories, including vehicle motion, gaps, jerk, collisions, delay events, and lane-change completion.

## Integration with another simulator

Preserve the following interface requirements:

1. Keep the six vehicle IDs and their ordering unchanged.
2. Match `PlatoonParallelEnv` observation keys and shapes, including `ego`, `platoon_state`, `local_predecessor`, `front_history`, `neighbor_history`, `communication`, `road_profile`, `v2i_flow`, and `lane_mission`.
3. Keep the 0.1-second control period. A different period requires recalibrating delays, jerk limits, prediction horizons, and recurrent transitions.
4. Preserve the nominal acceleration range of `[-3.0, 1.2] m/s^2`; emergency braking is handled by the safety layer and environment constraints.
5. Distinguish requested from applied acceleration and report the latter in `infos[agent]["applied_acceleration"]`.
6. Construct graph adjacency, V2V packet age, and neighboring trajectories from causal observations.

When replacing a SUMO map, update nodes, edges, connections, routes, and scenario configuration together. The environment depends on mainline, ramp, merging-zone, and route IDs; changing only the network file is insufficient.

## New-scenario adaptation

Keep the final checkpoints in `weights/` and write adaptation outputs under `outputs/`. The adaptation entry point restores network and optimizer state, clears the previous replay buffer, and starts a new curriculum.

Copy and customize `examples/new_scene_curriculum.py`, then inspect the setup with a dry run:

```powershell
python scripts/train_new_scene.py --scenario-file examples/new_scene_curriculum.py --output outputs/new_scene_dry_run --additional-steps 50000 --device cuda --dry-run
```

After reviewing the output, start training:

```powershell
python scripts/train_new_scene.py --scenario-file examples/new_scene_curriculum.py --output outputs/new_scene_01 --additional-steps 50000 --device cuda
```

For changes to traffic demand, speed distributions, delay probability, or disturbance strength, preserve observation and action spaces. For geometry changes, validate route connectivity and vehicle insertion with a rule-based controller before reinforcement learning.

Suggested adaptation budgets are 20,000-50,000 steps for traffic-distribution changes, 50,000-100,000 for new geometry or lane-change locations, and 100,000-200,000 for changed vehicle dynamics or sensor statistics. These are starting points, not convergence guarantees.

Build the curriculum gradually: basic following and restart behavior; low-density traffic with a single disturbance; communication delay, cut-ins, and traffic waves; then curves, ramps, and repeated lane changes. Use a predefined multi-seed validation set for checkpoint selection and keep the final test set separate.

The lateral head depends on the semantics of `lane_mission`. If the lane count, direction encoding, or conflict types change, first freeze the longitudinal checkpoint and adapt the lateral head. Evaluate completion, timing, gaps, TTC, platoon speed differences, jerk, and effects on background traffic before joint fine-tuning.

## Troubleshooting

- Missing predictors: supply all predictor assets and use `final_model.load_final_policy()` to resolve their paths.
- SUMO not found: add its `bin` directory to `PATH` and configure `SUMO_HOME` if required.
- Insufficient GPU memory: use `--device cpu` for inference or reduce concurrent jobs.
- Immediate gridlock on a new map: check routes, insertion positions, and background demand before training.
- Missing replay during adaptation: use `scripts/train_new_scene.py`, which initializes new-scenario replay while restoring network and optimizer state.

The code targets the documented SUMO configuration. Deployment to another simulator or physical vehicles requires separate validation of units, sensing, actuation, timing, and fallback behavior.
