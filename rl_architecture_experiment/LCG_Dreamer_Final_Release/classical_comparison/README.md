> Source-only GitHub export: datasets, maps, pretrained weights, calibration assets, evaluation results and release manifests are not included. Prepare these locally before running commands that require them.

# Classical Controller Comparison

This directory is isolated from the trained Graph controller. It implements
centralized DDPG, PPO, and SAC baselines with a shared six-acceleration action
space. The comparison protocol freezes:

- the complete observable Graph input (flattened without auxiliary labels);
- previous applied acceleration, so actuator dynamics remain Markovian;
- acceleration bounds of -3.0 to 1.2 m/s2;
- the existing environment reward and existing Graph training reward shaping;
- the same frozen CPIDLM reward reference for every controller;
- curriculum boundaries, training steps, training seed, evaluation scenarios,
  reset seeds, and episode count.

The team scalar used by classical algorithms is the arithmetic mean of the six
per-vehicle rewards. This is the same reduction used by the Graph actor's
node-wise imagined objective. No result-dependent reward tuning is allowed.

Train one baseline:

```powershell
python -m classical_comparison.train ddpg --output artifacts/classical_comparison/ddpg
```

Evaluate one completed controller:

```powershell
python -m classical_comparison.evaluate ddpg artifacts/classical_comparison/ddpg/complete.pt --output artifacts/classical_comparison/evaluation/ddpg
```

## Two-layer evaluation

The full 3.5 km route remains the system-level survival test. A second,
fixed-horizon follower-response test prevents an early leader-policy failure
from making every longitudinal metric undefined. It uses 20 controller-blind
NGSIM test-split braking waves, an identical externally tracked leader
trajectory, paired reset seeds, and the unchanged trained policies:

```powershell
python -m classical_comparison.fixed_horizon ddpg artifacts/classical_comparison/training/ddpg/complete.pt `
  --output artifacts/classical_comparison/fixed_horizon/ddpg
```

The external leader is an evaluation intervention only. It does not alter a
checkpoint, training reward, follower action, actuator limit, or safety shield.
Energy is interpreted only beside formation-service validity and tail mobility;
a controller that saves energy by stopping its followers is a mobility failure.
