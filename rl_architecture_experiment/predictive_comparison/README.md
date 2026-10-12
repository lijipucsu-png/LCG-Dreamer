> Source-only GitHub export: datasets, maps, pretrained weights, calibration assets, evaluation results and release manifests are not included. Prepare these locally before running commands that require them.

# Prediction-Equated Comparison

This package is isolated from `classical_comparison` and does not overwrite any
legacy controller, checkpoint, evaluation row, or manuscript result.

## Compared controllers

- `graph`: the completed MA-Latent-Graph-Dreamer checkpoint.
- `cacc`: the unchanged deterministic CACC controller, with no learned predictor.
- `ddpg_p`, `ppo_p`, `sac_p`: prediction-enabled classical RL controllers.
- `nc_kg_sac_p`: task-adapted knowledge-guided switching SAC from Wang et al.
  (2025), DOI `10.1038/s41467-025-62597-x`.

Every learned baseline receives the same frozen Graph-RSSM checkpoint and the
same frozen ST-GNN checkpoint. The policy input includes online predictions of
predecessor motion, physical state, cut-in probability, surrounding traffic,
and the full 30-step trajectory of the currently highest-risk cut-in vehicle.
No policy receives simulator future truth.

The NC baseline additionally preserves the source method's kinematic-wave and
Newell aggregate features, previous control instructions, SAC policy, and model
switching for 0/100/200 ms delay modes. Its CV-HVs-CV segment is explicitly
adapted to this study because all six platoon trucks are controlled vehicles.

## Fairness invariants

- One reward implementation and one reward source fingerprint.
- No controller-specific reward tuning.
- Identical action limits, stage-7 training scenarios, evaluation seeds, and
  episode horizon.
- Identical frozen prediction checkpoint hashes for all learned baselines.
- CACC remains unchanged and is identified as a non-predictive traditional
  baseline.
- Existing capable policies initialize the non-predictive input columns; all
  newly added prediction columns start at zero. Each method is then fine-tuned
  until its old plus new simulator interaction count equals Graph's 301,144
  steps. Offline initialization samples are not counted as simulator steps.

Run the complete resumable study with:

```powershell
python -m predictive_comparison.run_study `
  --root artifacts/predictive_comparison_v2 `
  --steps 301144 --episodes 20 --device cuda
```
