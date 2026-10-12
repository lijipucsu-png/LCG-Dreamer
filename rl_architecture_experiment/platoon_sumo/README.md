> Source-only GitHub export: datasets, maps, pretrained weights, calibration assets, evaluation results and release manifests are not included. Prepare these locally before running commands that require them.

# SUMO Platoon Environment

This package contains a reproducible 3.5 km SUMO motorway and a six-agent
PettingZoo Parallel environment. It is intentionally independent from the
existing `CPIDLM-MORE.py` training program.

Build the network and run a headless smoke test:

```powershell
.\.venv\Scripts\python.exe -m platoon_sumo.build_network
.\.venv\Scripts\python.exe -m platoon_sumo.run_demo --seed 42
```

Watch the scenario in SUMO GUI:

```powershell
.\.venv\Scripts\python.exe -m platoon_sumo.run_demo --gui --seed 42
```

RL code should instantiate `PlatoonParallelEnv`, call `reset(seed=...)`, and
send one action per live agent:

```python
actions = {
    agent: {
        "accel": np.array([0.0], dtype=np.float32),
        "lane_change": 0,
    }
    for agent in env.agents
}
observations, rewards, terminations, truncations, infos = env.step(actions)
```

The leader action retains the acceleration/lane-change API. Follower action
spaces admit only lane command zero, so the learned platoon follower policy is
strictly longitudinal. Lateral predictions are observations for human cut-in
intent, not follower control outputs. All physical
longitudinal values in observations use route station in metres rather than
the planar x-coordinate of the S-curve. The leader plus five followers are
12 m articulated trucks in the middle mainline lane with a 12 m initial net
gap. A predicted ramp cut-in may reduce the temporary target to 8 m, while an
independent collision layer retains a 5 m hard safety threshold.

The designated cut-in vehicle never bypasses gap acceptance. It changes lanes
only when both the explicit front/rear clearance check and SUMO's own safety
check pass. An episode records `cutin_blocked` when the platoon safely closes
the target gap, or `cutin_success` when a valid gap remains available.

Each episode samples zero to eight dense same-lane vehicles ahead of the
leader with randomized 14-22 m net gaps. Calibrated NGSIM background traffic
is preloaded across all three lanes and the full motorway, then augmented by
ongoing mainline and ramp inflows. Ordinary background vehicles retain
autonomous LC2013 lane changing after merging. Ramp-origin vehicles use an
explicit front/rear dynamic-gap check while they occupy the acceleration lane.
Only the sampled near-front chain is locked to the platoon's lane.

The 600-1400 m mountain section uses a densely sampled, curvature-continuous
left/right sequence. The motorway remains an independent, straight three-lane
road through the merge area. At 1600 m the ramp enters a 500 m acceleration
and merge lane. Only its first 116.7 m is physically separated by a solid-line
section; the remaining 383.3 m permits discretionary lane changes, while the
last 150 m retains the smooth taper into the right mainline lane.

V2V impairment is event based rather than permanently active. By default,
each packet has a 15% probability of receiving a uniformly sampled 100-200 ms
FIFO delay; other packets receive no added impairment delay. Episode options
may lower the probability or narrow the range, but delayed packets are always
bounded to 100-200 ms.
