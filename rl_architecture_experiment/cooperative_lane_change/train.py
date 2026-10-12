"""Focused BC-initialized PPO transfer for cooperative multi-lane changes."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .model import CooperativeLaneActorCritic
from .policy import CooperativeLanePolicy, create_base_policy
from .protocol import METHODS, LaneChangeProtocol


def _file_sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _episode(
    policy: CooperativeLanePolicy,
    protocol: LaneChangeProtocol,
    episode_seed: int,
    training: bool,
    collect_expert: bool = False,
) -> tuple[list[dict[str, Tensor]], dict[str, Any]]:
    env = PlatoonParallelEnv(max_time=protocol.max_time)
    observations, _ = env.reset(
        seed=episode_seed,
        options=protocol.scenario_options(episode_seed, training=training),
    )
    policy.reset()
    transitions: list[dict[str, Tensor]] = []
    minimum_gap = float("inf")
    maximum_gap = 0.0
    maximum_speed_spread = 0.0
    lane_reward_sum = 0.0
    team_reward_sum = 0.0
    final_info: dict[str, Any] = {}
    try:
        while env.agents:
            actions = policy.act(observations)
            trace = policy.last_trace
            if trace is None:
                raise RuntimeError("cooperative policy did not emit a trace")
            next_observations, rewards, terminations, truncations, infos = env.step(
                actions
            )
            policy.record_applied(infos)
            final_info = infos["leader"]
            component = final_info["reward_components"]
            lane_reward = float(component["lane_change_team_reward"])
            team_reward = float(np.mean(list(rewards.values())))
            reward = lane_reward + 0.05 * team_reward
            done = bool(
                any(terminations.values()) or any(truncations.values())
            )
            if done and not bool(final_info.get("success", False)):
                reward -= 20.0
            transitions.append(
                {
                    "features": trace.features,
                    "change_mask": trace.change_mask,
                    "gap_active": trace.gap_active,
                    "change": trace.change,
                    "gap_raw": trace.gap_raw,
                    "old_log_prob": trace.log_prob,
                    "old_value": trace.value,
                    "reward": torch.tensor(reward, dtype=torch.float32),
                    "done": torch.tensor(float(done), dtype=torch.float32),
                }
            )
            platoon = np.asarray(
                observations["leader"]["platoon_state"], np.float32
            )
            valid_gaps = platoon[1:, 4]
            valid_gaps = valid_gaps[valid_gaps > 0.0]
            if len(valid_gaps):
                minimum_gap = min(minimum_gap, float(np.min(valid_gaps)))
                maximum_gap = max(maximum_gap, float(np.max(valid_gaps)))
            speeds = np.asarray(
                [observations[agent]["ego"][1] for agent in PLATOON_IDS]
            )
            maximum_speed_spread = max(
                maximum_speed_spread, float(np.ptp(speeds))
            )
            lane_reward_sum += lane_reward
            team_reward_sum += team_reward
            observations = next_observations
        metrics = {
            "seed": episode_seed,
            "success": bool(final_info.get("success", False)),
            "collision": bool(final_info.get("collision", False)),
            "maneuvers_completed": int(
                final_info.get("lane_change_maneuvers_completed", 0)
            ),
            "episode_time_s": float(final_info.get("time", env.sim_time)),
            "leader_station_m": float(
                final_info.get("leader_station", 0.0)
            ),
            "minimum_gap_m": (
                float(minimum_gap) if np.isfinite(minimum_gap) else 0.0
            ),
            "maximum_gap_m": float(maximum_gap),
            "maximum_speed_spread_mps": float(maximum_speed_spread),
            "dual_lane_time_s": float(
                final_info.get("lane_change_dual_lane_time", 0.0)
            ),
            "lane_commands": int(
                final_info.get("lane_change_total_commands", 0)
            ),
            "lane_reward_sum": float(lane_reward_sum),
            "team_reward_sum": float(team_reward_sum),
            "steps": len(transitions),
            "expert": bool(collect_expert),
        }
        return transitions, metrics
    finally:
        env.close()


def _imitation_update(
    model: CooperativeLaneActorCritic,
    optimizer: torch.optim.Optimizer,
    demonstrations: list[dict[str, Tensor]],
    protocol: LaneChangeProtocol,
    device: torch.device,
) -> dict[str, float]:
    features = torch.stack([item["features"] for item in demonstrations]).to(device)
    masks = torch.stack([item["change_mask"] for item in demonstrations]).to(device)
    gap_active = torch.stack([item["gap_active"] for item in demonstrations]).to(device)
    changes = torch.stack([item["change"] for item in demonstrations]).to(device)
    gap_raw = torch.stack([item["gap_raw"] for item in demonstrations]).to(device)
    batch_size = min(protocol.minibatch_size, len(demonstrations))
    losses: list[float] = []
    model.train()
    for _ in range(protocol.imitation_epochs):
        order = torch.randperm(len(demonstrations), device=device)
        for start in range(0, len(demonstrations), batch_size):
            index = order[start : start + batch_size]
            logits, gap_mean, _ = model(features[index])
            active_count = masks[index].float().sum().clamp_min(1.0)
            lane_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits, changes[index], reduction="none"
                )
                * masks[index].float()
            ).sum() / active_count
            active_gap = gap_active[index].float()[:, None]
            gap_loss = (
                (gap_mean - gap_raw[index]).pow(2) * active_gap
            ).sum() / (active_gap.sum().clamp_min(1.0) * gap_mean.shape[-1])
            loss = lane_loss + 0.5 * gap_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"imitation_loss": float(np.mean(losses)) if losses else 0.0}


def _gae(
    transitions: list[dict[str, Tensor]], protocol: LaneChangeProtocol
) -> tuple[Tensor, Tensor]:
    rewards = torch.stack([item["reward"] for item in transitions])
    values = torch.stack([item["old_value"] for item in transitions])
    dones = torch.stack([item["done"] for item in transitions])
    advantages = torch.zeros_like(rewards)
    gae = torch.tensor(0.0)
    next_value = torch.tensor(0.0)
    for index in reversed(range(len(transitions))):
        continuation = 1.0 - dones[index]
        delta = rewards[index] + protocol.discount * next_value * continuation - values[index]
        gae = delta + protocol.discount * protocol.gae_lambda * continuation * gae
        advantages[index] = gae
        next_value = values[index]
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / advantages.std().clamp_min(1e-6)
    return advantages, returns


def _ppo_update(
    model: CooperativeLaneActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Tensor]],
    protocol: LaneChangeProtocol,
    device: torch.device,
) -> dict[str, float]:
    advantages, returns = _gae(transitions, protocol)
    batch = {
        key: torch.stack([item[key] for item in transitions]).to(device)
        for key in (
            "features",
            "change_mask",
            "gap_active",
            "change",
            "gap_raw",
            "old_log_prob",
        )
    }
    advantages = advantages.to(device)
    returns = returns.to(device)
    batch_size = min(protocol.minibatch_size, len(transitions))
    losses: list[float] = []
    kls: list[float] = []
    clip_fractions: list[float] = []
    model.train()
    for _ in range(protocol.ppo_epochs):
        order = torch.randperm(len(transitions), device=device)
        for start in range(0, len(transitions), batch_size):
            index = order[start : start + batch_size]
            log_prob, entropy, value = model.evaluate_actions(
                batch["features"][index],
                batch["change_mask"][index],
                batch["gap_active"][index],
                batch["change"][index],
                batch["gap_raw"][index],
            )
            log_ratio = log_prob - batch["old_log_prob"][index]
            ratio = log_ratio.exp()
            unclipped = ratio * advantages[index]
            clipped = ratio.clamp(
                1.0 - protocol.clip_ratio, 1.0 + protocol.clip_ratio
            ) * advantages[index]
            policy_loss = -torch.minimum(unclipped, clipped).mean()
            value_loss = 0.5 * (value - returns[index]).pow(2).mean()
            entropy_bonus = entropy.mean()
            loss = (
                policy_loss
                + protocol.value_weight * value_loss
                - protocol.entropy_weight * entropy_bonus
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            kls.append(float((-log_ratio).mean().detach().cpu()))
            clip_fractions.append(
                float((ratio.sub(1.0).abs() > protocol.clip_ratio).float().mean().cpu())
            )
    return {
        "ppo_loss": float(np.mean(losses)),
        "approx_kl": float(np.mean(kls)),
        "clip_fraction": float(np.mean(clip_fractions)),
    }


def train(
    method: str,
    protocol: LaneChangeProtocol,
    device_name: str,
) -> Path:
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    device = torch.device(device_name)
    torch.manual_seed(protocol.seed + METHODS.index(method) * 1000)
    np.random.seed(protocol.seed + METHODS.index(method) * 1000)
    base_policy = create_base_policy(method, protocol, device_name)
    model = CooperativeLaneActorCritic().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=protocol.learning_rate)
    expert_policy = CooperativeLanePolicy(
        base_policy, model, device, deterministic=True, expert=True
    )
    demonstrations: list[dict[str, Tensor]] = []
    for episode in range(protocol.imitation_episodes):
        seed = protocol.seed + METHODS.index(method) * 10_000 + episode
        transitions, metrics = _episode(
            expert_policy, protocol, seed, training=True, collect_expert=True
        )
        demonstrations.extend(transitions)
        print(json.dumps({"phase": "imitation", "method": method, **metrics}), flush=True)
    imitation_metrics = _imitation_update(
        model, optimizer, demonstrations, protocol, device
    )
    policy = CooperativeLanePolicy(
        base_policy, model, device, deterministic=False, expert=False
    )
    output = protocol.adapter_checkpoint(method)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_score = -float("inf")
    best_metrics: dict[str, Any] = {}
    recent_metrics: list[dict[str, Any]] = []
    for episode in range(protocol.training_episodes):
        seed = (
            protocol.seed
            + METHODS.index(method) * 10_000
            + protocol.imitation_episodes
            + episode
        )
        transitions, metrics = _episode(policy, protocol, seed, training=True)
        update_metrics = _ppo_update(
            model, optimizer, transitions, protocol, device
        )
        score = (
            1000.0 * float(metrics["success"])
            + 100.0 * metrics["maneuvers_completed"]
            + metrics["lane_reward_sum"]
            - 20.0 * float(metrics["collision"])
            - metrics["episode_time_s"]
        )
        row = {
            "phase": "ppo",
            "method": method,
            "episode": episode + 1,
            **metrics,
            **imitation_metrics,
            **update_metrics,
            "score": score,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        recent_metrics.append(row)
        if score > best_score:
            best_score = score
            best_metrics = row
            checkpoint = protocol.base_checkpoint(method)
            torch.save(
                {
                    "method": method,
                    "adapter_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "protocol": protocol.to_dict(),
                    "base_checkpoint": None if checkpoint is None else str(checkpoint),
                    "base_checkpoint_sha256": _file_sha256(checkpoint),
                    "best_score": best_score,
                    "best_metrics": best_metrics,
                    "base_controller_frozen": True,
                    "trained_parameters": sum(
                        parameter.numel() for parameter in model.parameters()
                    ),
                },
                output,
            )
        if episode + 1 >= 8:
            window = recent_metrics[-5:]
            stable = (
                len(window) == 5
                and all(item["success"] for item in window)
                and not any(item["collision"] for item in window)
                and float(
                    np.mean(
                        [item["maximum_speed_spread_mps"] for item in window]
                    )
                )
                <= 1.5
                and float(
                    np.max([item["maximum_gap_m"] for item in window])
                )
                <= 15.5
                and float(np.std([item["episode_time_s"] for item in window]))
                <= 3.0
            )
            if stable:
                print(
                    json.dumps(
                        {
                            "phase": "early_stop",
                            "method": method,
                            "episode": episode + 1,
                            "reason": "five stable safe successful episodes",
                        }
                    ),
                    flush=True,
                )
                break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--imitation-episodes", type=int)
    arguments = parser.parse_args()
    protocol = LaneChangeProtocol()
    if arguments.episodes is not None:
        protocol = replace(protocol, training_episodes=arguments.episodes)
    if arguments.imitation_episodes is not None:
        protocol = replace(
            protocol, imitation_episodes=arguments.imitation_episodes
        )
    output = train(arguments.method, protocol, arguments.device)
    print(json.dumps({"complete": True, "checkpoint": str(output)}))


if __name__ == "__main__":
    main()
