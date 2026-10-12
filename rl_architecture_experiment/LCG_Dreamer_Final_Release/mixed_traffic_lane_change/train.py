"""Equal-budget focused training for mixed-traffic repeated lane changes."""

from __future__ import annotations

import argparse
from dataclasses import replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from cooperative_lane_change.model import CooperativeLaneActorCritic
from platoon_sumo.env import PLATOON_IDS, PlatoonParallelEnv

from .algorithms import (
    NativeLaneAlgorithm,
    create_native_algorithm,
    restore_native_algorithm,
)
from .policy import (
    GraphMixedLanePolicy,
    MixedLaneTrace,
    NativeMixedLanePolicy,
    create_base_policy,
    graph_adapter,
)
from .protocol import LEARNED_METHODS, METHODS, MixedLaneProtocol


@lru_cache(maxsize=None)
def _sha256_cached(
    resolved_path: str, size: int, modified_ns: int
) -> str:
    digest = hashlib.sha256()
    with Path(resolved_path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    statistics = path.stat()
    return _sha256_cached(
        str(path.resolve()), statistics.st_size, statistics.st_mtime_ns
    )


def _episode(
    policy: GraphMixedLanePolicy | NativeMixedLanePolicy,
    protocol: MixedLaneProtocol,
    seed: int,
    training: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = PlatoonParallelEnv(max_time=protocol.max_time)
    observations, _ = env.reset(
        seed=seed, options=protocol.scenario_options(seed, training)
    )
    policy.reset()
    transitions: list[dict[str, Any]] = []
    minimum_gap = float("inf")
    maximum_gap = 0.0
    maximum_speed_difference = 0.0
    lane_reward_sum = 0.0
    team_reward_sum = 0.0
    rejected_commands = 0
    final_info: dict[str, Any] = {}
    try:
        while env.agents:
            actions = policy.act(observations)
            trace = policy.last_trace
            if trace is None:
                raise RuntimeError("learned mixed-lane policy emitted no trace")
            next_observations, rewards, terminations, truncations, infos = env.step(
                actions
            )
            policy.record_applied(infos)
            final_info = infos["leader"]
            component = final_info["reward_components"]
            lane_reward = float(component["lane_change_team_reward"])
            team_reward = float(np.mean(list(rewards.values())))
            reward = lane_reward + 0.05 * team_reward
            done = bool(any(terminations.values()) or any(truncations.values()))
            if done and not bool(final_info.get("success", False)):
                reward -= 20.0
            transitions.append(
                {
                    "features": trace.features,
                    "state": trace.state,
                    "change_mask": trace.change_mask,
                    "gap_active": trace.gap_active,
                    "change": trace.change,
                    "gap_raw": trace.gap_raw,
                    "action_unit": trace.action_unit,
                    "expert_target": np.concatenate(
                        (
                            np.where(
                                trace.change_mask.detach().cpu().numpy(),
                                0.9,
                                -0.9,
                            ),
                            np.full(5, -0.6, dtype=np.float32),
                        )
                    ).astype(np.float32),
                    "log_prob": trace.log_prob,
                    "value": trace.value,
                    "reward": float(reward),
                    "done": done,
                }
            )
            platoon = np.asarray(
                observations["leader"]["platoon_state"], np.float32
            )
            gaps = platoon[1:, 4]
            gaps = gaps[gaps > 0.0]
            if len(gaps):
                minimum_gap = min(minimum_gap, float(np.min(gaps)))
                maximum_gap = max(maximum_gap, float(np.max(gaps)))
            speeds = np.asarray(
                [observations[agent]["ego"][1] for agent in PLATOON_IDS],
                np.float64,
            )
            maximum_speed_difference = max(
                maximum_speed_difference, float(np.ptp(speeds))
            )
            rejected_commands += sum(
                int(bool(infos[agent].get("lane_change_rejected", False)))
                for agent in PLATOON_IDS
            )
            lane_reward_sum += lane_reward
            team_reward_sum += team_reward
            observations = next_observations
        for index, transition in enumerate(transitions):
            transition["next_state"] = (
                transitions[index + 1]["state"].copy()
                if index + 1 < len(transitions)
                else np.zeros_like(transition["state"])
            )
        metrics = {
            "seed": seed,
            "success": bool(final_info.get("success", False)),
            "collision": bool(final_info.get("collision", False)),
            "failure": bool(final_info.get("failure", False)),
            "maneuvers_completed": int(
                final_info.get("lane_change_maneuvers_completed", 0)
            ),
            "time_s": float(final_info.get("time", env.sim_time)),
            "leader_station_m": float(final_info.get("leader_station", 0.0)),
            "minimum_gap_m": float(
                minimum_gap if np.isfinite(minimum_gap) else 0.0
            ),
            "maximum_gap_m": float(maximum_gap),
            "maximum_speed_difference_mps": float(maximum_speed_difference),
            "traffic_delay_s": float(
                final_info.get("lane_change_traffic_delay_s", 0.0)
            ),
            "traffic_braking_integral": float(
                final_info.get("lane_change_traffic_braking_integral", 0.0)
            ),
            "traffic_hard_brake_events": int(
                final_info.get("lane_change_traffic_hard_brake_events", 0)
            ),
            "traffic_max_deceleration_mps2": float(
                final_info.get("lane_change_traffic_max_deceleration", 0.0)
            ),
            "traffic_spawned_count": int(
                final_info.get("lane_change_traffic_spawned_count", 0)
            ),
            "rejected_lane_commands": int(rejected_commands),
            "lane_reward_sum": float(lane_reward_sum),
            "team_reward_sum": float(team_reward_sum),
            "steps": len(transitions),
        }
        numeric = [
            value
            for value in metrics.values()
            if isinstance(value, (float, int)) and not isinstance(value, bool)
        ]
        if not np.isfinite(np.asarray(numeric, np.float64)).all():
            raise FloatingPointError(f"non-finite episode metrics: {metrics}")
        return transitions, metrics
    finally:
        env.close()


def _graph_imitation(
    model: CooperativeLaneActorCritic,
    optimizer: torch.optim.Optimizer,
    demonstrations: list[dict[str, Any]],
    protocol: MixedLaneProtocol,
    device: torch.device,
) -> dict[str, float]:
    features = torch.stack([item["features"] for item in demonstrations]).to(device)
    masks = torch.stack([item["change_mask"] for item in demonstrations]).to(device)
    gap_active = torch.stack([item["gap_active"] for item in demonstrations]).to(device)
    changes = torch.stack([item["change"] for item in demonstrations]).to(device)
    gap_raw = torch.stack([item["gap_raw"] for item in demonstrations]).to(device)
    losses: list[float] = []
    batch_size = min(protocol.minibatch_size, len(demonstrations))
    for _ in range(protocol.imitation_epochs):
        order = torch.randperm(len(demonstrations), device=device)
        for start in range(0, len(demonstrations), batch_size):
            index = order[start : start + batch_size]
            logits, gap_mean, _ = model(features[index])
            lane_loss = (
                nn.functional.binary_cross_entropy_with_logits(
                    logits, changes[index], reduction="none"
                )
                * masks[index].float()
            ).sum() / masks[index].float().sum().clamp_min(1.0)
            active = gap_active[index].float()[:, None]
            gap_loss = ((gap_mean - gap_raw[index]).pow(2) * active).sum() / (
                active.sum().clamp_min(1.0) * gap_mean.shape[-1]
            )
            loss = lane_loss + 0.5 * gap_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), protocol.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return {"imitation_loss": float(np.mean(losses))}


def _graph_ppo(
    model: CooperativeLaneActorCritic,
    optimizer: torch.optim.Optimizer,
    transitions: list[dict[str, Any]],
    protocol: MixedLaneProtocol,
    device: torch.device,
    episode_number: int,
) -> dict[str, float]:
    rewards = torch.as_tensor(
        [item["reward"] for item in transitions], dtype=torch.float32
    )
    values = torch.as_tensor(
        [item["value"] for item in transitions], dtype=torch.float32
    )
    dones = torch.as_tensor(
        [float(item["done"]) for item in transitions], dtype=torch.float32
    )
    advantage = torch.zeros_like(rewards)
    gae = torch.tensor(0.0)
    next_value = torch.tensor(0.0)
    for index in reversed(range(len(transitions))):
        continuation = 1.0 - dones[index]
        delta = rewards[index] + protocol.discount * next_value * continuation - values[index]
        gae = delta + protocol.discount * protocol.gae_lambda * continuation * gae
        advantage[index] = gae
        next_value = values[index]
    returns = advantage + values
    advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)
    batch = {
        key: torch.stack([item[key] for item in transitions]).to(device)
        for key in ("features", "change_mask", "gap_active", "change", "gap_raw")
    }
    old_log_prob = torch.as_tensor(
        [item["log_prob"] for item in transitions],
        dtype=torch.float32,
        device=device,
    )
    advantage = advantage.to(device)
    returns = returns.to(device)
    losses: list[float] = []
    prior_weight = float(
        protocol.imitation_prior_initial_weight
        * np.exp(-episode_number / protocol.imitation_prior_decay_episodes)
    )
    for _ in range(protocol.ppo_epochs):
        log_prob, entropy, value = model.evaluate_actions(
            batch["features"], batch["change_mask"], batch["gap_active"],
            batch["change"], batch["gap_raw"]
        )
        ratio = (log_prob - old_log_prob).exp()
        policy_loss = -torch.minimum(
            ratio * advantage,
            ratio.clamp(1.0 - protocol.clip_ratio, 1.0 + protocol.clip_ratio) * advantage,
        ).mean()
        value_loss = 0.5 * (value - returns).pow(2).mean()
        logits, gap_mean, _ = model(batch["features"])
        prior_lane_loss = (
            nn.functional.binary_cross_entropy_with_logits(
                logits,
                torch.ones_like(logits),
                reduction="none",
            )
            * batch["change_mask"].float()
        ).sum() / batch["change_mask"].float().sum().clamp_min(1.0)
        active = batch["gap_active"].float()[:, None]
        prior_gap_loss = ((gap_mean + 0.6).pow(2) * active).sum() / (
            active.sum().clamp_min(1.0) * gap_mean.shape[-1]
        )
        prior_loss = prior_lane_loss + 0.5 * prior_gap_loss
        loss = (
            policy_loss
            + protocol.value_weight * value_loss
            - protocol.entropy_weight * entropy.mean()
            + prior_weight * prior_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), protocol.max_grad_norm)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return {
        "ppo_loss": float(np.mean(losses)),
        "imitation_prior_weight": prior_weight,
    }


def _save(
    path: Path,
    method: str,
    protocol: MixedLaneProtocol,
    base_checkpoint: Path | None,
    episode: int,
    score: float,
    metrics: dict[str, Any],
    graph_model: CooperativeLaneActorCritic | None,
    graph_optimizer: torch.optim.Optimizer | None,
    native: NativeLaneAlgorithm | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "method": method,
        "episode": episode,
        "score": score,
        "metrics": metrics,
        "protocol": protocol.to_dict(),
        "base_checkpoint": None if base_checkpoint is None else str(base_checkpoint),
        "base_checkpoint_sha256": _sha256(base_checkpoint),
        "base_controller_frozen": True,
        "lateral_policy_shared": False,
        "lateral_algorithm": "graph_ppo" if method == "graph" else method,
    }
    if graph_model is not None:
        payload["lateral_state"] = graph_model.state_dict()
        payload["optimizer_state"] = graph_optimizer.state_dict() if graph_optimizer else None
        payload["trained_parameters"] = sum(
            parameter.numel() for parameter in graph_model.parameters()
        )
    else:
        assert native is not None
        payload["lateral_state"] = native.state_dict()
        modules = [
            value
            for value in vars(native).values()
            if isinstance(value, nn.Module)
        ]
        payload["trained_parameters"] = sum(
            parameter.numel() for module in modules for parameter in module.parameters()
        )
    torch.save(payload, path)


def _selection_score(metrics: dict[str, Any]) -> float:
    return float(
        1000.0 * float(metrics["success"])
        + 100.0 * metrics["maneuvers_completed"]
        - metrics["time_s"]
        - 5.0 * metrics["maximum_speed_difference_mps"]
        - 2.0 * metrics["traffic_delay_s"]
        - 25.0 * float(metrics["collision"])
    )


def _aggregate_selection_metrics(
    validation_metrics: list[dict[str, Any]],
) -> tuple[float, dict[str, Any]]:
    """Aggregate fixed validation scenes with feasibility as the first rank."""

    if not validation_metrics:
        raise ValueError("selection requires at least one validation episode")
    success_count = sum(bool(row["success"]) for row in validation_metrics)
    collision_count = sum(bool(row["collision"]) for row in validation_metrics)
    mean_maneuvers = float(
        np.mean([float(row["maneuvers_completed"]) for row in validation_metrics])
    )
    mean_quality = float(
        np.mean([_selection_score(row) for row in validation_metrics])
    )
    # This scalar is a deterministic encoding of a lexicographic ranking:
    # feasible scenarios, completed maneuvers, then continuous quality.
    score = float(
        1_000_000.0 * success_count
        + 10_000.0 * mean_maneuvers
        + mean_quality
    )
    mean_keys = (
        "time_s",
        "leader_station_m",
        "minimum_gap_m",
        "maximum_gap_m",
        "maximum_speed_difference_mps",
        "traffic_delay_s",
        "traffic_braking_integral",
        "traffic_hard_brake_events",
        "traffic_max_deceleration_mps2",
        "traffic_spawned_count",
        "rejected_lane_commands",
        "lane_reward_sum",
        "team_reward_sum",
        "steps",
    )
    aggregate: dict[str, Any] = {
        "success": success_count == len(validation_metrics),
        "success_count": success_count,
        "success_rate": float(success_count / len(validation_metrics)),
        "collision": collision_count > 0,
        "collision_count": collision_count,
        "failure": any(bool(row["failure"]) for row in validation_metrics),
        "maneuvers_completed": mean_maneuvers,
        "validation_episode_count": len(validation_metrics),
        "validation_metrics": validation_metrics,
    }
    for key in mean_keys:
        aggregate[key] = float(
            np.mean([float(row[key]) for row in validation_metrics])
        )
    return score, aggregate


def train(method: str, protocol: MixedLaneProtocol, device_name: str) -> Path:
    if method not in LEARNED_METHODS:
        raise ValueError("CACC+LC is rule based and must not be trained")
    method_index = METHODS.index(method)
    seed = protocol.seed + method_index * 10_000
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_name)
    base = create_base_policy(method, protocol, device_name)
    graph_model: CooperativeLaneActorCritic | None = None
    graph_optimizer: torch.optim.Optimizer | None = None
    native: NativeLaneAlgorithm | None = None
    output = protocol.lateral_checkpoint(method)
    assert output is not None
    latest = output.with_name("latest.pt")
    start_episode = 0
    imitation: dict[str, float]
    if method == "graph":
        graph_model = graph_adapter(device_name)
        graph_optimizer = torch.optim.Adam(
            graph_model.parameters(), lr=protocol.learning_rate
        )
        expert_policy: Any = GraphMixedLanePolicy(
            base, graph_model, device_name, deterministic=True, expert=True
        )
    else:
        native = create_native_algorithm(method, protocol, device_name, seed)
        expert_policy = NativeMixedLanePolicy(
            base, native, deterministic=True, expert=True
        )
    if latest.exists():
        resume = torch.load(latest, map_location="cpu", weights_only=False)
        if resume["method"] != method:
            raise ValueError(f"resume checkpoint is {resume['method']}, not {method}")
        if resume["protocol"]["fingerprint"] != protocol.to_dict()["fingerprint"]:
            raise ValueError("mixed-lane protocol changed since the saved run")
        start_episode = int(resume["episode"])
        imitation = {
            "imitation_loss": float(
                resume.get("metrics", {}).get("imitation_loss", np.nan)
            )
        }
        if method == "graph":
            assert graph_model is not None and graph_optimizer is not None
            graph_model.load_state_dict(resume["lateral_state"])
            graph_optimizer.load_state_dict(resume["optimizer_state"])
        else:
            native = restore_native_algorithm(
                method, protocol, device_name, seed, resume["lateral_state"]
            )
        print(
            json.dumps(
                {"phase": "resume", "method": method, "episode": start_episode}
            ),
            flush=True,
        )
    else:
        demonstrations: list[dict[str, Any]] = []
        for episode in range(protocol.imitation_episodes):
            episode_seed = seed + episode
            transitions, metrics = _episode(
                expert_policy, protocol, episode_seed, training=True
            )
            demonstrations.extend(transitions)
            print(
                json.dumps(
                    {"phase": "imitation", "method": method, **metrics}
                ),
                flush=True,
            )
        if method == "graph":
            assert graph_model is not None and graph_optimizer is not None
            imitation = _graph_imitation(
                graph_model, graph_optimizer, demonstrations, protocol, device
            )
        else:
            assert native is not None
            imitation = native.behavior_clone(
                np.stack([item["state"] for item in demonstrations]),
                np.stack([item["action_unit"] for item in demonstrations]),
            )
    if method == "graph":
        assert graph_model is not None
        policy: Any = GraphMixedLanePolicy(
            base, graph_model, device_name, deterministic=False
        )
    else:
        assert native is not None
        policy = NativeMixedLanePolicy(base, native, deterministic=False)
    best_score = -float("inf")
    if output.exists():
        best = torch.load(output, map_location="cpu", weights_only=False)
        if best["protocol"]["fingerprint"] == protocol.to_dict()["fingerprint"]:
            best_score = float(best["score"])

    def select_deployment(candidate_episode: int) -> None:
        nonlocal best_score
        if method == "graph":
            assert graph_model is not None
            deterministic_policy: Any = GraphMixedLanePolicy(
                base, graph_model, device_name, deterministic=True
            )
        else:
            assert native is not None
            deterministic_policy = NativeMixedLanePolicy(
                base, native, deterministic=True
            )
        selection_seeds = [
            protocol.seed + offset for offset in protocol.selection_seed_offsets
        ]
        validation_rows = []
        for selection_seed in selection_seeds:
            _, validation_metrics = _episode(
                deterministic_policy,
                protocol,
                selection_seed,
                training=False,
            )
            validation_rows.append(validation_metrics)
        score, aggregate = _aggregate_selection_metrics(validation_rows)
        selection_row = {
            "phase": "deterministic_selection",
            "method": method,
            "episode": candidate_episode,
            **aggregate,
            **imitation,
            "score": score,
            "selection_timing": "fixed_three_seed_deterministic_after_update",
            "selection_seeds": selection_seeds,
            "selection_ranking": (
                "lexicographic_success_maneuvers_continuous_quality"
            ),
        }
        print(json.dumps(selection_row, sort_keys=True), flush=True)
        if score > best_score:
            best_score = score
            _save(
                output,
                method,
                protocol,
                protocol.base_checkpoint(method),
                candidate_episode,
                score,
                selection_row,
                graph_model,
                graph_optimizer,
                native,
            )

    if start_episode == 0 and not output.exists():
        select_deployment(0)
    for episode in range(start_episode, protocol.training_episodes):
        episode_seed = seed + protocol.imitation_episodes + episode
        transitions, metrics = _episode(
            policy, protocol, episode_seed, training=True
        )
        score = _selection_score(metrics)
        if method == "graph":
            assert graph_model is not None and graph_optimizer is not None
            update = _graph_ppo(
                graph_model,
                graph_optimizer,
                transitions,
                protocol,
                device,
                episode + 1,
            )
        else:
            assert native is not None
            update = native.update_episode(transitions)
        if not all(np.isfinite(float(value)) for value in update.values()):
            raise FloatingPointError(f"non-finite {method} update: {update}")
        row = {
            "phase": "training",
            "method": method,
            "episode": episode + 1,
            **metrics,
            **imitation,
            **update,
            "score": score,
        }
        print(json.dumps(row, sort_keys=True), flush=True)
        _save(
            latest, method, protocol, protocol.base_checkpoint(method),
            episode + 1, score, row, graph_model, graph_optimizer, native
        )
        if (episode + 1) % protocol.selection_interval_episodes == 0:
            select_deployment(episode + 1)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=LEARNED_METHODS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--imitation-episodes", type=int)
    arguments = parser.parse_args()
    protocol = MixedLaneProtocol()
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
