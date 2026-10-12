import torch

from dreamerv3_ablation.protocol import PROTOCOL, VARIANTS
from platoon_control.graph_agents import GraphPlatoonController
from platoon_control.graph_rl import GraphDreamerTrainer, GraphRLConfig
from platoon_control.models.dreamer_v3_core import PercentileReturnNormalizer, SymLogTwoHot
from platoon_control.models.graph_dreamer import GraphDreamerConfig, MALatentGraphDreamer, NODE_COUNT


def compact_observation(batch: int, config: GraphDreamerConfig) -> dict[str, torch.Tensor]:
    adjacency = torch.zeros(batch, NODE_COUNT, NODE_COUNT, 2)
    for follower in range(1, NODE_COUNT):
        adjacency[:, follower, follower - 1, 0] = 1.0
    adjacency[..., 1] = 1.0 - torch.eye(NODE_COUNT).view(1, NODE_COUNT, NODE_COUNT)
    physical = torch.zeros(batch, NODE_COUNT, 7)
    physical[..., 0] = 20.0
    physical[:, 1:, 2] = 12.0
    physical[..., 5] = 10.0
    physical[..., 6] = 25.0
    return {
        "node_features": torch.randn(batch, NODE_COUNT, 46),
        "road_profile": torch.randn(batch, NODE_COUNT, 20, 3),
        "v2i_flow": torch.randn(batch, NODE_COUNT, 8, 4),
        "front_entities": torch.randn(batch, NODE_COUNT, 8, 6),
        "front_mask": torch.ones(batch, NODE_COUNT, 8),
        "neighbor_entities": torch.randn(batch, NODE_COUNT, 12, 8),
        "neighbor_mask": torch.ones(batch, NODE_COUNT, 12),
        "role": torch.tensor([[0, 1, 1, 1, 1, 1]]).expand(batch, -1),
        "adjacency": adjacency,
        "edge_ages": torch.zeros_like(adjacency),
        "physical_target": physical,
        "delay_target": torch.zeros(batch, NODE_COUNT, config.prediction_horizon, 3),
        "cutin_target": torch.zeros(batch, NODE_COUNT),
        "front_target": torch.zeros(batch, NODE_COUNT, 4),
        "follower_mask": torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.float32).expand(batch, -1),
    }


def sequence_batch(config: GraphDreamerConfig, batch: int = 2, steps: int = 3):
    observations = [compact_observation(batch, config) for _ in range(steps + 1)]
    stacked = {key: torch.stack([item[key] for item in observations], 1) for key in observations[0]}
    actions = torch.zeros(batch, steps, NODE_COUNT, 1)
    return {
        "observations": stacked,
        "applied_actions": actions,
        "requested_actions": actions.clone(),
        "raw_actions": actions.clone(),
        "rewards": torch.zeros(batch, steps),
        "continues": torch.ones(batch, steps),
        "shielded": torch.zeros(batch, steps, NODE_COUNT),
        "behavior_valid": torch.ones(batch, steps, NODE_COUNT),
    }


def small_config(**overrides) -> GraphDreamerConfig:
    values = dict(
        hidden_dim=32,
        stochastic_dim=4,
        stochastic_classes=4,
        embedding_dim=64,
        communication_dim=8,
        edge_belief_dim=32,
        dreamerv3_distributional_heads=True,
        dreamerv3_value_bins=31,
    )
    values.update(overrides)
    return GraphDreamerConfig(**values)


def test_protocol_is_a_single_module_incremental_ladder() -> None:
    PROTOCOL.validate()
    assert [item.key for item in VARIANTS] == [
        "a0_dreamerv3",
        "a1_graph_perception",
        "a2_latent_v2v",
        "a3_aux_prediction",
        "a4_full",
    ]


def test_twohot_head_reconstructs_targets_and_has_finite_gradients() -> None:
    head = SymLogTwoHot(8, 16, bins=31)
    features = torch.randn(12, 8, requires_grad=True)
    target = torch.linspace(-100.0, 100.0, 12)
    logits = head(features)
    loss = head.loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(features.grad).all()
    assert torch.allclose(head.target(target).sum(-1), torch.ones(12))


def test_percentile_return_normalizer_has_unit_floor() -> None:
    normalizer = PercentileReturnNormalizer(rate=1.0)
    offset, scale = normalizer.update(torch.zeros(32))
    assert offset == 0.0
    assert scale == 1.0


def test_flat_dreamerv3_receives_but_does_not_message_pass_topology() -> None:
    config = small_config(team_context_mode="fixed_order", enable_semantic_v2v=False)
    model = MALatentGraphDreamer(config).eval()
    observation = compact_observation(1, config)
    state = model.initial(1, torch.device("cpu"))
    action = torch.zeros(1, NODE_COUNT, 1)
    with torch.no_grad():
        _, _, first = model.observe_step(
            state, action, observation, observation["adjacency"], observation["edge_ages"]
        )
        changed = observation["adjacency"].clone()
        changed[:, 1, 0, 0] = 0.0
        _, _, second = model.observe_step(
            state, action, observation, changed, observation["edge_ages"]
        )
    assert not torch.allclose(first, second)


def test_graph_perception_ignores_v2v_relation_when_disabled() -> None:
    config = small_config(team_context_mode="graph", enable_semantic_v2v=False)
    model = MALatentGraphDreamer(config).eval()
    observation = compact_observation(1, config)
    state = model.initial(1, torch.device("cpu"))
    action = torch.zeros(1, NODE_COUNT, 1)
    without_v2v = observation["adjacency"].clone()
    without_v2v[..., 1] = 0.0
    with torch.no_grad():
        _, _, first = model.observe_step(
            state, action, observation, without_v2v, observation["edge_ages"]
        )
        _, _, second = model.observe_step(
            state, action, observation, observation["adjacency"], observation["edge_ages"]
        )
    assert torch.allclose(first, second)


def test_distributional_graph_world_and_actor_updates_are_finite() -> None:
    config = small_config(
        team_context_mode="graph",
        enable_semantic_v2v=True,
        enable_auxiliary_losses=True,
        enable_control_priors=False,
    )
    controller = GraphPlatoonController(config, device="cpu", predictor_directory=None)
    trainer = GraphDreamerTrainer(
        controller,
        GraphRLConfig(
            device="cpu",
            batch_size=2,
            sequence_length=3,
            imagination_weight=0.05,
        ),
    )
    batch = sequence_batch(config)
    world = trainer.train_world(batch, 0.0)
    actor = trainer.train_actor_critic(batch, horizon=3, communication_dropout=0.0)
    assert torch.isfinite(torch.tensor(list(world.values()))).all()
    assert torch.isfinite(torch.tensor(list(actor.values()))).all()
    assert trainer.world_updates == 1
    assert trainer.actor_updates == 1
    assert trainer.critic_updates == 1


def test_distributional_world_loss_accepts_node_level_rewards() -> None:
    config = small_config(enable_control_priors=False)
    model = MALatentGraphDreamer(config)
    batch = sequence_batch(config)
    batch["rewards"] = torch.zeros(2, 3, NODE_COUNT)
    losses, _, _ = model.world_model_loss(batch)
    assert torch.isfinite(losses["reward"])
