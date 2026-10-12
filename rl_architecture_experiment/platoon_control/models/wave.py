"""Conditional variational model for NGSIM braking-wave acceleration profiles."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass
class WavePrediction:
    mean: Tensor
    log_std: Tensor
    posterior_mean: Tensor
    posterior_log_var: Tensor
    prior_mean: Tensor
    prior_log_var: Tensor
    prior_profile: Tensor


class ConditionalWaveCVAE(nn.Module):
    """Generate a six-second acceleration trace conditioned on local traffic history."""

    def __init__(
        self,
        condition_dim: int = 8,
        hidden_dim: int = 128,
        latent_dim: int = 12,
        horizon: int = 60,
    ) -> None:
        super().__init__()
        self.horizon = horizon
        self.latent_dim = latent_dim
        self.condition_encoder = nn.GRU(condition_dim, hidden_dim, batch_first=True)
        self.target_encoder = nn.GRU(1, hidden_dim, batch_first=True)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.posterior = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2 * latent_dim)
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 2 * hidden_dim),
            nn.SiLU(),
            nn.Linear(2 * hidden_dim, 2 * horizon),
        )

    @staticmethod
    def _normal_parameters(values: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_var = values.chunk(2, dim=-1)
        return mean, log_var.clamp(-8.0, 5.0)

    def forward(
        self,
        condition: Tensor,
        target_acceleration: Tensor | None = None,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
    ) -> WavePrediction:
        _, condition_state = self.condition_encoder(condition)
        context = condition_state[-1]
        prior_mean, prior_log_var = self._normal_parameters(self.prior(context))
        if target_acceleration is None:
            posterior_mean, posterior_log_var = prior_mean, prior_log_var
        else:
            _, target_state = self.target_encoder(target_acceleration.unsqueeze(-1))
            posterior_mean, posterior_log_var = self._normal_parameters(
                self.posterior(torch.cat((context, target_state[-1]), dim=-1))
            )
        if deterministic:
            latent = posterior_mean
        else:
            noise = torch.randn(
                posterior_mean.shape,
                device=posterior_mean.device,
                dtype=posterior_mean.dtype,
                generator=generator,
            )
            latent = posterior_mean + (0.5 * posterior_log_var).exp() * noise
        decoded = self.decoder(torch.cat((context, latent), dim=-1))
        mean, log_std = decoded.view(-1, self.horizon, 2).unbind(-1)
        prior_decoded = self.decoder(torch.cat((context, prior_mean), dim=-1))
        prior_profile = prior_decoded.view(-1, self.horizon, 2)[..., 0].clamp(-8.0, 5.0)
        return WavePrediction(
            mean.clamp(-8.0, 5.0),
            log_std.clamp(-4.0, 2.0),
            posterior_mean,
            posterior_log_var,
            prior_mean,
            prior_log_var,
            prior_profile,
        )

    @staticmethod
    def loss(prediction: WavePrediction, target: Tensor, beta: float = 0.02) -> dict[str, Tensor]:
        inverse_variance = torch.exp(-2.0 * prediction.log_std)
        nll = (
            0.5 * (target - prediction.mean).square() * inverse_variance
            + prediction.log_std
        ).mean()
        posterior_variance = prediction.posterior_log_var.exp()
        prior_variance = prediction.prior_log_var.exp()
        kl = 0.5 * (
            prediction.prior_log_var
            - prediction.posterior_log_var
            + (posterior_variance + (prediction.posterior_mean - prediction.prior_mean).square())
            / prior_variance
            - 1.0
        ).mean()
        jerk = (prediction.mean[:, 1:] - prediction.mean[:, :-1]).square().mean()
        prior_reconstruction = torch.nn.functional.smooth_l1_loss(
            prediction.prior_profile / 4.0, target / 4.0
        )
        total = nll + beta * kl + 0.01 * jerk + 0.5 * prior_reconstruction
        return {
            "loss": total,
            "nll": nll,
            "kl": kl,
            "jerk": jerk,
            "prior_reconstruction": prior_reconstruction,
        }

    @torch.no_grad()
    def sample(
        self,
        condition: Tensor,
        generator: torch.Generator | None = None,
        deterministic: bool = False,
    ) -> Tensor:
        prediction = self(condition, deterministic=deterministic, generator=generator)
        if deterministic:
            return prediction.mean
        noise = torch.randn(
            prediction.mean.shape,
            device=prediction.mean.device,
            dtype=prediction.mean.dtype,
            generator=generator,
        )
        return (prediction.mean + prediction.log_std.exp() * noise).clamp(-6.0, 3.0)
