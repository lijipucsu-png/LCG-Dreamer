"""SUMO platoon-control scenario and PettingZoo environment."""

from typing import Any

__all__ = ["PlatoonParallelEnv"]


def __getattr__(name: str) -> Any:
    if name == "PlatoonParallelEnv":
        from .env import PlatoonParallelEnv

        return PlatoonParallelEnv
    raise AttributeError(name)
