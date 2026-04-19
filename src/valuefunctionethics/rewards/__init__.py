"""Maqasid proxy reward computation."""

from valuefunctionethics.rewards.maqasid import (
    RewardConfig,
    RewardResult,
    compute_reward,
    ethicality_proxy,
    is_collapsed,
    DIM_NAMES,
    DEFAULT_WEIGHTS,
)

__all__ = [
    "RewardConfig",
    "RewardResult",
    "compute_reward",
    "ethicality_proxy",
    "is_collapsed",
    "DIM_NAMES",
    "DEFAULT_WEIGHTS",
]
