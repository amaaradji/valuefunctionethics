"""
rewards/maqasid.py
==================
Maqasid al-Shari'ah-inspired reward computation.

State vector layout (indices 0-5):
    0  h  — life / health index
    1  e  — intellect / education index
    2  w  — wealth-distribution index
    3  f  — family-stability index
    4  t  — social-trust index
    5  r  — faith / moral-capital index (optional, disabled by default)

All indices are in [0, 1]; higher is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

DIM_NAMES: tuple[str, ...] = ("life", "intellect", "wealth", "family", "trust", "faith")
DIM_LIFE = 0
DIM_INTELLECT = 1
DIM_WEALTH = 2
DIM_FAMILY = 3
DIM_TRUST = 4
DIM_FAITH = 5

N_DIMS = len(DIM_NAMES)

DEFAULT_WEIGHTS: dict[str, float] = {
    "life": 1.0,
    "intellect": 1.0,
    "wealth": 1.0,
    "family": 1.0,
    "trust": 0.8,
    "faith": 0.0,  # disabled by default
}

CRISIS_THRESHOLD: float = 0.1   # index below this → crisis multiplier kicks in
CRISIS_MULTIPLIER: float = 2.0  # amplifies reward signal when dimension is critical
SURVIVAL_BONUS: float = 0.005   # small per-step bonus for not triggering collapse


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

class RewardResult(NamedTuple):
    """Return value of :func:`compute_reward`."""

    total: float
    components: dict[str, float]


# ---------------------------------------------------------------------------
# RewardConfig
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    """
    Configuration for the Maqasid reward function.

    Parameters
    ----------
    weights:
        Per-dimension reward weights. Defaults to :data:`DEFAULT_WEIGHTS`.
    use_faith:
        Whether to include the faith/moral-capital dimension in the reward.
        Defaults to False; set to True to enable.
    crisis_threshold:
        Index value below which a dimension is considered "in crisis".
    crisis_multiplier:
        Extra weight applied to a dimension's reward component when it is
        in crisis (to incentivise urgent recovery).
    survival_bonus:
        Small constant reward added every step the environment is not in
        a terminal collapse state.
    """

    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    use_faith: bool = False
    crisis_threshold: float = CRISIS_THRESHOLD
    crisis_multiplier: float = CRISIS_MULTIPLIER
    survival_bonus: float = SURVIVAL_BONUS

    def __post_init__(self) -> None:
        # Fill in any missing keys with defaults
        for k, v in DEFAULT_WEIGHTS.items():
            self.weights.setdefault(k, v)
        if not self.use_faith:
            self.weights["faith"] = 0.0


# ---------------------------------------------------------------------------
# Core reward function
# ---------------------------------------------------------------------------

def compute_reward(
    state_before: np.ndarray,
    state_after: np.ndarray,
    config: RewardConfig | None = None,
) -> RewardResult:
    """
    Compute the Maqasid proxy reward for one environment step.

    Parameters
    ----------
    state_before:
        Society state at time t, shape (6,), dtype float32.
    state_after:
        Society state at time t+1, shape (6,), dtype float32.
    config:
        Reward configuration. Uses defaults when None.

    Returns
    -------
    RewardResult
        Named tuple with ``total`` (float) and ``components`` (dict mapping
        each dimension name to its reward contribution).
    """
    if config is None:
        config = RewardConfig()

    state_before = np.asarray(state_before, dtype=np.float64)
    state_after = np.asarray(state_after, dtype=np.float64)

    if state_before.shape != (N_DIMS,):
        raise ValueError(f"state_before must have shape ({N_DIMS},), got {state_before.shape}")
    if state_after.shape != (N_DIMS,):
        raise ValueError(f"state_after must have shape ({N_DIMS},), got {state_after.shape}")

    deltas = state_after - state_before
    components: dict[str, float] = {}
    total = 0.0

    for i, name in enumerate(DIM_NAMES):
        w = config.weights.get(name, 0.0)
        if w == 0.0:
            components[name] = 0.0
            continue
        # Crisis amplification when a dimension is critically low
        crisis = config.crisis_multiplier if state_before[i] < config.crisis_threshold else 1.0
        contribution = float(w * crisis * deltas[i])
        components[name] = contribution
        total += contribution

    total += config.survival_bonus
    components["survival_bonus"] = config.survival_bonus

    return RewardResult(total=total, components=components)


# ---------------------------------------------------------------------------
# Collapse detection
# ---------------------------------------------------------------------------

def is_collapsed(state: np.ndarray, threshold: float = CRISIS_THRESHOLD / 2) -> bool:
    """
    Return True if any active index has hit the collapse floor.

    Uses half of CRISIS_THRESHOLD (0.05) as the hard termination threshold.
    Faith (index 5) is excluded from collapse detection regardless of config
    because its dynamics are optional and different.
    """
    active = state[:DIM_FAITH]  # indices 0-4
    return bool(np.any(active < threshold))


# ---------------------------------------------------------------------------
# Ethicality proxy (post-hoc, requires a trained value function)
# ---------------------------------------------------------------------------

def ethicality_proxy(
    value_fn,
    state: np.ndarray,
    next_states: dict[int, np.ndarray],
) -> dict[int, float]:
    """
    Compute the ethicality proxy V(s') - V(s) for each candidate action.

    Parameters
    ----------
    value_fn:
        Callable ``(state: np.ndarray) -> float`` — the learned value function.
        For stable-baselines3 policies, wrap ``policy.predict_values(obs)``.
    state:
        Current state, shape (6,).
    next_states:
        Mapping from action_id → expected next state after that action.

    Returns
    -------
    dict mapping action_id → ethicality score (positive = ethically better).
    """
    v_s = float(value_fn(state))
    return {a: float(value_fn(s_prime)) - v_s for a, s_prime in next_states.items()}
