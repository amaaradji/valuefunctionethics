"""
tests/test_rewards.py
=====================
Unit tests for the Maqasid proxy reward module.

Run with:  pytest tests/test_rewards.py -v
"""

import numpy as np
import pytest

from valuefunctionethics.rewards.maqasid import (
    DEFAULT_WEIGHTS,
    DIM_FAITH,
    DIM_NAMES,
    N_DIMS,
    SURVIVAL_BONUS,
    RewardConfig,
    RewardResult,
    compute_reward,
    ethicality_proxy,
    is_collapsed,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _state(h=0.5, e=0.5, w=0.5, f=0.5, t=0.5, r=0.5) -> np.ndarray:
    return np.array([h, e, w, f, t, r], dtype=np.float32)


# ---------------------------------------------------------------------------
# RewardConfig defaults
# ---------------------------------------------------------------------------

class TestRewardConfig:
    def test_default_weights_match_module_defaults(self):
        cfg = RewardConfig()
        for k, v in DEFAULT_WEIGHTS.items():
            assert cfg.weights[k] == pytest.approx(v)

    def test_faith_disabled_sets_weight_zero(self):
        cfg = RewardConfig(use_faith=False)
        assert cfg.weights["faith"] == 0.0

    def test_faith_enabled_preserves_weight(self):
        cfg = RewardConfig(use_faith=True, weights={"faith": 0.7})
        assert cfg.weights["faith"] == pytest.approx(0.7)

    def test_missing_weights_filled_from_defaults(self):
        cfg = RewardConfig(weights={"life": 2.0})
        assert cfg.weights["intellect"] == pytest.approx(DEFAULT_WEIGHTS["intellect"])
        assert cfg.weights["life"] == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# compute_reward — basic behaviour
# ---------------------------------------------------------------------------

class TestComputeReward:
    def test_returns_reward_result(self):
        s0 = _state()
        s1 = _state(h=0.6)
        result = compute_reward(s0, s1)
        assert isinstance(result, RewardResult)
        assert isinstance(result.total, float)
        assert isinstance(result.components, dict)

    def test_no_change_gives_only_survival_bonus(self):
        s = _state()
        result = compute_reward(s, s.copy())
        assert result.total == pytest.approx(SURVIVAL_BONUS)
        # all dimension components should be zero
        for name in DIM_NAMES:
            if name != "faith":
                assert result.components[name] == pytest.approx(0.0)

    def test_improvement_gives_positive_reward(self):
        s0 = _state(h=0.4)
        s1 = _state(h=0.6)  # life improved
        result = compute_reward(s0, s1)
        assert result.total > 0
        assert result.components["life"] > 0

    def test_decline_gives_negative_reward(self):
        s0 = _state(h=0.6)
        s1 = _state(h=0.4)
        result = compute_reward(s0, s1)
        assert result.total < 0
        assert result.components["life"] < 0

    def test_component_keys_present(self):
        result = compute_reward(_state(), _state())
        for name in DIM_NAMES:
            assert name in result.components
        assert "survival_bonus" in result.components

    def test_total_equals_sum_of_components(self):
        s0 = _state(h=0.3, e=0.5, w=0.7)
        s1 = _state(h=0.4, e=0.55, w=0.65)
        result = compute_reward(s0, s1)
        component_sum = sum(result.components.values())
        assert result.total == pytest.approx(component_sum)

    def test_faith_excluded_when_disabled(self):
        cfg = RewardConfig(use_faith=False)
        s0 = _state(r=0.2)
        s1 = _state(r=0.9)  # big faith change — should not affect total
        result = compute_reward(s0, s1, config=cfg)
        assert result.components["faith"] == pytest.approx(0.0)

    def test_faith_included_when_enabled(self):
        cfg = RewardConfig(use_faith=True, weights={"faith": 1.0})
        s0 = _state(r=0.2)
        s1 = _state(r=0.9)
        result = compute_reward(s0, s1, config=cfg)
        assert result.components["faith"] != pytest.approx(0.0)
        assert result.components["faith"] > 0

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            compute_reward(np.zeros(3), np.zeros(6))

    def test_weights_scale_contributions(self):
        cfg_low = RewardConfig(weights={"life": 0.5})
        cfg_high = RewardConfig(weights={"life": 2.0})
        s0 = _state(h=0.4)
        s1 = _state(h=0.6)
        r_low = compute_reward(s0, s1, config=cfg_low)
        r_high = compute_reward(s0, s1, config=cfg_high)
        # life contribution scales with weight
        assert r_high.components["life"] == pytest.approx(
            r_low.components["life"] * 4.0
        )


# ---------------------------------------------------------------------------
# Crisis multiplier
# ---------------------------------------------------------------------------

class TestCrisisMultiplier:
    def test_crisis_amplifies_reward(self):
        cfg_default = RewardConfig(crisis_multiplier=1.0)
        cfg_crisis = RewardConfig(crisis_multiplier=3.0)
        # put life in crisis zone
        s0 = _state(h=0.05)
        s1 = _state(h=0.08)
        r_default = compute_reward(s0, s1, config=cfg_default)
        r_crisis = compute_reward(s0, s1, config=cfg_crisis)
        assert r_crisis.components["life"] == pytest.approx(
            r_default.components["life"] * 3.0
        )

    def test_no_crisis_multiplier_above_threshold(self):
        cfg = RewardConfig(crisis_threshold=0.1, crisis_multiplier=5.0)
        s0 = _state(h=0.5)  # well above threshold
        s1 = _state(h=0.6)
        result = compute_reward(s0, s1, config=cfg)
        # contribution = weight * 1.0 * delta (no multiplier)
        expected = cfg.weights["life"] * 1.0 * 0.1
        assert result.components["life"] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# is_collapsed
# ---------------------------------------------------------------------------

class TestIsCollapsed:
    def test_healthy_state_not_collapsed(self):
        assert not is_collapsed(_state())

    def test_low_health_triggers_collapse(self):
        assert is_collapsed(_state(h=0.02))

    def test_low_education_triggers_collapse(self):
        assert is_collapsed(_state(e=0.02))

    def test_faith_alone_does_not_trigger_collapse(self):
        # Faith index is excluded from collapse check
        assert not is_collapsed(_state(r=0.01))

    def test_boundary_not_collapsed(self):
        # exactly at threshold (0.05) should NOT collapse (must be strictly below)
        assert not is_collapsed(_state(h=0.05))

    def test_just_below_boundary_collapses(self):
        assert is_collapsed(_state(h=0.049))


# ---------------------------------------------------------------------------
# ethicality_proxy
# ---------------------------------------------------------------------------

class TestEthicalityProxy:
    def _mock_value_fn(self, state: np.ndarray) -> float:
        """Simple mock: V(s) = sum of first four dimensions."""
        return float(np.sum(state[:4]))

    def test_returns_dict_keyed_by_action(self):
        s = _state()
        next_states = {
            0: _state(h=0.7),
            1: _state(e=0.7),
            2: _state(w=0.3),  # worse
        }
        result = ethicality_proxy(self._mock_value_fn, s, next_states)
        assert set(result.keys()) == {0, 1, 2}

    def test_improving_action_has_positive_score(self):
        s = _state(h=0.5)
        next_states = {0: _state(h=0.8)}  # life improved
        result = ethicality_proxy(self._mock_value_fn, s, next_states)
        assert result[0] > 0

    def test_declining_action_has_negative_score(self):
        s = _state(h=0.5)
        next_states = {0: _state(h=0.2)}  # life declined
        result = ethicality_proxy(self._mock_value_fn, s, next_states)
        assert result[0] < 0

    def test_neutral_action_has_zero_score(self):
        s = _state()
        next_states = {0: s.copy()}
        result = ethicality_proxy(self._mock_value_fn, s, next_states)
        assert result[0] == pytest.approx(0.0)

    def test_scores_are_relative_to_current_state(self):
        s = _state(h=0.5, e=0.5)
        next_states = {
            0: _state(h=0.7, e=0.5),  # +0.2 life
            1: _state(h=0.5, e=0.8),  # +0.3 edu
        }
        result = ethicality_proxy(self._mock_value_fn, s, next_states)
        # action 1 should score higher
        assert result[1] > result[0]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_inputs_same_output(self):
        s0 = _state(h=0.4, e=0.6, w=0.5, f=0.3)
        s1 = _state(h=0.5, e=0.65, w=0.48, f=0.35)
        r1 = compute_reward(s0, s1)
        r2 = compute_reward(s0, s1)
        assert r1.total == pytest.approx(r2.total)
        for k in r1.components:
            assert r1.components[k] == pytest.approx(r2.components[k])
