# Design Notes

This document captures architectural decisions and the rationale behind them.

---

## Why gymnasium?

gymnasium (successor to OpenAI gym) is the de facto standard API for RL environments in Python. Using it means:
- Off-the-shelf compatibility with stable-baselines3, Tianshou, CleanRL, etc.
- Standardised reset/step/render interface that researchers are familiar with.
- Easy transition to PettingZoo (multi-agent) later.

## Why stable-baselines3?

- Battle-tested PPO and A2C implementations.
- Built-in logging, callbacks, and model saving.
- CPU-friendly defaults for reproducibility without GPU.
- Value-function access is straightforward via `policy.predict_values(obs)`.

## Why discrete actions (not continuous)?

A `Discrete(6)` action space keeps the policy tables small and the value estimates V(s) easy to plot as heatmaps. The conceptual questions (which governance choice is most ethical?) are inherently categorical.

## On the "faith" dimension

Including a religious dimension in an RL reward is epistemically risky. Our design decision:
- Default `use_faith=False`.
- When enabled, treat it strictly as a **moral-capital proxy** (honesty norms, prosocial behaviour) rather than anything theologically substantive.
- Document this clearly and prominently.

## Delayed dynamics rationale

Real governance actions have delayed payoffs (investment in education returns dividends years later). Modelling this:
- Prevents naive greedy policies from looking optimal.
- Tests whether the RL agent can learn to value deferred rewards.
- Makes the value function V(s) more interesting: it must integrate future expected outcomes, not just immediate deltas.

## Ethicality proxy: V(s') − V(s) vs. A(s, a)

Both measure the same thing in expectation. `A(s, a) = Q(s, a) − V(s)` is cleaner because:
- It is already computed during PPO training.
- It is centred around zero (positive = better than average, negative = worse).
- It decomposes naturally across the Maqasid components when we split the reward.

We store both and let downstream analysis choose.

## Seed strategy

Every stochastic operation goes through a single `np.random.default_rng(seed)` instance. The environment propagates this generator to all sub-components. This ensures:
- Full determinism for a given seed.
- No global state mutation (no `np.random.seed()`).
