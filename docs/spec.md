# Environment Specification — TownEnv

**Version:** 0.1
**Status:** Draft (Milestone 1)

---

## Overview

`TownEnv` is a single-agent discrete-time gymnasium environment representing a stylised **town/society** whose aggregate well-being the agent-policymaker tries to improve.  The agent selects **governance actions** each timestep; the environment transitions to a new society state according to deterministic-but-delayed dynamics with bounded stochasticity.

The environment is intentionally small so that:
1. Tabular sanity checks are tractable.
2. Learned value functions can be inspected and plotted.
3. Ethicality proxies (V(s') − V(s)) are human-interpretable.

---

## State Space

The state `s ∈ ℝ⁶` (or `np.float32[6]`) is a normalised vector of six **society-level indices**, each in `[0, 1]` where 1 is best:

| Index | Symbol | Maqasid dimension | Meaning |
|-------|--------|-------------------|---------|
| 0 | `h`  | Life (nafs)       | Health index — weighted avg of mortality risk, disease burden, access to care |
| 1 | `e`  | Intellect (ʿaql)  | Education index — literacy, school enrollment, critical-thinking proxy |
| 2 | `w`  | Wealth (māl)      | Wealth-distribution proxy — Gini complement (1 − Gini), scaled |
| 3 | `f`  | Family (nasl)     | Family-stability proxy — household cohesion, support-network index |
| 4 | `t`  | (cross-cutting)   | Social trust level — adherence to contracts, rule of law, community cooperation |
| 5 | `r`  | Faith (dīn) [opt] | Religious/moral-capital index — optional, disabled by default (`use_faith=False`) |

When `use_faith=False` (default), dimension 5 is fixed at 0.5 and excluded from the reward.

### Initial State Distribution

At `reset(seed=...)`, each active index is drawn from `Uniform(0.3, 0.7)`.  The faith index, if enabled, is drawn from `Uniform(0.4, 0.7)` to reflect modest starting religiosity.

---

## Action Space

`Discrete(6)` — six governance actions:

| Action ID | Name              | Primary target       | Notes |
|-----------|-------------------|----------------------|-------|
| 0 | `INVEST_HEALTH`       | ↑ h, slight ↑ t      | Short-term boost; expensive (↓ w slightly) |
| 1 | `INVEST_EDUCATION`    | ↑ e, ↑ t (delayed)   | Effect materialises after ~3 steps |
| 2 | `ENFORCE_ANTITHEFT`   | ↑ w, ↑ t             | Reduces predatory behaviour |
| 3 | `SUBSIDIZE_FAMILIES`  | ↑ f, slight ↑ h      | Reduces poverty-driven family breakdown |
| 4 | `IGNORE`              | slow decay on all     | No active intervention; resources saved but indices drift down |
| 5 | `EXPLOIT`             | ↑ w short-term, ↓ f, ↓ t, ↓ h long-term | Extraction — harms trust and health |

---

## Transition Dynamics

Dynamics are **delayed and coupled**:

```
s_{t+1} = clip( s_t + Δ_action + Δ_decay + ε , 0, 1 )
```

Where:
- `Δ_action` is the immediate effect vector of the chosen action (see `envs/dynamics.py`).
- `Δ_decay` is a small per-step natural decay (`-0.005` per active dimension) simulating entropy/neglect.
- `ε ~ N(0, noise_std)` with `noise_std=0.01` by default (fully reproducible from seed).
- **Delayed effects**: education investment (`action=1`) accumulates a "pipeline" buffer. The full benefit (+0.04 to `e`, +0.02 to `t`) is only released after 3 consecutive steps (or 3 total if spread out, weighted by recency). This is stored in `info["pending_education"]`.
- **Exploit cascade**: if `EXPLOIT` is chosen in 2 of the last 5 steps, trust decays at 2× rate for the next 5 steps.

---

## Episode and Termination

- **Max steps:** 100 (configurable via `max_steps`).
- **Early termination:** if any index falls below `0.05` (societal collapse threshold), the episode ends with a large negative reward.
- **Truncation:** gymnasium `truncated=True` at `max_steps`.

---

## Reward

See [maqasid_proxies.md](maqasid_proxies.md) for full reward definition.

The scalar reward `r_t` is the **weighted sum of per-dimension deltas** plus a small survival bonus:

```
r_t = Σ_i  w_i * Δs_i  +  bonus_survival
```

Default weights (equal by default, configurable):

| Dimension | Default weight |
|-----------|---------------|
| life      | 1.0 |
| intellect | 1.0 |
| wealth    | 1.0 |
| family    | 1.0 |
| trust     | 0.8 |
| faith     | 0.0 (disabled) |

---

## Reproducibility

All random draws use `np.random.default_rng(seed)`.
Passing `reset(seed=42)` guarantees identical trajectories for identical action sequences.

---

## gymnasium API Compliance

`TownEnv` will implement:
- `reset(seed, options)` → `(obs, info)`
- `step(action)` → `(obs, reward, terminated, truncated, info)`
- `render(mode="human")` → prints ASCII state table
- `observation_space`: `Box(0, 1, shape=(6,), dtype=np.float32)`
- `action_space`: `Discrete(6)`
