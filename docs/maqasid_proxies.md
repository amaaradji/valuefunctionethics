# Maqasid al-Shari'ah Proxies — Measurement Specification

**Version:** 0.1
**Status:** Draft (Milestone 1)

> **Scope note:** This document defines empirical, measurable proxies inspired by the classical Islamic legal framework of Maqasid al-Shari'ah (objectives of the law). The goal is to operationalise these objectives as scalar indices in a toy simulation. Nothing here constitutes a religious ruling or a claim about Islamic jurisprudence.

---

## Background

Classical Islamic legal theory identifies five (sometimes six) essential objectives that law should protect and promote:

1. **Life (nafs)** — physical existence, health, bodily integrity
2. **Intellect (ʿaql)** — reason, knowledge, education, mental health
3. **Wealth (māl)** — economic security, fair distribution, property rights
4. **Family/Lineage (nasl)** — family integrity, child welfare, social bonds
5. **Faith/Religion (dīn)** — spiritual well-being, moral capital (handled as optional)

A sixth — **honour (ʿird)** — is sometimes added; we fold it into the trust dimension.

In this simulation, each objective maps to a **continuous index in [0, 1]** computed from the society state vector. Changes in these indices per timestep form the components of the reward signal.

---

## Proxy Definitions

### 1. Life Index  `h`

**Real-world analogue:** under-5 mortality rate, life expectancy, disease burden (DALYs), access to emergency care.

**Simulation proxy:**
- `h ∈ [0, 1]`; higher = better population health.
- **Per-step measurement:** `h_t` read directly from state vector.
- **Per-step delta:** `Δh = h_{t+1} - h_t`.
- **Critical threshold:** if `h < 0.1`, the dimension is "in crisis" — a penalty multiplier of 2× applies to this dimension's reward component.

**Actions that primarily affect it:**
- `INVEST_HEALTH` → +0.04 immediate
- `SUBSIDIZE_FAMILIES` → +0.015 (nutrition/stability)
- `EXPLOIT` → −0.03 (diverted resources, stress)

---

### 2. Intellect Index  `e`

**Real-world analogue:** adult literacy rate, school enrollment, access to information, freedom of thought.

**Simulation proxy:**
- `e ∈ [0, 1]`; higher = more educated, more informed population.
- **Delayed dynamics:** education returns are not immediate. An "education pipeline" of depth 3 stores fractional investments; the full benefit is released only once the pipeline is full.
- **Per-step delta:** `Δe = e_{t+1} - e_t`.

**Actions that primarily affect it:**
- `INVEST_EDUCATION` → adds to pipeline; eventual +0.04 to `e`
- `EXPLOIT` → −0.01 (resource diversion, propaganda/suppression)

---

### 3. Wealth-Distribution Index  `w`

**Real-world analogue:** Gini-complement (1 − Gini coefficient), poverty headcount (inverted), basic-needs fulfilment.

**Simulation proxy:**
- `w ∈ [0, 1]`; higher = more equal, less poverty.
- Does not represent total GDP; represents distribution fairness.
- **Per-step delta:** `Δw = w_{t+1} - w_t`.

**Actions that primarily affect it:**
- `ENFORCE_ANTITHEFT` → +0.025 (property security)
- `SUBSIDIZE_FAMILIES` → +0.01 (poverty reduction)
- `EXPLOIT` → +0.03 short-term for the "ruler", but −0.04 to population `w` (extraction)
- `INVEST_HEALTH` → −0.01 (resource cost)

---

### 4. Family-Stability Index  `f`

**Real-world analogue:** household intact rate, divorce/separation rate (inverted), child-welfare score, elderly-care coverage.

**Simulation proxy:**
- `f ∈ [0, 1]`; higher = more stable family structures.
- **Per-step delta:** `Δf = f_{t+1} - f_t`.

**Actions that primarily affect it:**
- `SUBSIDIZE_FAMILIES` → +0.03
- `EXPLOIT` → −0.025 (economic stress fractures families)
- `IGNORE` → −0.008 (passive neglect)

---

### 5. Social Trust Index  `t`

**Real-world analogue:** rule-of-law index, contract enforcement, community cooperation, interpersonal trust surveys.

**Note:** Trust is a **cross-cutting** enabler; it is not a primary Maqasid dimension but captures honour (ʿird) and acts as a multiplier on all other outcomes. We treat it as a dimension with a downweighted (0.8) reward contribution.

**Simulation proxy:**
- `t ∈ [0, 1]`; higher = more cooperative, lower corruption.
- **Cascade effect:** sustained exploitation (≥2 of last 5 steps) doubles trust decay rate.
- **Per-step delta:** `Δt = t_{t+1} - t_t`.

---

### 6. Faith/Moral-Capital Index  `r`  *(optional, disabled by default)*

**Rationale for inclusion:** In the Islamic framework, preservation of dīn is listed first among the Maqasid. However, operationalising it as a simulation variable is epistemically challenging and risks oversimplification.

**Design decision:** disabled by default (`use_faith=False`). When enabled, it is treated as a proxy for **moral-capital** — honesty norms, prosocial behaviour, long-termism — not for theological correctness.

**Proxy:** community moral-capital score derived from honesty-in-transactions events and long-term thinking proxies.

**When disabled:** `r = 0.5` (neutral constant); excluded from reward.

---

## Composite Reward Formula

```python
def compute_reward(
    state_before: np.ndarray,
    state_after: np.ndarray,
    weights: dict,
    use_faith: bool = False,
    crisis_multiplier: float = 2.0,
) -> tuple[float, dict]:
    """
    Returns (scalar_reward, components_dict).

    components_dict keys: "life", "intellect", "wealth", "family", "trust", "faith"
    """
    deltas = state_after - state_before
    dims = ["life", "intellect", "wealth", "family", "trust", "faith"]

    components = {}
    total = 0.0
    for i, name in enumerate(dims):
        if name == "faith" and not use_faith:
            components[name] = 0.0
            continue
        w = weights.get(name, 1.0)
        # apply crisis multiplier if dimension is in critical zone
        crisis = crisis_multiplier if state_before[i] < 0.1 else 1.0
        c = w * crisis * deltas[i]
        components[name] = c
        total += c
    return total, components
```

---

## Ethicality Proxy

For a trained agent with value function V(s):

```
ethicality_proxy(s, a) = V(s') - V(s)  ≈  A(s, a)
```

Where `s'` is the **expected next state** under action `a` from state `s`.

This measure:
- Is **positive** for actions that move society toward better Maqasid outcomes (as learned by the agent).
- Is **negative** for harmful or regressive actions.
- Can be computed **counterfactually** by comparing V(s') across all possible actions.

The advantage function `A(s, a) = Q(s, a) − V(s)` from PPO/A2C directly gives this proxy during training.

---

## Limitations and Caveats

1. All index values are invented for research purposes. They are not calibrated to real-world data.
2. The weights are equal by default; choosing weights is itself a normative act that should be documented.
3. "Faith" as a measurable index is a gross simplification; users should consider disabling it.
4. The toy dynamics do not capture second- and third-order real-world effects.
5. This is a research prototype, not a policy tool.
