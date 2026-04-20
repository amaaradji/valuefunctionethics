# valuefunctionethics — Implementation

## Project Summary

This repository is the implementation and evaluation component of a research project proposing a **Maqasid al-Shari'ah-inspired value function** as an ethicality proxy for reinforcement learning agents.

The project operationalises the five objectives of Islamic law (Maqasid al-Shari'ah) — preservation of **life**, **intellect**, **wealth**, **family/lineage**, and **faith** — as measurable proxies for a society-level reward signal. A reinforcement learning agent learns a value function V(s) over aggregated society states. The advantage A(s,a) = V(s') − V(s) then serves as an empirical ethicality proxy for any proposed action.

The companion paper (LaTeX) lives in the `AfterlifeRL` repository (separate repo, same project).

> **Disclaimer:** This is simulation-based research. It does not issue religious rulings. All "maqasid" references are operationalised as empirical, measurable proxies for studying learned value alignment.

---

Python package implementing the Maqasid al-Shari'ah-inspired RL ethicality proxy.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cpu  # CPU-only
pip install -e ".[dev]"
```

Requires Python 3.11+.

## Key Commands

```bash
python experiments/hello_run.py  # smoke test — should print "All imports OK"
pytest                           # run full test suite
```

## Structure

```
src/valuefunctionethics/
├── envs/       # Gymnasium environments (TownEnv, ...)
├── rewards/    # Maqasid proxy reward functions
├── agents/     # Baseline and trained agents
└── utils/      # Seeding, logging, plotting helpers

experiments/
├── configs/    # YAML/JSON experiment configs
├── outputs/    # Plots and metric CSVs (gitignored)
├── hello_run.py
├── run_train.py   # RL training entry point (Milestone 4)
└── run_eval.py    # Evaluation entry point (Milestone 3+)

docs/
├── spec.md              # Environment state/action/transition spec
├── maqasid_proxies.md   # Precise proxy definitions
└── design_notes.md      # Architecture decisions and rationale
```

## Milestones

| # | Description | Status |
|---|-------------|--------|
| 0 | Repo scaffold, hello run | ✅ done |
| 1 | Maqasid proxy reward module + unit tests | ✅ done |
| 2 | `TownEnv` Gymnasium environment | 🔜 next |
| 3 | Baseline agents + evaluation script | 🔜 |
| 4 | PPO training + value logging + ethicality proxy script | 🔜 |
| 5 | Multi-agent extension (optional) | 🔜 |

## Conventions

- All source code lives under `src/valuefunctionethics/` (PEP 517 src layout).
- Tests go in `tests/` and use `pytest`.
- Experiment outputs (plots, CSVs) are gitignored — they live in `experiments/outputs/`.
- Read `docs/spec.md` before touching `envs/` and `docs/maqasid_proxies.md` before touching `rewards/`.

## Ethical Note

The five Maqasid dimensions are design objectives for a toy simulation reward function. No claim is made about real-world Islamic law.
