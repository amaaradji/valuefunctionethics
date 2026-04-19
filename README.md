# Value Function Ethics

A minimal, reproducible research artifact demonstrating how a **learned RL value function** can serve as a proxy for the **ethicality of actions** in a simulated multi-agent "second world," using **Maqasid al-Shari'ah-inspired** measurable proxies.

> **Disclaimer:** This is a simulation-based research tool. It does not issue religious rulings. All "maqasid" references are operationalised as empirical, measurable proxies for the purpose of studying learned value alignment.

---

## Conceptual Background

Islam provides an ethical framework grounded in objective metaphysical necessity. The cosmological argument establishes a single, absolutely simple, transcendent Creator as the logically necessary First Cause. In this framework, moral obligations derive directly from this foundation — they are objective truths discovered through reason and confirmed by revelation, not arbitrary commands or human inventions.

In this project, that framework is operationalised for machine learning. We use Maqasid al-Shari'ah (the objectives of Islamic law) to define a society-level reward signal over five dimensions — preservation of **life**, **intellect**, **wealth**, **family/lineage**, and **faith** (optional, handled carefully). A reinforcement learning agent learns a value function V(s) over aggregated society states. The learned V(s) then acts as an empirical ethicality proxy: the advantage A(s,a) = V(s') − V(s) tells us how much a proposed action moves the world toward or away from those objectives.

---

## Repository Structure

```
valuefunctionethics/
├── src/valuefunctionethics/
│   ├── envs/           # Gymnasium environments (TownEnv, ...)
│   ├── rewards/        # Maqasid proxy reward functions
│   ├── agents/         # Baseline + trained agents
│   └── utils/          # Seeding, logging, plotting helpers
├── experiments/
│   ├── configs/        # YAML/JSON experiment configs
│   ├── outputs/        # Plots and metric CSVs (gitignored)
│   ├── hello_run.py    # Milestone 0 smoke test
│   ├── run_train.py    # Milestone 4: RL training entry point
│   └── run_eval.py     # Milestone 3+: evaluation entry point
├── docs/
│   ├── spec.md                # Environment state/action/transition spec
│   ├── maqasid_proxies.md     # Precise proxy definitions
│   └── design_notes.md        # Architecture decisions and rationale
├── tests/              # pytest test suite
├── pyproject.toml
├── LICENSE             # MIT
└── README.md           # ← you are here
```

---

## Quickstart

### 1. Install

```bash
# Clone
git clone https://github.com/amaaradji/valuefunctionethics.git
cd valuefunctionethics

# Create a virtual environment (Python 3.11+)
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# Install the package + dependencies
pip install -e ".[dev]"
```

> **CPU-only PyTorch:** if you don't have a GPU, install torch separately first:
> ```bash
> pip install torch --index-url https://download.pytorch.org/whl/cpu
> pip install -e ".[dev]"
> ```

### 2. Smoke test (Milestone 0)

```bash
python experiments/hello_run.py
```

Expected output: versions printed, "All imports OK".

### 3. Run tests

```bash
pytest
```

---

## Milestones

| # | Description | Status |
|---|-------------|--------|
| 0 | Repo scaffold, dependency management, hello run | ✅ done |
| 1 | Spec docs + Maqasid proxy reward module + unit tests | ✅ done |
| 2 | `TownEnv` gymnasium environment with delayed dynamics | 🔜 next |
| 3 | Baseline agents + evaluation script | 🔜 |
| 4 | PPO training + value logging + ethicality proxy script | 🔜 |
| 5 | Multi-agent extension (optional) | 🔜 |

---

## Reproducing a Baseline Run

*(Available after Milestone 3 — commands will be added here)*

---

## Ethical Note

The five Maqasid dimensions are used purely as **design objectives** for a reward function in a toy simulation. Numbers in the simulation are invented for research purposes. No claim is made about what Islamic law requires or permits in any real-world situation.

---

## License

MIT — see [LICENSE](LICENSE).
