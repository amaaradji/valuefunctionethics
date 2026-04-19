#!/usr/bin/env python
"""
hello_run.py — Milestone 0 smoke test.

Verifies that the package is installed and importable, then prints version info.

Usage:
    python experiments/hello_run.py
"""

import sys

print(f"Python {sys.version}")

import valuefunctionethics

print(f"valuefunctionethics {valuefunctionethics.__version__} — package OK")

import gymnasium
import numpy
import pandas
import matplotlib

print(f"gymnasium {gymnasium.__version__}")
print(f"numpy     {numpy.__version__}")
print(f"pandas    {pandas.__version__}")
print(f"matplotlib {matplotlib.__version__}")

try:
    import stable_baselines3

    print(f"stable-baselines3 {stable_baselines3.__version__}")
except ImportError:
    print("stable-baselines3 not installed yet (optional at Milestone 0)")

print("\nAll imports OK — scaffold is ready.")
