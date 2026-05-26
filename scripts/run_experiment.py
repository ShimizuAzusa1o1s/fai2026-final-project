"""
n_estimators Experiment Script.

Sweeps over a list of ``n_estimators`` values for the Random Forest,
re-trains the model for each value, profiles ``FlatMC``'s simulation
throughput, and records the results to ``results/experiment_data.json``.

Usage:
    python scripts/run_experiment.py

Output:
    results/experiment_data.json  -- JSON array with one object per run:
        { "n": <int>, "throughput_calls": <int>, "simulations": <int> }
"""

import subprocess
import re
import os
import json

# Estimator counts to sweep
estimators = [5, 10, 20, 50, 75, 100]
results = []

for n in estimators:
    print(f"--- Testing n_estimators = {n} ---")

    # Re-train the RF model with the current n_estimators
    subprocess.run(
        ["python", "scripts/train_rf_model.py", "--estimators", str(n)],
        check=True,
        stdout=subprocess.DEVNULL
    )

    # Profile FlatMC and scrape the _predict_proba call count from cProfile output
    print(f"Profiling (n={n})...")
    profile_out = subprocess.run(
        ["python", "profile_flatmc.py"],
        capture_output=True,
        text=True
    ).stdout
    calls_match = re.search(
        r'\s+(\d+)\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+[\d.]+\s+.*_predict_proba',
        profile_out
    )
    throughput = int(calls_match.group(1)) if calls_match else 0

    res = {
        "n": n,
        "throughput_calls": throughput,
        "simulations": throughput // 90  # Approximate simulations per candidate
    }
    results.append(res)
    print(res)

os.makedirs("results", exist_ok=True)
with open("results/experiment_data.json", "w") as f:
    json.dump(results, f, indent=2)
