#!/usr/bin/env python3
"""
Experiment Runner: Test MCTS Variants Against Baselines

Runs each variant in a tournament against the TA baselines + the mcts_penalty baseline,
then summarizes results for comparison.

Usage:
    python scripts/run_mcts_experiments.py
"""

import json
import subprocess
import sys
import os
from datetime import datetime

# Define experiments: (variant_name, module_path, class_name, extra_args)
EXPERIMENTS = [
    {
        "name": "baseline_mcts_penalty",
        "module": "src.players.b12705048.mcts_penalty",
        "class": "MCTS",
        "args": {},
        "label": "MCTSbase",
    },
    {
        "name": "v1_shallow_rollout",
        "module": "src.players.b12705048.mcts_v1_shallow",
        "class": "MCTS",
        "args": {},
        "label": "V1shallow",
    },
    {
        "name": "v2_progressive_elim",
        "module": "src.players.b12705048.mcts_v2_progressive",
        "class": "MCTS",
        "args": {},
        "label": "V2progrs",
    },
    {
        "name": "v3_variance_aware",
        "module": "src.players.b12705048.mcts_v3_variance",
        "class": "MCTS",
        "args": {},
        "label": "V3var",
    },
]

# Base tournament config
BASE_CONFIG = {
    "baselines": [
        ["src.players.TA.public_baselines1", "Baseline1", {}, "B1"],
        ["src.players.TA.public_baselines1", "Baseline2", {}, "B2"],
        ["src.players.TA.public_baselines1", "Baseline3", {}, "B3"],
        ["src.players.TA.public_baselines1", "Baseline4", {}, "B4"],
        ["src.players.TA.public_baselines1", "Baseline5", {}, "B5"],
    ],
    "engine": {
        "n_players": 4,
        "n_rounds": 10,
        "verbose": False,
        "timeout": 1.0,
        "timeout_buffer": 5.0,
    },
    "tournament": {
        "type": "random_partition",
        "duplication_mode": "cycle",
        "num_games_per_player": 100,
        "num_workers": 10,
        "scoring": {
            "baseline_upper_pct": 0.8,
            "baseline_lower_pct": 0.2,
            "score_at_upper_pct": 80,
            "score_at_lower_pct": 20,
        },
    },
}


def run_experiment(experiment):
    """Run a single experiment and return its result file path."""
    name = experiment["name"]
    config = json.loads(json.dumps(BASE_CONFIG))  # deep copy

    # Add the test player
    config["players"] = [
        [
            experiment["module"],
            experiment["class"],
            experiment["args"],
            experiment["label"],
        ]
    ]

    # Write temp config
    config_path = f"configs/tournament/temp_exp_{name}.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Running experiment: {name}")
    print(f"{'='*60}")

    # Run tournament
    result = subprocess.run(
        [sys.executable, "run_tournament.py", "--config", config_path],
        capture_output=False,
    )

    if result.returncode != 0:
        print(f"ERROR: Experiment {name} failed with return code {result.returncode}")
        return None

    # Find the most recent result file matching this config
    results_dir = "results/tournament"
    result_files = sorted(
        [f for f in os.listdir(results_dir) if f.endswith(f"temp_exp_{name}.json")],
        reverse=True,
    )

    if result_files:
        return os.path.join(results_dir, result_files[0])
    return None


def extract_results(result_path, experiment_name):
    """Extract key metrics from a tournament result file."""
    with open(result_path) as f:
        data = json.load(f)

    standings = data.get("standings", [])
    for s in standings:
        if not s.get("is_baseline", True):
            return {
                "name": experiment_name,
                "avg_score": s["avg_score"],
                "avg_rank": s["avg_rank"],
                "est_elo": s["est_elo"],
                "calibrated_score": s.get("calibrated_score", None),
                "timeout_count": s.get("timeout_count", 0),
                "games_played": s.get("games_played", 0),
            }
    return None


def main():
    print("=" * 60)
    print("MCTS Variant Experiment Suite")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    for exp in EXPERIMENTS:
        result_path = run_experiment(exp)
        if result_path:
            metrics = extract_results(result_path, exp["name"])
            if metrics:
                results.append(metrics)
                print(f"\n  >> {exp['name']}: avg_rank={metrics['avg_rank']:.3f}, "
                      f"avg_score={metrics['avg_score']:.2f}, "
                      f"elo={metrics['est_elo']:.0f}, "
                      f"cal_score={metrics['calibrated_score']:.1f}, "
                      f"timeouts={metrics['timeout_count']}")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Variant':<25} {'Avg Rank':<10} {'Avg Score':<11} {'Elo':<8} {'Cal Score':<10} {'TOs':<5}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["avg_rank"]):
        cal = f"{r['calibrated_score']:.1f}" if r['calibrated_score'] is not None else "-"
        print(f"{r['name']:<25} {r['avg_rank']:<10.3f} {r['avg_score']:<11.2f} "
              f"{r['est_elo']:<8.0f} {cal:<10} {r['timeout_count']:<5}")

    print("-" * 80)
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save summary
    summary_path = f"results/tournament/experiment_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
