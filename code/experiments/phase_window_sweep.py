"""Check the window-length effect on tempo matching using identical test samples."""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

import phase_tempo_benchmark as benchmark


def main():
    maximum_window = 75
    rows = []
    with threadpool_limits(limits=1):
        for kind, scenario in [("asymmetric", "slow_tempo"), ("asymmetric", "fast_tempo"),
                               ("weak_marker", "slow_tempo"), ("weak_marker", "constant")]:
            for seed in benchmark.SEEDS:
                benchmark.WINDOW = maximum_window
                signal, truth, speed = benchmark.test_record(seed, kind, scenario)
                for window in [12, 25, 50, 75]:
                    benchmark.WINDOW = window
                    grid, reference, bank, pca = benchmark.fit_model(seed, kind)
                    _, paths = benchmark.evaluate(grid, reference, bank, pca, signal, truth, 8)
                    for name in ["rigid_snapshot", "tempo_snapshot", "scalar_causal_dtw"]:
                        estimate = grid[paths[name][maximum_window - window:]]
                        expected = truth[maximum_window - 1:]
                        if len(estimate) != benchmark.SAMPLES:
                            raise AssertionError("Window sweep must score identical raw test endpoints")
                        rows.append({"waveform": kind, "scenario": scenario, "seed": seed,
                                     "window": window, "method": name, **benchmark.metrics(estimate, expected)})
    keys = sorted({(r["waveform"], r["scenario"], r["window"], r["method"]) for r in rows})
    summary = []
    for key in keys:
        subset = [r for r in rows if (r["waveform"], r["scenario"], r["window"], r["method"]) == key]
        summary.append({"waveform": key[0], "scenario": key[1], "window": key[2], "method": key[3],
                        **{name: float(np.mean([r[name] for r in subset]))
                           for name in ["mae_deg", "p95_deg", "over_90_deg", "cycle_drift"]}})
    result = {"configuration": {"scored_raw_endpoints": "74..1273 inclusive, identical across all windows",
                                 "windows": [12, 25, 50, 75], "p": 8, "seeds": list(benchmark.SEEDS)},
              "summary": summary, "runs": rows}
    (benchmark.OUT / "window_sweep.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in summary:
        if row["method"] == "tempo_snapshot":
            print(row["waveform"], row["scenario"], row["window"], round(row["mae_deg"], 3))


if __name__ == "__main__":
    main()
