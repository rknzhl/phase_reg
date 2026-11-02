"""Compare frozen Project77-style phase matching with tempo matching and causal DTW."""

import ast
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/tmp/project77-matplotlib")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "output" / "phase_tempo_benchmark"
TAU = 2 * np.pi
PERIOD = 100
WINDOW = 50
GRID = 180
SAMPLES = 1200
SPEEDS = np.round(np.arange(0.5, 1.801, 0.05), 8)
SEEDS = range(5)
NOISE = 0.04


def notebook_kernel():
    notebook = ROOT / "code" / "main" / "07_paper_algorithm.ipynb"
    definitions = []
    for cell in json.loads(notebook.read_text())["cells"]:
        if cell["cell_type"] == "code":
            definitions.extend(
                node for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.FunctionDef)
                and node.name in {"phase_distance", "kernel_weights"}
            )
    namespace = {"np": np}
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(notebook), "exec"), namespace)
    return namespace["kernel_weights"]


KERNEL = notebook_kernel()


def waveform(phase, kind="asymmetric"):
    if kind == "asymmetric":
        return np.cos(phase) + 0.5 * np.sin(2 * phase) + 0.25 * np.cos(3 * phase)
    if kind == "weak_marker":
        return np.cos(2 * phase) + 0.12 * np.sin(phase) + 0.08 * np.cos(3 * phase)
    if kind == "symmetric":
        return np.cos(2 * phase)
    raise ValueError(kind)


def fit_model(seed, kind):
    rng = np.random.default_rng(10000 + seed)
    calibration_phase = TAU * np.arange(PERIOD) / PERIOD
    calibration = waveform(calibration_phase, kind) + rng.normal(0, NOISE, PERIOD)
    grid = TAU * np.arange(GRID) / GRID
    weights = KERNEL(grid, calibration_phase, 0.003, 1)
    reference = np.einsum("ij,j->i", weights, calibration)
    if not np.isfinite(reference).all():
        raise FloatingPointError("Nonfinite calibration reference")
    interpolate = CubicSpline(np.r_[grid, TAU], np.r_[reference, reference[0]], bc_type="periodic")
    lags = np.arange(WINDOW - 1, -1, -1)
    phase = grid[:, None, None] - TAU / PERIOD * SPEEDS[None, :, None] * lags
    bank = interpolate(phase % TAU)
    pca = PCA(n_components=WINDOW, svd_solver="full").fit(bank.reshape(-1, WINDOW))
    return grid, reference, bank, pca


def test_record(seed, kind, scenario):
    rng = np.random.default_rng(20000 + seed)
    count = SAMPLES + WINDOW - 1
    t = np.arange(count)
    if scenario == "constant":
        speed = np.ones(count)
    elif scenario == "constant_unseen":
        speed = np.full(count, 1.575)
    elif scenario == "slow_tempo":
        speed = 1.15 + 0.48 * np.sin(TAU * t / 700 + 0.7)
    elif scenario == "fast_tempo":
        speed = 1.15 + 0.48 * np.sin(TAU * t / 100 + 0.7)
    elif scenario == "shape_change":
        speed = np.ones(count)
    elif scenario == "tempo_and_gain":
        speed = 1.15 + 0.48 * np.sin(TAU * t / 700 + 0.7)
    else:
        raise ValueError(scenario)
    phase = rng.uniform(0, TAU) + np.r_[0, np.cumsum(TAU / PERIOD * speed[1:])]
    clean = waveform(phase, kind)
    if scenario == "shape_change":
        clean = clean + 0.30 * np.sin(3 * phase + 0.6)
    if scenario == "tempo_and_gain":
        clean = clean * speed**2
    signal = clean + rng.normal(0, NOISE, count)
    return signal, phase, speed


def causal_path(emission, penalty):
    costs = emission[0].copy()
    path = np.empty(len(emission), dtype=int)
    path[0] = np.argmin(costs)
    increments = np.arange(5)
    step_penalty = penalty * (increments - GRID / PERIOD) ** 2
    for t in range(1, len(emission)):
        transitions = np.stack([
            np.roll(costs, increment) + step_penalty[increment]
            for increment in increments
        ])
        costs = emission[t] + np.min(transitions, axis=0)
        costs -= costs.min()
        path[t] = np.argmin(costs)
    return path


def metrics(estimate, truth):
    error = np.abs(np.angle(np.exp(1j * (estimate - truth))))
    return {
        "mae_deg": float(np.rad2deg(error.mean())),
        "p95_deg": float(np.rad2deg(np.quantile(error, 0.95))),
        "within_15_deg": float(np.mean(error <= np.deg2rad(15))),
        "over_90_deg": float(np.mean(error > np.pi / 2)),
        "cycle_drift": float(abs((np.unwrap(estimate)[-1] - np.unwrap(estimate)[0]
                                  - truth[-1] + truth[0]) / TAU)),
        "coverage": 1.0,
    }


def evaluate(grid, reference, bank, pca, signal, truth, p):
    windows = np.lib.stride_tricks.sliding_window_view(signal, WINDOW)
    points = np.einsum("ij,kj->ik", windows - pca.mean_, pca.components_[:p])
    projected = np.einsum("ij,kj->ik", bank.reshape(-1, WINDOW) - pca.mean_, pca.components_[:p])
    if not np.isfinite(points).all() or not np.isfinite(projected).all():
        raise FloatingPointError("Nonfinite PCA projection")
    distances = cdist(points, projected, metric="sqeuclidean").reshape(len(points), GRID, len(SPEEDS)) / WINDOW
    rigid = distances[:, :, np.flatnonzero(SPEEDS == 1)[0]]
    tempo = distances.min(axis=2)
    scalar_emission = (signal[:, None] - reference[None, :]) ** 2
    paths = {
        "rigid_snapshot": rigid.argmin(axis=1),
        "tempo_snapshot": tempo.argmin(axis=1),
        "rigid_causal": causal_path(rigid, 0.0005),
        "tempo_causal": causal_path(tempo, 0.0005),
        "scalar_causal_dtw": causal_path(scalar_emission, 0.0005)[WINDOW - 1:],
    }
    expected = truth[WINDOW - 1:]
    return {name: metrics(grid[path], expected) for name, path in paths.items()}, paths


def main():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    example = None
    scenarios = ["constant", "constant_unseen", "slow_tempo", "fast_tempo", "shape_change", "tempo_and_gain"]
    with threadpool_limits(limits=1):
        for kind in ["asymmetric", "weak_marker", "symmetric"]:
            for seed in SEEDS:
                grid, reference, bank, pca = fit_model(seed, kind)
                selected = scenarios if kind == "asymmetric" else ["constant", "slow_tempo"]
                for scenario in selected:
                    signal, truth, speed = test_record(seed, kind, scenario)
                    dimensions = [2, 4, 8, 16] if scenario == "slow_tempo" and kind != "asymmetric" else [8]
                    for p in dimensions:
                        result, paths = evaluate(grid, reference, bank, pca, signal, truth, p)
                        for name, values in result.items():
                            rows.append({"waveform": kind, "scenario": scenario, "seed": seed, "p": p,
                                         "method": name, **values,
                                         "pca_variance": float(pca.explained_variance_ratio_[:p].sum())})
                        if kind == "asymmetric" and seed == 0 and scenario == "slow_tempo":
                            example = (grid, signal, truth, speed, paths)
                print(f"finished {kind} seed={seed}", flush=True)
    keys = sorted({(r["waveform"], r["scenario"], r["p"], r["method"]) for r in rows})
    summary = []
    for key in keys:
        subset = [r for r in rows if (r["waveform"], r["scenario"], r["p"], r["method"]) == key]
        summary.append({"waveform": key[0], "scenario": key[1], "p": key[2], "method": key[3],
                        "seeds": len(subset), **{
                            name: float(np.mean([r[name] for r in subset]))
                            for name in ["mae_deg", "p95_deg", "within_15_deg", "over_90_deg", "cycle_drift", "pca_variance"]
                        }})
    result = {
        "configuration": {"period_samples": PERIOD, "window_samples": WINDOW, "phase_grid": GRID,
                          "test_samples": SAMPLES, "noise_std": NOISE, "seeds": list(SEEDS),
                          "speeds": SPEEDS.tolist(), "nw_bandwidth": 0.003,
                          "pca_training": "tempo-augmented circular windows synthesized from one calibration cycle",
                          "phase_target": "known generator phase at the END of each causal window",
                          "decoder": "frozen models; no test-time fitting or per-test phase alignment",
                          "dtw": "cyclic forward min-sum alignment from the FIRST raw sample, increments 0..4 bins, causal endpoint estimate",
                          "limitations": ["synthetic benchmark, not human accuracy", "given calibration cycle boundaries",
                                          "constant-speed window bank", "no proof of novelty", "not the full notebook07 estimator",
                                          "causal endpoint estimates need not form one monotone path"]},
        "runtime_seconds": time.perf_counter() - started,
        "summary": summary, "runs": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    lines = ["Project77 causal phase/tempo benchmark", "", json.dumps(result["configuration"], indent=2), "",
             "Mean errors across five independent calibration/test noise seeds. No test phase alignment.",
             "The same PCA basis, reference, raw windows, noise and phase grid are used for rigid/tempo comparisons.",
             "The existing notebook kernel is reused; matching is an isolated frozen-model ablation.", "",
             "waveform scenario p method MAE_deg P95_deg within15 over90 PCA_variance"]
    for r in summary:
        lines.append(f"{r['waveform']:12s} {r['scenario']:16s} {r['p']:2d} {r['method']:19s} "
                     f"{r['mae_deg']:7.3f} {r['p95_deg']:7.3f} {r['within_15_deg']:.3f} "
                     f"{r['over_90_deg']:.3f} {r['pca_variance']:.6f}")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n")
    grid, signal, truth, speed, paths = example
    t = np.arange(SAMPLES) / PERIOD
    fig, axes = plt.subplots(3, 1, figsize=(11, 8))
    axes[0].plot(t, speed[WINDOW - 1:], color="black")
    axes[0].set_ylabel("Relative cadence")
    for name in ["rigid_snapshot", "tempo_snapshot", "scalar_causal_dtw"]:
        error = np.rad2deg(np.abs(np.angle(np.exp(1j * (grid[paths[name]] - truth[WINDOW - 1:])))))
        axes[1].plot(t, error, label=name, alpha=0.85)
    axes[1].set_ylabel("Phase error, degrees")
    axes[1].legend(loc="upper right")
    methods = ["rigid_snapshot", "tempo_snapshot", "tempo_causal", "scalar_causal_dtw"]
    colors = ["#8d6b45", "#7465a8", "#71804b", "#3c7185"]
    width = 0.18
    for j, (name, color) in enumerate(zip(methods, colors)):
        values = [next(r["mae_deg"] for r in summary if r["waveform"] == "asymmetric"
                       and r["scenario"] == scenario and r["p"] == 8 and r["method"] == name)
                  for scenario in scenarios]
        axes[2].bar(np.arange(len(scenarios)) + j * width, values, width=width, color=color, label=name)
    axes[2].set_xticks(np.arange(len(scenarios)) + 1.5 * width,
                       ["same speed", "new speed", "slow tempo", "fast tempo", "shape change", "tempo + gain"])
    axes[2].set_ylabel("Mean phase error, degrees")
    axes[2].legend(fontsize=8)
    axes[0].set_title("One-cycle calibration: frozen Hankel-PCA matching versus causal time alignment")
    axes[0].set_xlim(0, t[-1])
    axes[1].set_xlim(0, t[-1])
    axes[2].set_xlim(-0.5, len(scenarios))
    axes[1].set_xlabel("Time in calibration periods")
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png", dpi=170)
    plt.close(fig)
    print(f"saved {OUT}; runtime={result['runtime_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
