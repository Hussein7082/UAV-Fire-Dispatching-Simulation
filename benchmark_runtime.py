"""Measure the supplied RARD implementation; export every sample and Figure 12.

Run from this directory:
    python benchmark_runtime.py --input-dir data --output-dir runtime_results
Regenerate the table/figure from saved measurements without timing again:
    python benchmark_runtime.py --output-dir runtime_results --plot-only

Timings are machine dependent. This script does not reproduce historical timings.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
from datetime import datetime, timezone

# Set before NumPy/SciPy import. CPU affinity is separately recorded below.
for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pandas as pd
import scipy
import matplotlib
import matplotlib.pyplot as plt
import paper1_uav_fire_dispatching_training as model

SIZES = ((15, 10), (50, 20), (100, 20), (100, 50), (200, 50), (400, 100))
MODEL_PARAMETERS = (
    "UAV_SPEED_MPS", "BATTERY_CAPACITY_WH", "FLIGHT_POWER_W", "HOVER_POWER_W",
    "HOVER_TIME_MIN", "SAFETY_RESERVE_RATIO", "ENERGY_UNCERTAINTY_RATIO",
    "ENERGY_DISTURBANCE_MAX", "WIND_PENALTY_MAX", "BASE_DEADLINE_MIN",
    "MIN_DEADLINE_MIN", "DEADLINE_SENSITIVITY", "ALPHA_DELAY", "BETA_ENERGY",
    "DELTA_RISK_REWARD", "BIG_M", "REGION_BOUNDS",
)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def set_affinity(core):
    info = {"requested": core, "before": None, "after": None, "error": None}
    if hasattr(os, "sched_getaffinity"):
        info["before"] = sorted(os.sched_getaffinity(0))
        try:
            if core != "none":
                selected = info["before"][0] if core == "auto" else int(core)
                os.sched_setaffinity(0, {selected})
        except (OSError, ValueError) as exc:
            info["error"] = str(exc)
        info["after"] = sorted(os.sched_getaffinity(0))
    else:
        info["error"] = "CPU affinity is unavailable through Python on this platform."
    return info


def machine_info(affinity):
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    return {
        "utc_time": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu_model": cpu, "logical_cpu_count": os.cpu_count(), "affinity": affinity,
        "python": platform.python_version(), "numpy": np.__version__,
        "pandas": pd.__version__, "scipy": scipy.__version__,
        "matplotlib": matplotlib.__version__,
        "thread_environment": {k: os.environ[k] for k in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")},
        "timer": vars(time.get_clock_info("perf_counter")),
    }


def make_instance(fires, bases, n, m, scenario_seed, jitter_sd):
    rng = np.random.default_rng(scenario_seed)
    original = (n, m) == (len(fires), len(bases))
    fi = np.arange(n) if original else rng.integers(len(fires), size=n)
    bi = np.arange(m) if original else rng.integers(len(bases), size=m)
    fcoords = fires.iloc[fi][["InitialLatitude", "InitialLongitude"]].to_numpy(copy=True)
    bcoords = bases.iloc[bi][["BaseLatitude", "BaseLongitude"]].to_numpy(copy=True)
    if not original:
        for coords in (fcoords, bcoords):
            coords += rng.normal(0.0, jitter_sd, size=coords.shape)
            bounds = model.REGION_BOUNDS
            coords[:, 0] = np.clip(coords[:, 0], bounds["lat_min"], bounds["lat_max"])
            coords[:, 1] = np.clip(coords[:, 1], bounds["lon_min"], bounds["lon_max"])
    risk = fires.iloc[fi]["FireRiskScore"].to_numpy(copy=True)
    return fcoords, bcoords, risk, fi, bi


def timed_round(fcoords, bcoords, risk, disturbance_seed):
    rng = np.random.default_rng(disturbance_seed)
    # Argument preparation, instance generation and RNG initialization are excluded.
    flat, flon = fcoords[:, 0], fcoords[:, 1]
    blat, blon = bcoords[:, 0], bcoords[:, 1]
    start = time.perf_counter_ns()
    matrices = model.compute_matrices(flat, flon, risk, blat, blon, rng=rng, export=False)
    middle = time.perf_counter_ns()
    assignment = model.rard_assignment(risk, matrices)
    end = time.perf_counter_ns()
    # Validation and serialization are excluded from both timed stages.
    assert len({i for i, _ in assignment}) == len(assignment)
    assert len({j for _, j in assignment}) == len(assignment)
    assert all(matrices["feasible"][i, j] for i, j in assignment)
    return {
        "construction_ns": middle - start, "assignment_ns": end - middle,
        "total_ns": end - start, "feasible_pairs": int(matrices["feasible"].sum()),
        "dispatch_count": len(assignment),
        "assignment_sha256": hashlib.sha256(json.dumps(assignment).encode()).hexdigest(),
    }


def summarize_and_plot(output):
    raw = pd.read_csv(output / "runtime_repetitions.csv")
    records = []
    for (n, m), group in raw.groupby(["N", "M"], sort=False):
        assert (group.total_ns == group.construction_ns + group.assignment_ns).all()
        records.append({
            "N": int(n), "M": int(m), "pairs": int(n * m),
            "repetitions": len(group), "feasible_pairs_mean": group.feasible_pairs.mean(),
            "construction_ms_mean": group.construction_ns.mean() / 1e6,
            "assignment_ms_mean": group.assignment_ns.mean() / 1e6,
            "total_ms_mean": group.total_ns.mean() / 1e6,
            "total_ms_sd": group.total_ns.std(ddof=1) / 1e6,
        })
    summary = pd.DataFrame(records)
    summary.to_csv(output / "runtime_summary.csv", index=False)
    lines = []
    for row in summary.itertuples():
        lines.append(
            f"{row.N} & {row.M} & {row.pairs} & {row.feasible_pairs_mean:.1f} & "
            f"{row.construction_ms_mean:.3f} & {row.assignment_ms_mean:.3f} & "
            f"${row.total_ms_mean:.3f} \\pm {row.total_ms_sd:.3f}$ \\\\"
        )
    (output / "runtime_table_rows.tex").write_text("\n".join(lines) + "\n")
    fig, axes = plt.subplots(2, 1, figsize=(7.0, 6.4), constrained_layout=True)
    ax = axes[0]
    ax.errorbar(summary.pairs, summary.total_ms_mean, yerr=summary.total_ms_sd,
                marker="o", color="#222222", capsize=3, label="Complete round (mean ± SD)")
    ax.plot(summary.pairs, summary.construction_ms_mean, "s--", color="#0072B2", label="Matrix construction")
    ax.plot(summary.pairs, summary.assignment_ms_mean, "^:", color="#D55E00", label="Assignment")
    ax.set(xscale="log", yscale="log", xlabel="Candidate pairs (N × M)", ylabel="Runtime (ms)")
    ax.set_title("(a) Measured runtime", loc="left", fontsize=12)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    ax = axes[1]
    x = np.arange(len(summary))
    construction_share = 100 * summary.construction_ms_mean / summary.total_ms_mean
    assignment_share = 100 - construction_share
    ax.bar(x, construction_share, color="#0072B2", label="Matrix construction")
    ax.bar(x, assignment_share, bottom=construction_share, color="#D55E00", label="Assignment")
    for i, share in enumerate(construction_share):
        ax.text(i, 50, f"{share:.1f}%", ha="center", color="white", fontsize=10)
    ax.set_xticks(x, [f"{r.N} × {r.M}" for r in summary.itertuples()])
    ax.set(xlabel="Incidents × UAVs (N × M)", ylabel="Share of mean runtime (%)", ylim=(0, 108))
    ax.set_title("(b) Runtime by stage", loc="left", fontsize=12)
    ax.legend(fontsize=9, loc="lower right")
    fig.savefig(output / "fig12_scalability_runtime.png", dpi=300)
    fig.savefig(output / "fig12_scalability_runtime.pdf")
    plt.close(fig)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path("runtime_results"))
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--jitter-sd-deg", type=float, default=0.01)
    parser.add_argument("--core", default="auto", help="auto pins one available Linux CPU; none disables pinning")
    parser.add_argument("--plot-only", action="store_true")
    args = parser.parse_args()
    output = args.output_dir
    if args.plot_only:
        print(summarize_and_plot(output).to_string(index=False))
        return
    if args.repetitions < 2 or args.warmups < 0 or args.jitter_sd_deg < 0:
        parser.error("Require repetitions >= 2, warmups >= 0 and jitter SD >= 0.")
    if output.exists() and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory to retain earlier measurements.")
    output.mkdir(parents=True, exist_ok=True)
    affinity = set_affinity(args.core)
    input_paths = [args.input_dir / "preprocessed_fire_sample.csv", args.input_dir / "uav_base_locations.csv"]
    fires, bases = (pd.read_csv(path) for path in input_paths)
    assert len(fires) == 15 and len(bases) == 10, "Benchmark expects the bundled 15-incident/10-base snapshots."
    config = {
        "benchmark_version": 1, "machine": machine_info(affinity), "sizes": SIZES,
        "seed": args.seed, "repetitions": args.repetitions, "warmups_per_size": args.warmups,
        "jitter_sd_degrees": args.jitter_sd_deg,
        "instance_protocol": "15x10 uses exact saved rows. Larger sizes independently sample incidents and bases with replacement; add independent zero-mean Gaussian coordinate jitter, then clip to REGION_BOUNDS. Retain sampled incident risk scores.",
        "seed_protocol": "SeedSequence([seed, size_index, repetition+warmups]) produces scenario and disturbance uint32 seeds. Warmup repetitions use negative indices.",
        "construction_boundary": "Exact compute_matrices(export=False), including nested Haversine loops, wind and realized disturbance draws, nominal/actual/protected energy and both feasibility matrices.",
        "assignment_boundary": "Exact rard_assignment, including cost construction, dense infeasibility mask, scipy linear_sum_assignment and extraction of valid pairs.",
        "excluded": "Imports, data loading, scenario generation, RNG creation, validation, CSV/figure writing, DQN training, communications and vehicle actuation.",
        "standard_deviation": "Sample SD (ddof=1) across measured repetitions; includes runtime and scenario variation.",
        "model_parameters": {k: getattr(model, k) for k in MODEL_PARAMETERS},
        "sha256": {p.name: sha256(p) for p in [*input_paths, Path(model.__file__), Path(__file__)]},
    }
    (output / "runtime_config.json").write_text(json.dumps(config, indent=2) + "\n")
    raw, coordinates = [], []
    for size_index, (n, m) in enumerate(SIZES):
        for rep in range(-args.warmups, args.repetitions):
            seeds = np.random.SeedSequence([args.seed, size_index, rep + args.warmups]).generate_state(2)
            scenario_seed, disturbance_seed = map(int, seeds)
            fcoords, bcoords, risk, fi, bi = make_instance(fires, bases, n, m, scenario_seed, args.jitter_sd_deg)
            measurement = timed_round(fcoords, bcoords, risk, disturbance_seed)
            if rep < 0:
                continue
            raw.append({"N": n, "M": m, "repetition": rep, "scenario_seed": scenario_seed,
                        "disturbance_seed": disturbance_seed, **measurement})
            for kind, coords, source_rows in [("fire", fcoords, fi), ("base", bcoords, bi)]:
                for index, ((lat, lon), source_row) in enumerate(zip(coords, source_rows)):
                    coordinates.append({"N": n, "M": m, "repetition": rep, "kind": kind,
                                        "index": index, "source_row": int(source_row),
                                        "latitude": lat, "longitude": lon,
                                        "risk": float(risk[index]) if kind == "fire" else None})
        print(f"Completed {n} × {m}", flush=True)
    pd.DataFrame(raw).to_csv(output / "runtime_repetitions.csv", index=False)
    pd.DataFrame(coordinates).to_csv(output / "runtime_instances.csv", index=False)
    print(summarize_and_plot(output).to_string(index=False))


if __name__ == "__main__":
    main()
