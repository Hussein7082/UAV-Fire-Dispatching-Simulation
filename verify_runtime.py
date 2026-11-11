"""Verify the saved runtime experiment without requiring identical wall times.

Run: python verify_runtime.py
Reconstructs all 60 measured instances and assignments, checks file hashes,
and independently recomputes summary statistics from raw nanosecond samples.
"""
import hashlib
import ast
import json
from pathlib import Path
import statistics

import benchmark_runtime as benchmark

np, pd, model = benchmark.np, benchmark.pd, benchmark.model
ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "runtime_results"


def main():
    config = json.loads((RESULTS / "runtime_config.json").read_text())
    for name, expected in config["sha256"].items():
        path = (ROOT / "data" / name) if name.endswith(".csv") else ROOT / name
        if name == "paper1_uav_fire_dispatching_training.py" and benchmark.sha256(path) != expected:
            # Later DQN-only corrections do not rewrite earlier timing evidence.
            # Retain the exact measured source and prove the timed functions and
            # their settings have not changed in the current implementation.
            measured = RESULTS / "benchmarked_source.py"
            assert benchmark.sha256(measured) == expected, "Measured source snapshot differs."
            def functions(source):
                return {node.name: ast.dump(node, include_attributes=False)
                        for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
            old, new = functions(measured.read_text()), functions(path.read_text())
            for function in ["haversine_km", "configure_missions", "compute_matrices", "solve", "rard_assignment"]:
                assert old[function] == new[function], f"Timed function changed: {function}; rerun benchmark."
            for key, value in config["model_parameters"].items():
                assert getattr(model, key) == value, f"Benchmark setting changed: {key}"
            path = measured
        assert benchmark.sha256(path) == expected, f"Source/input hash differs: {name}"
    # Instance generation originally used pandas' default parser.
    fires_default = pd.read_csv(ROOT / "data/preprocessed_fire_sample.csv")
    bases_default = pd.read_csv(ROOT / "data/uav_base_locations.csv")
    raw = pd.read_csv(RESULTS / "runtime_repetitions.csv")
    inputs = pd.read_csv(RESULTS / "runtime_instances.csv", float_precision="round_trip")
    summary = pd.read_csv(RESULTS / "runtime_summary.csv")
    assert len(raw) == len(config["sizes"]) * config["repetitions"]
    assert not raw.duplicated(["N", "M", "repetition"]).any()
    assert (raw.total_ns == raw.construction_ns + raw.assignment_ns).all()
    assert (raw[["total_ns", "construction_ns", "assignment_ns"]] > 0).all().all()
    for row in raw.itertuples():
        size_index = config["sizes"].index([row.N, row.M])
        seeds = np.random.SeedSequence([config["seed"], size_index,
                                       row.repetition + config["warmups_per_size"]]).generate_state(2)
        assert list(map(int, seeds)) == [row.scenario_seed, row.disturbance_seed]
        fcoords, bcoords, risk, fi, bi = benchmark.make_instance(
            fires_default, bases_default, row.N, row.M, row.scenario_seed,
            config["jitter_sd_degrees"])
        saved = inputs[(inputs.N == row.N) & (inputs.M == row.M) & (inputs.repetition == row.repetition)]
        for kind, coordinates, source_rows in [("fire", fcoords, fi), ("base", bcoords, bi)]:
            part = saved[saved.kind == kind].sort_values("index")
            np.testing.assert_array_equal(part.source_row, source_rows)
            np.testing.assert_array_equal(part[["latitude", "longitude"]].values, coordinates)
            if kind == "fire":
                np.testing.assert_array_equal(part.risk, risk)
        matrices = model.compute_matrices(fcoords[:, 0], fcoords[:, 1], risk,
                                         bcoords[:, 0], bcoords[:, 1],
                                         rng=np.random.default_rng(row.disturbance_seed), export=False)
        assignment = model.rard_assignment(risk, matrices)
        assert int(matrices["feasible"].sum()) == row.feasible_pairs
        assert len(assignment) == row.dispatch_count
        assert hashlib.sha256(json.dumps(assignment).encode()).hexdigest() == row.assignment_sha256
    for row in summary.itertuples():
        group = raw[(raw.N == row.N) & (raw.M == row.M)]
        assert len(group) == config["repetitions"] == row.repetitions
        expected = [statistics.mean(group.feasible_pairs),
                    statistics.mean(group.construction_ns) / 1e6,
                    statistics.mean(group.assignment_ns) / 1e6,
                    statistics.mean(group.total_ns) / 1e6,
                    statistics.stdev(group.total_ns) / 1e6]
        actual = [row.feasible_pairs_mean, row.construction_ms_mean,
                  row.assignment_ms_mean, row.total_ms_mean, row.total_ms_sd]
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    print("PASS: source and input hashes; all 60 instances, seeds, feasible counts, dispatches and assignment hashes; all six summary rows and sample standard deviations.")
    print("Wall-clock timings are observations, not values expected to reproduce exactly on another run.")


if __name__ == "__main__":
    main()
