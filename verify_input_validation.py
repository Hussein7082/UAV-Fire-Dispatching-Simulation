"""Exercise malformed-input rejection and zero-dispatch scoring."""
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd

import paper1_uav_fire_dispatching_training as model

ROOT = Path(__file__).resolve().parent


def expect_error(call, message):
    try:
        call()
    except ValueError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected input error containing {message!r}")


def verify_snapshots(directory):
    sample = pd.read_csv(ROOT / "data/preprocessed_fire_sample.csv")
    bases = pd.read_csv(ROOT / "data/uav_base_locations.csv")
    inputs = directory / "inputs"
    inputs.mkdir()
    base_path = inputs / "uav_base_locations.csv"
    sample_path = inputs / "preprocessed_fire_sample.csv"
    bases.to_csv(base_path, index=False)
    for name, value, message in [
        ("InitialLatitude", 91, "valid latitude"),
        ("InitialLongitude", "invalid", "finite numbers"),
        ("DiscoveryAcres", -1, "nonnegative"),
        ("FireRiskScore", 1.5, "between 0 and 1"),
    ]:
        invalid = sample.copy()
        if isinstance(value, str):
            invalid[name] = invalid[name].astype(object)
        invalid.loc[0, name] = value
        invalid.to_csv(sample_path, index=False)
        expect_error(lambda: model.load_saved_inputs(inputs), message)
    sample.to_csv(sample_path, index=False)
    bases.loc[1, "UAV_ID"] = bases.loc[0, "UAV_ID"]
    bases.to_csv(base_path, index=False)
    expect_error(lambda: model.load_saved_inputs(inputs), "present and unique")
    bases = pd.read_csv(ROOT / "data/uav_base_locations.csv")
    bases.to_csv(base_path, index=False)
    loaded, _, latitude, longitude = model.load_saved_inputs(inputs)
    assert len(loaded) == 15 and len(latitude) == len(longitude) == 10


def verify_raw_and_checkpoints(directory):
    raw = directory / "raw.csv"
    raw.write_text("InitialLatitude,InitialLongitude,AcresNote\n34,-118,0.2\n")
    model.DATASET_PATH = str(raw)
    expect_error(model.load_dataset, "acreage column is required")
    raw.write_text("InitialLatitude,InitialLongitude,DiscoveryAcres\n34,-118,0.2\n34.2,-118.2,inf\n")
    sample, _, clustering = model.load_dataset()
    assert len(sample) == len(clustering) == 1
    assert sample["FireRiskScore"].iloc[0] == 0.05

    path = directory / "invalid.npz"
    np.savez(path, format_version=1)
    expect_error(lambda: model.load_dqn_network(path), "Missing DQN checkpoint fields")
    with np.load(ROOT / "sequential_reference/dqn_model.npz", allow_pickle=False) as checkpoint:
        fields = dict(checkpoint)
    fields["learning_rate"] = np.array(np.nan)
    np.savez(path, **fields)
    expect_error(lambda: model.load_dqn_network(path), "finite positive scalar")
    fields["learning_rate"] = np.array(model.DRL_LR)
    fields["W1"] = np.zeros((1, 1))
    np.savez(path, **fields)
    expect_error(lambda: model.load_dqn_network(path), "Invalid checkpoint array: W1")


def verify_zero_dispatch():
    risk = np.array([0.05, 1.0])
    matrices = model.compute_matrices(
        np.array([34.0, 34.2]), np.array([-118.0, -118.2]), risk,
        np.array([36.0, 36.0, 36.0]), np.array([-120.5, -120.4, -120.3]),
        export=False,
    )
    assignments = model.run_models(risk, matrices)
    assert all(not pairs for pairs in assignments.values())
    result = model.evaluate(assignments, risk, matrices)
    np.testing.assert_allclose(result["Risk-Weighted Delay"], np.dot(risk, matrices["deadline"]))
    np.testing.assert_allclose(result["Unserved Fire Risk"], risk.sum())
    assert result["Total Energy (Wh)"].eq(0).all()
    assert result["Total Robust Energy (Wh)"].eq(0).all()
    assert result["Mission Success Rate (%)"].eq(0).all()
    assert result["Average Response Delay (min)"].isna().all()
    assert result["Deadline Satisfaction Rate (%)"].isna().all()
    assert result["Average Remaining Battery (Wh)"].isna().all()


def main():
    with tempfile.TemporaryDirectory() as temp:
        directory = Path(temp)
        model.OUTPUT_DIR = temp
        verify_snapshots(directory)
        verify_raw_and_checkpoints(directory)
        verify_zero_dispatch()
    print("PASS: valid snapshot loading, invalid input rejection and all-unserved scoring.")


if __name__ == "__main__":
    main()
