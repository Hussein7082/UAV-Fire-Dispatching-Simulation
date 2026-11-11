"""Verify paired policy evaluation and replay the saved DQN without retraining.

Run from any directory: python verify_sequential.py
"""
from pathlib import Path
import hashlib
import json
import tempfile

import numpy as np
import pandas as pd
import paper1_uav_fire_dispatching_training as model

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "sequential_reference"


def read(name):
    return pd.read_csv(REFERENCE / name, float_precision="round_trip")


def canonical(frame):
    return frame.sort_values(["Policy", "Episode"]).reset_index(drop=True)


def main():
    config = json.loads((REFERENCE / "sequential_config.json").read_text())
    assert hashlib.sha256((ROOT / "paper1_uav_fire_dispatching_training.py").read_bytes()).hexdigest() == config["source_sha256"]
    assert hashlib.sha256((REFERENCE / "dqn_model.npz").read_bytes()).hexdigest() == config["checkpoint_sha256"]
    expected = read("sequential_policy_episodes.csv")
    schedule = read("sequential_episode_schedule.csv")
    summary = read("sequential_policy_comparison.csv").set_index("Policy")
    assert len(expected) == 400 and len(schedule) == 100
    assert not expected.duplicated(["Policy", "Episode"]).any()
    conditions = ["BaseIndex", "FireIndices", "Wind", "BaseLatitude", "BaseLongitude",
                  "FireCount", "InitialBatteryWh", "TotalFireRisk"]
    for _, group in expected.groupby("Episode"):
        assert len(group) == 4
        assert (group[conditions].nunique(dropna=False) == 1).all()
    for policy, group in expected.groupby("Policy"):
        assert np.isclose(summary.loc[policy, "Served Ratio (%)"],
                          100 * group.Served.sum() / group.FireCount.sum(), rtol=0, atol=1e-12)
        assert np.isclose(summary.loc[policy, "Average Mission Cost"], group.MissionCost.mean(), atol=1e-10)
        assert (group.Return == -group.MissionCost).all()
        pd.testing.assert_frame_equal(group[schedule.columns].reset_index(drop=True), schedule, check_exact=True)
    net = model.load_dqn_network(REFERENCE / "dqn_model.npz")
    with tempfile.TemporaryDirectory(prefix="rard_sequential_check_") as tmp:
        model.OUTPUT_DIR = tmp
        sample, cols, blat, blon = model.load_saved_inputs(ROOT / "data")
        flat, flon, risk = sample[cols["lat"]].values, sample[cols["lon"]].values, sample.FireRiskScore.values
        for name, values in [("fire_lat", flat), ("fire_lon", flon), ("risk", risk), ("base_lat", blat), ("base_lon", blon)]:
            assert hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest() == config["input_arrays_sha256"][name]
        # Reverse execution order after unrelated random draws. Neither can alter
        # the shared schedule or any policy outcome.
        np.random.seed(918273)
        np.random.random(10003)
        before = np.random.get_state()
        model.compare_sequential_policies(net, flat, flon, risk, blat, blon,
            policy_order=["DQN Policy", "Risk First", "Nearest First", "Random"])
        after = np.random.get_state()
        assert before[0] == after[0] and before[2:] == after[2:]
        np.testing.assert_array_equal(before[1], after[1])
        actual = pd.read_csv(Path(tmp) / "sequential_policy_episodes.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(canonical(actual), canonical(expected), check_exact=True)
        # The standalone greedy evaluation must use that same DQN schedule.
        result = model.evaluate_drl_policy(net, flat, flon, risk, blat, blon)
        assert np.isclose(result["Greedy Served Ratio (%)"], summary.loc["DQN Policy", "Served Ratio (%)"])
        assert np.isclose(result["Greedy Average Return"], summary.loc["DQN Policy", "Average Return"])
    print("PASS: all four policies share all 100 episode conditions; all 400 episode outcomes replay exactly from the checkpoint with reversed policy order and unrelated RNG draws; global RNG state is unchanged; summaries and standalone DQN evaluation agree.")


if __name__ == "__main__":
    main()
