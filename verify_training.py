"""Check training diagnostics and the implemented reward accounting.

Run: python verify_training.py
No training or network update is performed by this verifier.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import paper1_uav_fire_dispatching_training as model

ROOT = Path(__file__).resolve().parent
REFERENCE = ROOT / "sequential_reference"


def main():
    hist = pd.read_csv(REFERENCE / "drl_training_history.csv", float_precision="round_trip")
    diag = pd.read_csv(REFERENCE / "drl_training_diagnostics.csv", float_precision="round_trip")
    summary = json.loads((REFERENCE / "drl_training_summary.json").read_text())
    assert len(hist) == len(diag) == 800
    np.testing.assert_array_equal(diag.Episode, hist.Episode)
    np.testing.assert_array_equal(diag.ReportedReturn, hist["Cumulative Reward"])
    np.testing.assert_array_equal(diag.Steps, diag[["FireActions", "ChargeActions", "RetargetActions", "ReturnActions"]].sum(axis=1))
    np.testing.assert_allclose(diag.ReportedReturn,
        -diag.ActionCost-model.C_PENALTY_RL*(diag.TotalFireRisk-diag.ServedRisk), atol=1e-10, rtol=0)
    np.testing.assert_allclose(diag.NoServiceUpperBound, -model.C_PENALTY_RL*diag.TotalFireRisk, atol=1e-10, rtol=0)
    no_service = diag.Served.eq(0)
    below = diag.ReportedReturn.le(diag.NoServiceUpperBound + 1e-8)
    assert int(no_service.sum()) == summary["no_service_count"] == 288
    assert int(below.sum()) == summary["at_or_below_reference_count"] == 307
    assert int((below & ~no_service).sum()) == summary["served_but_at_or_below_reference_count"] == 19
    assert int((no_service & diag.FireActions.eq(0) & diag.ChargeActions.eq(0)).sum()) == summary["only_retarget_or_return_no_service_count"] == 64
    cost = diag.loc[no_service, "ActionCost"].to_numpy()
    for key, value in [
        ("no_service_action_cost_median", np.median(cost)),
        ("no_service_action_cost_p90", np.percentile(cost, 90)),
        ("return_iqr", np.subtract(*np.percentile(hist["Cumulative Reward"], [75,25]))),
        ("final_moving_average", hist["Cumulative Reward"].iloc[-25:].mean()),
        ("second_half_moving_average_slope", np.polyfit(hist.Episode.iloc[400:], hist["Moving Average"].iloc[400:], 1)[0]),
    ]:
        assert np.isclose(summary[key], value, atol=1e-10, rtol=0), key
    np.testing.assert_allclose(hist["Moving Average"], hist["Cumulative Reward"].rolling(25,min_periods=1).mean(), atol=1e-10, rtol=0)

    # Hand-computable zero-distance cases exercise service, return, retargeting
    # and charging; episode accounting must include the unserved-risk term.
    def environment():
        env = model.FireDispatchEnv(np.array([34.0]), np.array([-118.0]), np.array([1.0]), 34.0, -118.0)
        env.reset(wind=0.1)
        return env
    env = environment(); _, reward, done = env.step(0)
    expected_service_cost = 30.0 + 0.01*(220.0*0.5*1.1)
    assert done and env.served == 1
    assert np.isclose(reward - 1000.0, -expected_service_cost)
    env = environment(); _, reward, done = env.step(3)
    assert done and env.served == 0 and reward-1000.0 == -1000.0
    env = environment(); _, r1, _ = env.step(2); _, r2, done = env.step(3)
    assert done and r1+r2-1000.0 == -1000.5
    env = environment(); _, r1, _ = env.step(1); _, r2, done = env.step(3)
    assert done and r1+r2-1000.0 == -1030.0
    # A feasible low-risk mission can cost more than its 50-unit service bonus.
    sample = pd.read_csv(ROOT / "data/preprocessed_fire_sample.csv")
    bases = pd.read_csv(ROOT / "data/uav_base_locations.csv")
    fire, base = sample.iloc[10], bases.iloc[2]
    env = model.FireDispatchEnv(np.array([fire.InitialLatitude]), np.array([fire.InitialLongitude]),
                                np.array([fire.FireRiskScore]), base.BaseLatitude, base.BaseLongitude)
    env.reset(wind=0.0)
    assert env._feasible(0) and fire.FireRiskScore == 0.05
    distance = float(model.haversine_km(base.BaseLatitude,base.BaseLongitude,fire.InitialLatitude,fire.InitialLongitude))
    travel_minutes = distance*1000/20/60
    full_cost = 2*travel_minutes+30+0.01*(2*300*travel_minutes/60+110)
    _, reward, done = env.step(0)
    assert done and np.isclose(reward, 50-full_cost) and full_cost > 50
    # Algebraic illustration only: equal undiscounted totals do not imply equal
    # discounted objectives when the service bonus arrives at different steps.
    assert sum([1000,0])-1000 == sum([0,1000])-1000 == 0
    assert np.dot([1,0.95],[1000,0]) != np.dot([1,0.95],[0,1000])
    print("PASS: all 800 diagnostic records, reward identities, action counts, summary statistics, moving averages, hand-computable transitions, low-risk service-cost counterexample and discounted/undiscounted distinction.")


if __name__ == "__main__":
    main()
