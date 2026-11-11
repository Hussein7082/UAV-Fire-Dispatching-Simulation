# Risk-Aware Robust Dispatching simulation

The script runs the single-round and paired randomized experiments for the study “Risk-Aware Robust UAV Dispatching for Early Fire Emergency Response under Energy and Delay Constraints.”

## Download and extract the data

Download the repository using **Code → Download ZIP** on GitHub, then extract the downloaded repository ZIP. You can also clone the repository with Git.

[Public repository](https://github.com/Hussein7082/UAV-Fire-Dispatching-Simulation)

The five data archives in the repository must be extracted before running the simulation or verification scripts. Open a terminal in the folder containing this README and the Python scripts, then run:

```sh
python -m zipfile -e data.zip .
python -m zipfile -e original_results.zip .
python -m zipfile -e reference_results.zip .
python -m zipfile -e runtime_results.zip .
python -m zipfile -e sequential_reference.zip .
```

Each archive already contains its named folder. Extract it into the repository root so that, for example, the incident file is at `data/preprocessed_fire_sample.csv`. Avoid creating an extra nested `data/data/` folder.

| Archive | Extracted folder | Contents |
| --- | --- | --- |
| `data.zip` | `data/` | Incident sample and proxy UAV base coordinates |
| `original_results.zip` | `original_results/` | Archived outputs, including historical sequential comparisons |
| `reference_results.zip` | `reference_results/` | Paired dispatch results, configuration and saved round inputs |
| `runtime_results.zip` | `runtime_results/` | Runtime measurements, instances, configuration and benchmark outputs |
| `sequential_reference.zip` | `sequential_reference/` | Trained DQN checkpoint, paired evaluations and training diagnostics |

The corrected sequential results used in Figures 8 and 9 are in `sequential_reference/`. The historical sequential comparison files in `original_results/` are retained for traceability.

The manuscript submission ZIP already contains these extracted folders inside `repository/`. When using that package, proceed directly to the installation and run commands below.

## Run the experiments

Run the commands below from the GitHub repository root, or from the `repository/` directory when using the manuscript submission ZIP. The reference replay was checked with Python 3.12.14 and the pinned versions in `requirements.txt`.

```sh
python -m pip install -r requirements.txt
python paper1_uav_fire_dispatching_training.py --input-dir data --output-dir results
```

The command runs the single round, battery sweep, learning experiments, and 200 paired dispatch rounds. Use `--experiment randomized` for the 200-round analysis alone or `--experiment single` for the single-round workflow. The analysis defaults to `--rounds 200 --round-seed 42 --bootstrap-resamples 10000`.

The two CSVs in `data/` provide the 15 preprocessed incidents and 10 proxy UAV base coordinates used for the reported runs. They are the inputs for replay. The cited source is the National Interagency Fire Center (NIFC) Wildland Fire Incident Locations dataset. The archived export and clustering rows are unavailable. [NIFC data portal](https://data-nifc.opendata.arcgis.com/)

The separate `--dataset PATH` route prepares a scenario from a supplied raw CSV. An acreage column is required. The code matches recognized column names case-insensitively and filters by the geographic window and California state code when available, drops invalid values, and retains acreage values at or below the 95th percentile of the remaining pool. The evaluation sample and the clustering input draw from that pool with seeds 42 and 43. Clustering uses up to 4,000 rows. The two draws may overlap and have no date split. A new export may yield different incidents and bases.

The saved evaluation sample contains seven incidents at 0.01 acre and eight at 0.10 acre. The corresponding risks are 0.05 and 1, and their sum is 8.35. The paired rounds resample this fixed case. These runs cannot characterize performance at intermediate risk scores or on independent incident sets. The saved inputs have no incident dates or stable source identifiers. Simultaneous availability is a simulation assumption.

## Input checks and zero-dispatch results

Snapshot validation checks numeric fields, latitude and longitude ranges, nonnegative acreage, risk scores between 0 and 1, unique base identifiers, and the configured base battery capacity. The checkpoint loader checks archive fields, format version, learning rate, and finite real-valued arrays of the expected shapes. The CLI reports expected input errors before running the simulation.

An assignment with zero dispatches has zero total energy and robust energy. Risk-weighted delay is the sum of the unserved incidents' risk-weighted deadlines. Average response delay, deadline satisfaction among dispatches, and average remaining battery are undefined and are written as blank CSV fields. Mission success uses the number of safe returns divided by the smaller of the incident and fleet counts.

The reference training command seeds NumPy's original training stream with 42. Training draws for base selection, wind, exploration and replay sampling use that stream in their original order. Sequential evaluation uses the separately seeded environment and random-action streams described below.

```sh
python verify_input_validation.py
python verify_drl_environment.py
python verify_sequential.py
python verify_training.py
python verify_runtime.py
```

## Randomized round protocol

- Each round draws 10 incidents and 8 bases without replacement. The draw order is retained for the nearest-fire heuristic. The incident coordinates and risk scores come from the saved sample.
- Each candidate pair receives independent uniform draws for the wind penalty and realized energy disturbance, both on $[0,0.2]$. All five schemes and all variants use the same draws within a round. The disturbance distribution stays fixed across protection levels.
- The optimization schemes use the assignment routine from the single-round script. It selects the largest feasible set of assignments and minimizes the scheme's cost at that cardinality. Nearest-fire uses the existing greedy rule.
- The 600 Wh comparisons report means and sample standard deviations across rounds. Risk-weighted delay includes the deadline-weighted penalty for every unassigned fire, including rounds without dispatches. Average response delay uses rounds with at least one dispatch.
- Each paired difference is RARD minus the baseline in the same round. The 95% percentile intervals use 10,000 resamples of those differences with bootstrap seed 43. The fraction of improved rounds counts strictly negative differences. The percentage reduction divides the mean decrease by the baseline mean.
- At 250 Wh, the safe return percentage uses the total number of dispatched missions as its denominator. A safe return meets the reference risk-scaled deadline and lands with the reference 20% reserve. The ablations change the relevant admission rule or cost term while keeping those evaluation criteria fixed. The paired safety intervals use differences in per-round safe return percentages where both variants dispatch a mission.
- Measured energy coverage is the fraction of dispatched missions whose realized energy fits the protected estimate. The sweep compares this fraction with the analytical coverage from the bounded disturbance model.

## Outputs

| Analysis | Files in the output directory |
| --- | --- |
| Single-round comparison, Table 1 | `model_comparison_results.csv`, `selected_assignments_by_model.csv` |
| Randomized dispatch, Table 2 | `randomized_rounds.csv`, `randomized_summary.csv` |
| Paired comparisons, Table 3 | `paired_bootstrap.csv` |
| Protection sweep, Table 4 | `protection_rounds.csv`, `protection_summary.csv` |
| Component ablations, Table 5 | `ablation_rounds.csv`, `ablation_summary.csv`, `ablation_paired_safety.csv` |
| Round inputs and settings | `randomized_round_inputs.csv`, `randomized_config.json` |
| Validation and protection figures | `figures/fig10_statistical_validation.png`, `figures/fig11_protection_level_sweep.png` |

The round input file records zero-based row indices into the incident and base snapshots. The configuration file records the model settings, seeds, and input hashes. The `reference_results/` directory contains the output of the documented seed-42 run on the bundled snapshots.

The `original_results/` folder preserves the supplied matrix, assignment, battery-sweep, and learning CSV outputs. Its `sequential_policy_comparison.csv` and `drl_policy_evaluation.csv` are historical and superseded by `sequential_reference/` for the corrected comparison. Use `sequential_reference/` for Figures 8 and 9. The replay generates Figures 10 and 11. Their publication copies are in the manuscript folder of the submission ZIP.

## Paired sequential policy evaluation for Figures 8 and 9

The 800-episode training algorithm, training seed 42, reward and network settings
are unchanged by the pairing correction. Evaluation precomputes 100 cases with
a local `RandomState(49)` generator: draw one base index, then one uniform wind
penalty per case. Each case retains the nearest 15 incidents in distance order.
All four policies receive the same ordered incident list, base, wind and initial
battery. This matches the former deterministic-policy evaluation conditions.
The random policy selects its four actions using a separate `default_rng(50)`
generator. The two evaluation streams do not consume or reset NumPy's global
random state. The standalone greedy DQN evaluation now uses the same schedule.

Reproduce the corrected comparison without training again:

```sh
python paper1_uav_fire_dispatching_training.py --input-dir data --output-dir sequential_replay --experiment sequential --dqn-model sequential_reference/dqn_model.npz
python verify_sequential.py
```

The ordinary `--experiment single` and default `all` workflows also produce this
comparison after training. `sequential_reference/` contains:

| File | Contents |
| --- | --- |
| `dqn_model.npz` | Network weights after the documented 800-episode training run; loaded without pickle |
| `drl_training_history.csv` | Reproduced training history, unchanged from the supplied history |
| `drl_policy_evaluation.csv` | Standalone greedy DQN evaluation on the shared schedule |
| `sequential_episode_schedule.csv` | 100 cases, with zero-based base and ordered incident row indices and wind |
| `sequential_policy_episodes.csv` | All 400 episode outcomes, initial conditions, action counts, served counts and costs |
| `sequential_policy_comparison.csv` | Four policy summaries used in Figures 8 and 9 |
| `sequential_config.json` | Environment/action seeds, protocol, input-array hashes, source hash and model-file hash |

The served ratio pools served incidents over all incident opportunities. The mean
mission cost is the mean negative undiscounted reported episode return. A common
initial-condition schedule does not imply identical action trajectories. The
results describe one trained network and this geographic sample; multiple training
seeds and independent regions have not been evaluated.

`verify_sequential.py` recomputes all summary values, checks every pair of episode
conditions, reloads the network and replays every policy in reversed order after
unrelated global random draws. All 400 episode records must match exactly. It also
checks that evaluation leaves the global random state unchanged.

## Runtime benchmark for Table 6 and Figure 12

```sh
python benchmark_runtime.py --input-dir data --output-dir runtime_new
```

The benchmark imports the exact matrix construction and RARD assignment functions
from the main script. It performs two unmeasured warm-up rounds and ten measured
rounds for each of the six sizes. The baseline 15 × 10 size uses the saved
coordinates. Larger sizes sample incidents and bases with replacement, add
independent Gaussian coordinate jitter (standard deviation 0.01 degree), and clip
to the model's geographic window. Risk scores are retained. The seed is 20261110.
Instance generation and file I/O are outside the timed interval. The timed matrix
stage includes the wind and realized energy draws made by the supplied function.
The assignment stage includes cost construction and the SciPy solve.

On Linux the script attempts to pin itself to one available logical CPU. The
result and any failure are recorded. Other platforms can run the benchmark but
may not support this affinity interface. Library thread settings are set to one.
Use `--core none` to disable pinning. Timing values depend on the machine and load;
rerunning is expected to change the timing samples. The script refuses to replace
a nonempty output folder. Choose a new name for each measurement run.

The included `runtime_results/` folder contains the measurements used in the
corrected manuscript, not a reconstruction of the unsupported earlier timings.

| File | Contents |
| --- | --- |
| `runtime_repetitions.csv` | All 60 measured rounds; stage and total times in ns, seeds, feasible pair counts, dispatch counts and assignment hashes |
| `runtime_instances.csv` | Coordinates, source row indices and incident risks for every measured instance |
| `runtime_config.json` | Generation protocol, settings, hardware, software, CPU affinity, timing boundaries and file hashes |
| `runtime_summary.csv` | Means and sample standard deviations used in Table 6 |
| `runtime_table_rows.tex` | Table 6 numeric rows generated directly from the summary |
| `fig12_scalability_runtime.png` and `.pdf` | Figure 12 generated from the same raw timing samples |

Regenerate the table rows and figures without making new measurements:

```sh
python benchmark_runtime.py --output-dir runtime_results --plot-only
```

The benchmarking scope is one simulated RARD dispatch calculation. It excludes
learning, communications, launch preparation and physical vehicle operation.
Large synthetic instances are a computational workload, not independent field
observations. These mean timings are not worst-case latency guarantees.

The step-1 measured source is preserved as `runtime_results/benchmarked_source.py`.
Later sequential-evaluation changes do not modify the timed RARD functions.
`verify_runtime.py` checks the recorded source hash against that snapshot, compares
all five timed/helper functions and model settings with the current code, and
replays the 60 recorded instances using the current implementation. Timing samples
are never rewritten to match a later source hash. A change to any timed function
or its settings requires a new benchmark and corresponding manuscript update.

## Training reward and Figure 7 diagnostics

Training uses the immediate reward `1000 * newly_served_risk - delay - 0.01 * energy`.
The replay buffer stores this reward divided by 100. The TD target uses discount
0.95 and no bootstrap term for terminal transitions. The reported episode return
instead sums the unscaled immediate rewards without discounting and subtracts
1000 times the initial total fire risk. Its negative equals the reported mission
cost: action costs plus the unserved-risk penalty. This identity does not establish
policy invariance under discounted training.

`sequential_reference/drl_training_diagnostics.csv` contains all 800 episode
service counts, action counts, costs, returns and initial conditions.
`drl_training_summary.json` contains the statistics used in Section 8.9.
To check the stored diagnostics and hand-computable reward transitions:

```sh
python verify_training.py
```

Running `--experiment single` retrains the network and exports these files and
Figure 7. The additional logging does not change training draws or updates: the
800 history rows and every trained network weight match the step-2 checkpoint.
For this sample, -8350 is the upper bound on zero-service returns, not a test for
whether service occurred. There are 288 zero-service episodes; 307 returns lie
at or below this bound, including 19 episodes with service. Neither the moving
average nor these diagnostics establishes convergence or gradient dominance.

## DQN observations and service feasibility

The DQN experiment is a separate single-UAV sequential simulator. Its seven
features, in order, are residual battery divided by full capacity; transformed
target distance; current target risk; transformed base distance; wind divided by
0.2; pending count divided by 15; and an energy-feasibility indicator. Distances
use `min(distance / d_max, 1.5) / 1.5`, where `d_max` is the largest initial
base-to-incident distance, floored at 1e-6 km. The base is the charging location.
The observation omits coordinates, the full pending geometry, cursor and step
count; it is not asserted to be Markov sufficient.

An episode uses fixed incident attributes and one constant wind penalty. Service
is admitted if outbound + hover + return energy is no greater than residual
battery minus 120 Wh (20% of the 600 Wh full capacity). Energy includes the
current wind factor. This test does not apply RARD's uncertainty protection,
risk-scaled deadline or extra realized-energy disturbance. Successful service
spends outbound and hover energy, leaving the return leg reserved. Failed service
discards the target without moving and incurs 0.5 minutes. Recharge takes place
at the same base with 30 minutes of charging delay. Terminal return is accounted
for in cost, without updating terminal position/battery or exporting a realized
landing-reserve outcome. Termination occurs on return, an empty pending list or
60 decisions. The sequential comparison reports service and cost, not the RARD
safe-return metric.

```sh
python verify_drl_environment.py
```

This check covers feature values, an observation alias caused by the omitted step
count, reserve-boundary admission, failed service, recharge and terminal return.
A constructed case also confirms that DQN admits a 25.018858-minute arrival even
though the corresponding high-risk RARD deadline is 15 minutes. This case tests
the implementation; it is not a new experimental result for the incident sample.
