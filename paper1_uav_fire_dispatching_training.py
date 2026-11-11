# Simulation and training code for the Risk-Aware Robust Dispatching study.
# Use --input-dir data to replay the saved incident sample and UAV bases.
# For preprocessing from raw records, place the NIFC CSV beside this script
# as Wildland_Fire_Incident_Locations.csv or pass --dataset PATH.

import argparse
import hashlib
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.optimize import linear_sum_assignment

# The dataset is searched next to the script, so the run works from any directory.
DATASET_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Wildland_Fire_Incident_Locations.csv")
OUTPUT_DIR = "paper1_realistic_results"
RANDOM_SEED = 42
N_FIRE_SAMPLES = 15
N_HISTORICAL_BASE_SAMPLES = 4000
N_UAVS = 10
SELECTED_STATE = "US-CA"  # e.g., "US-CA" or None for all states
REGION_BOUNDS = {"lat_min": 32.50, "lat_max": 36.00, "lon_min": -120.50, "lon_max": -117.00}  # Southern California window of the records; set to None to disable
UAV_SPEED_MPS = 20.0
BATTERY_CAPACITY_WH = 600.0
FLIGHT_POWER_W = 300.0
HOVER_POWER_W = 220.0
HOVER_TIME_MIN = 30.0
SAFETY_RESERVE_RATIO = 0.20
ENERGY_UNCERTAINTY_RATIO = 0.10
ENERGY_DISTURBANCE_MAX = 0.20
BASE_DEADLINE_MIN = 30.0
MIN_DEADLINE_MIN = 8.0
DEADLINE_SENSITIVITY = 0.50
ALPHA_DELAY = 1.0
BETA_ENERGY = 0.01
DELTA_RISK_REWARD = 100.0
BIG_M = 1e9
# --- Paired randomized dispatch experiments ---
N_RANDOMIZED_ROUNDS = 200
N_ROUND_FIRES = 10
N_ROUND_UAVS = 8
N_BOOTSTRAP_RESAMPLES = 10_000
CONSTRAINED_BATTERY_WH = 250.0
PROTECTION_LEVELS = (0.0, 0.05, 0.10, 0.15, 0.20)
# --- Deep reinforcement learning (DQN) dispatching parameters ---
DRL_EPISODES = 800
DRL_FIRES_PER_EPISODE = 15
DRL_MAX_STEPS = 60
DRL_GAMMA = 0.95
DRL_LR = 1e-3
DRL_HIDDEN = 64
DRL_BATCH = 64
DRL_REPLAY_SIZE = 20000
DRL_TARGET_SYNC = 250
DRL_EPS_START, DRL_EPS_END, DRL_EPS_DECAY = 1.0, 0.05, 0.996
DRL_REWARD_SCALE = 100.0  # rewards are divided by this factor inside the network updates only
DRL_MOVING_AVG = 25
DRL_EVAL_EPISODES = 100
DRL_EVAL_ENV_SEED = RANDOM_SEED + 7
DRL_EVAL_ACTION_SEED = RANDOM_SEED + 8
A_DELAY_RL = ALPHA_DELAY          # a: delay weight, aligned with alpha
B_ENERGY_RL = BETA_ENERGY         # b: energy weight, aligned with beta
C_PENALTY_RL = 1000.0             # c: risk penalty weight; no service-dominance guarantee for every fire
CHARGE_TIME_MIN = 30.0
RETARGET_TIME_MIN = 0.5
WIND_PENALTY_MAX = 0.2
np.random.seed(RANDOM_SEED)
# Figures folder next to the running script, matching the LaTeX path figures/figX_*.png
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

def save_fig(fname):
    plt.savefig(os.path.join(OUTPUT_DIR, fname), bbox_inches="tight")
    plt.savefig(os.path.join(FIGURES_DIR, fname), bbox_inches="tight")

figure_font = "Times New Roman" if "Times New Roman" in font_manager.get_font_names() else "DejaVu Serif"
plt.rcParams.update({"font.family": figure_font, "font.size": 12, "figure.dpi": 300, "savefig.dpi": 300})

def find_column(df, names):
    """Resolve known column names without substring matches."""
    columns = {column.strip().casefold(): column for column in df.columns}
    for name in names:
        match = columns.get(name.casefold())
        if match is not None:
            return match
    return None


def find_fire_columns(header):
    return {
        "lat": find_column(header, ["Latitude", "InitialLatitude", "POOLatitude", "FireLatitude", "IncidentLatitude", "lat", "Y"]),
        "lon": find_column(header, ["Longitude", "InitialLongitude", "POOLongitude", "FireLongitude", "IncidentLongitude", "lon", "X"]),
        "acres": find_column(header, ["DiscoveryAcres", "InitialResponseAcres", "DailyAcres", "CalculatedAcres", "GISAcres", "Acres", "FireSize"]),
    }


def haversine_km(lat1, lon1, lat2, lon2):
    R=6371.0
    lat1,lon1,lat2,lon2 = map(np.radians,[lat1,lon1,lat2,lon2])
    dlat,dlon = lat2-lat1, lon2-lon1
    a=np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R*2*np.arcsin(np.sqrt(a))

def norm_minmax(x, minv=0.05):
    x=np.asarray(x,dtype=float)
    if np.nanmax(x)==np.nanmin(x): return np.ones_like(x)*minv
    return minv+(1-minv)*(x-np.nanmin(x))/(np.nanmax(x)-np.nanmin(x))

def save_df(df, name):
    p=os.path.join(OUTPUT_DIR,name); df.to_csv(p,index=False); return p

def load_dataset():
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Missing {DATASET_PATH}. Download NIFC Wildland Fire Incident Locations CSV and place it beside this script.")
    header=pd.read_csv(DATASET_PATH,nrows=0)
    cols = find_fire_columns(header)
    lat, lon, acres = cols["lat"], cols["lon"], cols["acres"]
    state=find_column(header,["POOState","State","InitialResponseState","IncidentState","FireState"])
    name=find_column(header,["IncidentName","FireName","Name","Incident Name"])
    if lat is None or lon is None: raise ValueError("Latitude/longitude columns not found.")
    if acres is None:
        raise ValueError("An acreage column is required (for example, DiscoveryAcres).")
    needed=[c for c in [lat,lon,acres,state,name] if c is not None]
    df=pd.read_csv(DATASET_PATH,usecols=needed,low_memory=False)
    if SELECTED_STATE and state:
        df=df[df[state].astype(str).str.upper()==SELECTED_STATE.upper()]
    if REGION_BOUNDS is not None:
        df=df[(pd.to_numeric(df[lat],errors="coerce").between(REGION_BOUNDS["lat_min"],REGION_BOUNDS["lat_max"]))&(pd.to_numeric(df[lon],errors="coerce").between(REGION_BOUNDS["lon_min"],REGION_BOUNDS["lon_max"]))]
    for c in [lat,lon,acres]: df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df[np.isfinite(df[[lat, lon, acres]]).all(axis=1)]
    df=df[(df[lat].between(-90,90))&(df[lon].between(-180,180))&(df[acres]>=0)]
    if len(df)==0: raise ValueError("No valid records after preprocessing.")
    df=df[df[acres] <= df[acres].quantile(0.95)]
    sample=df.sample(min(N_FIRE_SAMPLES,len(df)),random_state=RANDOM_SEED).reset_index(drop=True)
    sample["FireRiskScore"]=norm_minmax(sample[acres].values,0.05)
    save_df(sample,"preprocessed_fire_sample.csv")
    hist=df.sample(min(N_HISTORICAL_BASE_SAMPLES,len(df)),random_state=RANDOM_SEED+1)
    hist_coords=np.c_[hist[lat].values,hist[lon].values]
    print(f"Loaded {len(sample)} fire incidents. Columns: lat={lat}, lon={lon}, acres={acres}")
    return sample,{"lat":lat,"lon":lon,"acres":acres,"name":name},hist_coords

def kmeans_centers(coords, k, seed=RANDOM_SEED, n_init=10, max_iter=200):
    rng=np.random.default_rng(seed)
    best=None; best_inertia=np.inf
    for _ in range(n_init):
        C=coords[rng.choice(len(coords),k,replace=False)].copy()
        for _ in range(max_iter):
            d=((coords[:,None,:]-C[None,:,:])**2).sum(-1)
            lab=d.argmin(1)
            newC=np.array([coords[lab==kk].mean(0) if np.any(lab==kk) else C[kk] for kk in range(k)])
            if np.allclose(newC,C): C=newC; break
            C=newC
        d=((coords[:,None,:]-C[None,:,:])**2).sum(-1)
        inertia=d.min(1).sum()
        if inertia<best_inertia: best_inertia,best=inertia,C
    return best

def create_bases(fire_lat, fire_lon):
    coords=np.c_[fire_lat,fire_lon]
    k=min(N_UAVS,len(coords))
    bases=kmeans_centers(coords,k)
    save_df(pd.DataFrame({"UAV_ID":range(k),"BaseLatitude":bases[:,0],"BaseLongitude":bases[:,1],"BatteryWh":BATTERY_CAPACITY_WH}),"uav_base_locations.csv")
    return bases[:,0], bases[:,1]

def configure_missions(matrices, capacity, *, protection=ENERGY_UNCERTAINTY_RATIO,
                       reserve=SAFETY_RESERVE_RATIO, deadline=None):
    if deadline is None:
        deadline = matrices["deadline"]
    within_deadline = matrices["delay"] <= deadline[:, None]
    energy = matrices["energy"]
    robust_energy = energy * (1 + protection)
    return dict(
        matrices, deadline=deadline, robust=robust_energy,
        feasible=((robust_energy <= capacity * (1 - reserve)) & within_deadline).astype(int),
        feasible_det=((energy <= capacity) & within_deadline).astype(int),
    )


def compute_matrices(fire_lat, fire_lon, risk, base_lat, base_lon, *,
                     rng=None, export=True):
    N,M=len(fire_lat),len(base_lat)
    if rng is None:
        rng = np.random.default_rng(RANDOM_SEED + 3)
    wind=rng.uniform(0.0,WIND_PENALTY_MAX,size=(N,M))
    dist=np.zeros((N,M)); delay=np.zeros((N,M))
    deadline=np.maximum(MIN_DEADLINE_MIN, BASE_DEADLINE_MIN*(1-DEADLINE_SENSITIVITY*risk))
    hover_wh=HOVER_POWER_W*(HOVER_TIME_MIN/60)
    for i in range(N):
        for j in range(M):
            d=haversine_km(base_lat[j],base_lon[j],fire_lat[i],fire_lon[i]); dist[i,j]=d
            delay[i,j]=(d*1000/UAV_SPEED_MPS)/60
    e_tot=(2*FLIGHT_POWER_W*(delay/60)+hover_wh)*(1+wind)
    xi=rng.uniform(0.0,ENERGY_DISTURBANCE_MAX,size=(N,M))
    e_act=e_tot*(1+xi)
    matrices = configure_missions(
        {"distance":dist,"delay":delay,"energy":e_tot,"actual":e_act,"wind":wind,"deadline":deadline},
        BATTERY_CAPACITY_WH,
    )
    if export:
        for key, filename in [
            ("distance", "matrix_distance_km.csv"),
            ("delay", "matrix_delay_min.csv"),
            ("energy", "matrix_energy_total_wh.csv"),
            ("robust", "matrix_energy_robust_wh.csv"),
            ("wind", "matrix_wind_penalty.csv"),
            ("feasible", "matrix_feasible.csv"),
        ]:
            save_df(pd.DataFrame(matrices[key]), filename)
    return matrices

def solve(cost, feasible):
    c=cost.astype(float).copy(); c[feasible==0]=BIG_M
    rows,cols=linear_sum_assignment(c)
    return [(int(r),int(col)) for r,col in zip(rows,cols) if c[r,col] < BIG_M]

def greedy_nearest(dist, feas):
    N,M=dist.shape; assigned=set(); pairs=[]
    for j in range(M):
        order=np.argsort(dist[:,j])
        for i in order:
            if feas[i,j] and int(i) not in assigned:
                pairs.append((int(i),int(j))); assigned.add(int(i)); break
    return pairs

def rard_assignment(risk, matrices, *, alpha=ALPHA_DELAY, delta=DELTA_RISK_REWARD):
    cost = (
        alpha * risk[:, None] * matrices["delay"]
        + BETA_ENERGY * matrices["robust"]
        - delta * risk[:, None]
    )
    return solve(cost, matrices["feasible"])


def run_models(risk,m):
    d,delay,e=m["distance"],m["delay"],m["energy"]
    feas_det=m["feasible_det"]
    delay_norm=delay/(np.max(delay) or 1.0); e_norm=e/np.max(e)
    assigns={
        "Nearest Fire": greedy_nearest(d,feas_det),
        "Minimum Delay": solve(delay,feas_det),
        "Minimum Energy": solve(e,feas_det),
        "Energy-Delay": solve(0.5*delay_norm+0.5*e_norm,feas_det),
        "Proposed": rard_assignment(risk, m),
    }
    return assigns


def score_assignment(assignment, risk, matrices, capacity, *, evaluation_deadline=None):
    if evaluation_deadline is None:
        evaluation_deadline = matrices["deadline"]
    served = {i for i, _ in assignment}
    unserved = [i for i in range(len(risk)) if i not in served]
    delays = [matrices["delay"][i, j] for i, j in assignment]
    safe_budget = capacity * (1 - SAFETY_RESERVE_RATIO)
    safe_returns = sum(
        1 for i, j in assignment
        if matrices["actual"][i, j] <= safe_budget
        and matrices["delay"][i, j] <= evaluation_deadline[i]
    )
    return {
        "Dispatched Missions": len(assignment),
        "Average Response Delay (min)": np.mean(delays) if delays else np.nan,
        "Risk-Weighted Delay": sum(risk[i] * matrices["delay"][i, j] for i, j in assignment)
        + sum(risk[i] * evaluation_deadline[i] for i in unserved),
        "Unserved Fire Risk": sum(risk[i] for i in unserved),
        "Safe Returns": safe_returns,
    }


def evaluate(assigns, risk, matrices):
    rows = []
    max_missions = min(len(risk), matrices["distance"].shape[1])
    for model, assignment in assigns.items():
        served = len(assignment)
        energies = [matrices["energy"][i, j] for i, j in assignment]
        robust = [matrices["robust"][i, j] for i, j in assignment]
        score = score_assignment(assignment, risk, matrices, BATTERY_CAPACITY_WH)
        on_time = sum(matrices["delay"][i, j] <= matrices["deadline"][i] for i, j in assignment)
        rows.append({
            "Model": model,
            "Served Fires": served,
            "Average Response Delay (min)": score["Average Response Delay (min)"],
            "Total Energy (Wh)": np.sum(energies),
            "Total Robust Energy (Wh)": np.sum(robust),
            "Mission Success Rate (%)": 100 * (score["Safe Returns"] / max_missions) if max_missions else np.nan,
            "Deadline Satisfaction Rate (%)": 100 * (on_time / served) if served else np.nan,
            "Risk-Weighted Delay": score["Risk-Weighted Delay"],
            "Unserved Fire Risk": score["Unserved Fire Risk"],
            "Average Remaining Battery (Wh)": np.mean([BATTERY_CAPACITY_WH - energy for energy in robust]) if served else np.nan,
        })
    results = pd.DataFrame(rows)
    save_df(results, "model_comparison_results.csv")
    return results


def battery_sensitivity(risk,m):
    levels=[300,500,700,1000]; data={"Battery Capacity (Wh)":levels}
    for model in ["Nearest Fire","Minimum Delay","Minimum Energy","Energy-Delay","Proposed"]: data[model]=[]
    for B in levels:
        mm=configure_missions(m,B)
        assigns=run_models(risk,mm)
        M=m["distance"].shape[1]
        for model,A in assigns.items():
            safe=score_assignment(A,risk,mm,B)["Safe Returns"]
            data[model].append(safe/min(len(risk),M)*100)
    df=pd.DataFrame(data); save_df(df,"battery_capacity_sensitivity.csv"); return df

def plot_bar(vals,ylabel,title,fname):
    names=["Nearest","Delay","Energy","E-D","Proposed"]; colors=["#7f7f7f","#1f77b4","#2ca02c","#ff7f0e","#d62728"]
    plt.figure(figsize=(7,4.2)); bars=plt.bar(names,vals,color=colors,edgecolor="black",linewidth=.7); bars[-1].set_hatch("//")
    finite = vals[np.isfinite(vals)]
    if not finite.size:
        plt.text(0.5, 0.5, "No dispatched missions", transform=plt.gca().transAxes,
                 ha="center", va="center")
    else:
        offset = finite.max() * 0.015
        for bar in bars:
            height = bar.get_height()
            if np.isfinite(height):
                plt.text(bar.get_x() + bar.get_width() / 2, height + offset,
                         f"{height:.2f}", ha="center", va="bottom", fontsize=9)
    plt.ylabel(ylabel); plt.title(title); plt.grid(axis="y",linestyle="--",color="0.8"); plt.tight_layout(); save_fig(fname); plt.close()

def make_figures(sample,cols,base_lat,base_lon,results,batt):
    lat,lon=cols["lat"],cols["lon"]; risk=sample["FireRiskScore"].values
    plt.figure(figsize=(7,5)); sc=plt.scatter(sample[lon],sample[lat],c=risk,cmap="YlOrRd",s=25,edgecolor="black",linewidth=.2,label="Fire incidents")
    plt.scatter(base_lon,base_lat,marker="^",s=130,c="blue",edgecolor="black",label="UAV bases")
    plt.colorbar(sc,label="Fire Risk Score"); plt.xlabel("Longitude"); plt.ylabel("Latitude"); plt.title("Geographic Distribution of Fire Incidents and UAV Bases"); plt.legend(); plt.grid(True,linestyle="--",color="0.8"); plt.tight_layout(); save_fig("fig1_geographic_distribution.png"); plt.close()
    plot_bar(results["Average Response Delay (min)"].values,"Average Response Delay (min)","Average Response Delay Comparison","fig2_average_response_delay.png")
    plot_bar(results["Mission Success Rate (%)"].values,"Mission Success Rate (%)","Mission Success Rate Comparison","fig3_mission_success_rate.png")
    plot_bar(results["Risk-Weighted Delay"].values,"Risk-Weighted Delay","Risk-Weighted Delay Comparison","fig4_risk_weighted_delay.png")
    plot_bar(results["Total Robust Energy (Wh)"].values,"Total Robust Energy (Wh)","Total Robust Energy Consumption Comparison","fig5_total_robust_energy.png")
    plt.figure(figsize=(7,4.5)); colors=["#7f7f7f","#1f77b4","#2ca02c","#ff7f0e","#d62728"]
    widths=[5.0,3.8,2.6,1.6,1.0]; styles=["-","--","-.",":","-"]; marks=["o","s","^","D","*"]; msizes=[11,9,8,6,12]
    for k,(model,c) in enumerate(zip(["Nearest Fire","Minimum Delay","Minimum Energy","Energy-Delay","Proposed"],colors)): plt.plot(batt["Battery Capacity (Wh)"],batt[model],marker=marks[k],markersize=msizes[k],linestyle=styles[k],linewidth=widths[k],color=c,label=model,zorder=2+k)
    plt.xlabel("Battery Capacity (Wh)"); plt.ylabel("Mission Success Rate (%)"); plt.title("Impact of Battery Capacity on Mission Success"); plt.grid(True,linestyle="--",color="0.8"); plt.legend(); plt.tight_layout(); save_fig("fig6_battery_capacity_sensitivity.png"); plt.close()

def save_assignments(assigns,sample,cols):
    rows=[]
    for model,A in assigns.items():
        for i,j in A:
            rows.append({"Model":model,"FireIndex":i,"UAVIndex":j,"Latitude":sample.loc[i,cols["lat"]],"Longitude":sample.loc[i,cols["lon"]],"DiscoveryAcres":sample.loc[i,cols["acres"]],"FireRiskScore":sample.loc[i,"FireRiskScore"]})
    save_df(pd.DataFrame(rows),"selected_assignments_by_model.csv")

# ---------------------------------------------------------------
# Deep reinforcement learning (DQN) dispatching policy
# State: battery level, distance to candidate fire, fire urgency,
#        distance to charging station, wind penalty, remaining fire ratio.
# Actions: 0 go to fire, 1 go to charging station, 2 choose next target, 3 return to base.
# Environment reward: c * newly served risk - a * action delay - b * action energy.
# Reported undiscounted episode return: sum(environment rewards)
# minus c*sum(initial risks), equal to -a*total delay - b*total energy
# minus c*unserved risk. Training discounts scaled environment rewards (gamma=0.95).
# This undiscounted accounting identity does not establish policy invariance for
# the discounted training objective.
# ---------------------------------------------------------------
class FireDispatchEnv:
    N_ACTIONS = 4
    N_STATES = 7

    def __init__(self, f_lat, f_lon, f_risk, b_lat, b_lon):
        self.f_lat, self.f_lon, self.f_risk = f_lat, f_lon, f_risk
        self.b_lat, self.b_lon = b_lat, b_lon
        self.d_max = max(float(np.max(haversine_km(b_lat, b_lon, f_lat, f_lon))), 1e-6)

    def _leg(self, lat1, lon1, lat2, lon2):
        d = float(haversine_km(lat1, lon1, lat2, lon2))
        t_min = (d*1000/UAV_SPEED_MPS)/60
        e_wh = FLIGHT_POWER_W*(t_min/60)*(1+self.wind)
        return t_min, e_wh

    def reset(self, *, wind=None):
        self.battery = BATTERY_CAPACITY_WH
        self.lat, self.lon = self.b_lat, self.b_lon
        self.pending = list(np.argsort(-self.f_risk))
        self.cursor = 0
        self.served = 0
        self.total_risk = float(np.sum(self.f_risk))
        # Training retains its original draws; evaluation supplies a fixed schedule.
        self.wind = np.random.uniform(0.0, WIND_PENALTY_MAX) if wind is None else float(wind)
        if not 0.0 <= self.wind <= WIND_PENALTY_MAX:
            raise ValueError("Wind must be within the configured support.")
        self.steps = 0
        return self._state()

    def _target(self):
        if not self.pending: return None
        return self.pending[self.cursor % len(self.pending)]

    def _mission_energy(self, i):
        t_go, e_go = self._leg(self.lat, self.lon, self.f_lat[i], self.f_lon[i])
        _, e_back = self._leg(self.f_lat[i], self.f_lon[i], self.b_lat, self.b_lon)
        hover_wh = HOVER_POWER_W*(HOVER_TIME_MIN/60)*(1+self.wind)
        return t_go, e_go, hover_wh, e_back

    def _feasible(self, i):
        _, e_go, hover_wh, e_back = self._mission_energy(i)
        return e_go + hover_wh + e_back <= self.battery - BATTERY_CAPACITY_WH*SAFETY_RESERVE_RATIO

    def _state(self):
        i = self._target()
        if i is None: d_fire, risk, feas = 0.0, 0.0, 0.0
        else:
            d_fire = min(float(haversine_km(self.lat, self.lon, self.f_lat[i], self.f_lon[i]))/self.d_max, 1.5)/1.5
            risk = float(self.f_risk[i])
            feas = 1.0 if self._feasible(i) else 0.0
        d_ch = min(float(haversine_km(self.lat, self.lon, self.b_lat, self.b_lon))/self.d_max, 1.5)/1.5
        return np.array([self.battery/BATTERY_CAPACITY_WH, d_fire, risk, d_ch,
                         self.wind/WIND_PENALTY_MAX, len(self.pending)/DRL_FIRES_PER_EPISODE, feas])

    def _return_cost(self):
        t, e = self._leg(self.lat, self.lon, self.b_lat, self.b_lon)
        return -A_DELAY_RL*t - B_ENERGY_RL*e

    def step(self, action):
        self.steps += 1
        reward, done = 0.0, False
        i = self._target()
        if action == 0 and i is not None:
            t_go, e_go, hover_wh, e_back = self._mission_energy(i)
            if self._feasible(i):
                self.battery -= e_go + hover_wh
                self.lat, self.lon = self.f_lat[i], self.f_lon[i]
                reward = C_PENALTY_RL*float(self.f_risk[i]) - A_DELAY_RL*(t_go+HOVER_TIME_MIN) - B_ENERGY_RL*(e_go+hover_wh)
                self.served += 1
            else:
                reward = -A_DELAY_RL*RETARGET_TIME_MIN
            self.pending.remove(i); self.cursor = 0
        elif action == 1:
            t, e = self._leg(self.lat, self.lon, self.b_lat, self.b_lon)
            if e <= self.battery:
                self.lat, self.lon = self.b_lat, self.b_lon
                self.battery = BATTERY_CAPACITY_WH
                reward = -A_DELAY_RL*(t+CHARGE_TIME_MIN) - B_ENERGY_RL*e
            else:
                reward = self._return_cost(); done = True
        elif action == 2:
            self.cursor += 1
            reward = -A_DELAY_RL*RETARGET_TIME_MIN
        else:
            reward = self._return_cost(); done = True
        if not done and (not self.pending or self.steps >= DRL_MAX_STEPS):
            reward += self._return_cost(); done = True
        return self._state(), reward, done

class DQNetwork:
    def __init__(self, n_in, n_hidden, n_out, lr, seed=0):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, np.sqrt(2.0/n_in), (n_in, n_hidden)); self.b1 = np.zeros(n_hidden)
        self.W2 = rng.normal(0, np.sqrt(2.0/n_hidden), (n_hidden, n_out)); self.b2 = np.zeros(n_out)
        self.lr = lr

    def forward(self, x):
        z1 = x @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        return h @ self.W2 + self.b2, (x, z1, h)

    def predict(self, x):
        return self.forward(np.atleast_2d(np.asarray(x, dtype=float)))[0]

    def update(self, x, actions, targets):
        q, (xi, z1, h) = self.forward(x)
        grad_q = np.zeros_like(q)
        rows = np.arange(len(x))
        grad_q[rows, actions] = 2.0*(q[rows, actions] - targets)/len(x)
        dW2 = h.T @ grad_q; db2 = grad_q.sum(0)
        dh = grad_q @ self.W2.T; dh[z1 <= 0] = 0.0
        dW1 = xi.T @ dh; db1 = dh.sum(0)
        self.W1 -= self.lr*dW1; self.b1 -= self.lr*db1
        self.W2 -= self.lr*dW2; self.b2 -= self.lr*db2

    def clone_weights(self, other):
        self.W1, self.b1 = other.W1.copy(), other.b1.copy()
        self.W2, self.b2 = other.W2.copy(), other.b2.copy()

def _sample_episode_env(f_lat, f_lon, risk, base_lat, base_lon):
    j = np.random.randint(len(base_lat))
    d = haversine_km(base_lat[j], base_lon[j], f_lat, f_lon)
    near = np.argsort(d)[:DRL_FIRES_PER_EPISODE]
    return FireDispatchEnv(f_lat[near], f_lon[near], risk[near], base_lat[j], base_lon[j])

def train_drl(sample, cols, base_lat, base_lon):
    f_lat = sample[cols["lat"]].values; f_lon = sample[cols["lon"]].values
    risk = sample["FireRiskScore"].values
    online = DQNetwork(FireDispatchEnv.N_STATES, DRL_HIDDEN, FireDispatchEnv.N_ACTIONS, DRL_LR, RANDOM_SEED)
    target = DQNetwork(FireDispatchEnv.N_STATES, DRL_HIDDEN, FireDispatchEnv.N_ACTIONS, DRL_LR, RANDOM_SEED)
    target.clone_weights(online)
    replay, returns, diagnostics = [], [], []
    eps, step_count = DRL_EPS_START, 0
    for ep in range(DRL_EPISODES):
        env = _sample_episode_env(f_lat, f_lon, risk, base_lat, base_lon)
        s = env.reset(); ep_ret, done = 0.0, False
        action_counts = np.zeros(env.N_ACTIONS, dtype=int)
        served_risk = 0.0
        while not done:
            if np.random.rand() < eps: a = np.random.randint(env.N_ACTIONS)
            else: a = int(np.argmax(online.predict(s)[0]))
            target_index, previous_served = env._target(), env.served
            action_counts[a] += 1
            s2, r, done = env.step(a)
            if env.served > previous_served:
                served_risk += float(env.f_risk[target_index])
            replay.append((s, a, r/DRL_REWARD_SCALE, s2, done))
            if len(replay) > DRL_REPLAY_SIZE: replay.pop(0)
            s = s2; ep_ret += r; step_count += 1
            if len(replay) >= DRL_BATCH:
                idx = np.random.randint(len(replay), size=DRL_BATCH)
                xs = np.array([replay[k][0] for k in idx]); acts = np.array([replay[k][1] for k in idx])
                rs = np.array([replay[k][2] for k in idx]); xs2 = np.array([replay[k][3] for k in idx])
                ds = np.array([replay[k][4] for k in idx], dtype=float)
                q_next = target.predict(xs2).max(axis=1)
                online.update(xs, acts, rs + DRL_GAMMA*(1-ds)*q_next)
            if step_count % DRL_TARGET_SYNC == 0: target.clone_weights(online)
        eps = max(DRL_EPS_END, eps*DRL_EPS_DECAY)
        reported_return = ep_ret - C_PENALTY_RL*env.total_risk
        returns.append(reported_return)
        diagnostics.append({"Episode": ep+1, "Served": env.served, "Steps": env.steps,
            "BaseLatitude": float(env.b_lat), "BaseLongitude": float(env.b_lon),
            "Wind": env.wind, "TotalFireRisk": env.total_risk, "ServedRisk": served_risk,
            "ActionCost": C_PENALTY_RL*served_risk-ep_ret, "ReportedReturn": reported_return,
            "NoServiceUpperBound": -C_PENALTY_RL*env.total_risk,
            "FireActions": int(action_counts[0]), "ChargeActions": int(action_counts[1]),
            "RetargetActions": int(action_counts[2]), "ReturnActions": int(action_counts[3])})
        if (ep+1) % 100 == 0: print(f"DRL episode {ep+1}/{DRL_EPISODES}, epsilon={eps:.3f}, return={returns[-1]:.1f}")
    hist = pd.DataFrame({"Episode": np.arange(1, DRL_EPISODES+1), "Cumulative Reward": returns})
    hist["Moving Average"] = hist["Cumulative Reward"].rolling(DRL_MOVING_AVG, min_periods=1).mean()
    save_df(hist, "drl_training_history.csv")
    diagnostic_frame = pd.DataFrame(diagnostics)
    save_df(diagnostic_frame, "drl_training_diagnostics.csv")
    stats = summarize_training_history(hist, diagnostic_frame)
    (Path(OUTPUT_DIR) / "drl_training_summary.json").write_text(json.dumps(stats, indent=2) + "\n")
    save_dqn_network(online, Path(OUTPUT_DIR) / "dqn_model.npz")
    evalres = evaluate_drl_policy(online, f_lat, f_lon, risk, base_lat, base_lon)
    save_df(pd.DataFrame([evalres]), "drl_policy_evaluation.csv")
    return hist, evalres, online

def evaluate_drl_policy(net, f_lat, f_lon, risk, base_lat, base_lon):
    sr, ret = evaluate_sequential_policy(
        lambda env, state: int(np.argmax(net.predict(state)[0])),
        f_lat, f_lon, risk, base_lat, base_lon,
    )
    return {"Greedy Served Ratio (%)": sr, "Greedy Average Return": ret}


def save_dqn_network(net, path):
    np.savez(path, W1=net.W1, b1=net.b1, W2=net.W2, b2=net.b2,
             learning_rate=np.array(net.lr), format_version=np.array(1))


def load_dqn_network(path):
    checkpoint = np.load(path, allow_pickle=False)
    if not isinstance(checkpoint, np.lib.npyio.NpzFile):
        raise ValueError("The DQN checkpoint must be a .npz archive.")
    with checkpoint as data:
        required = {"format_version", "learning_rate", "W1", "b1", "W2", "b2"}
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"Missing DQN checkpoint fields: {', '.join(missing)}.")
        version = data["format_version"]
        if version.ndim != 0 or version.dtype.kind not in "iu" or version.item() != 1:
            raise ValueError("Unsupported DQN checkpoint format.")
        rate = data["learning_rate"]
        if rate.ndim != 0 or rate.dtype.kind not in "iuf" or not np.isfinite(rate) or rate <= 0:
            raise ValueError("The DQN checkpoint learning rate must be a finite positive scalar.")
        net = DQNetwork(FireDispatchEnv.N_STATES, DRL_HIDDEN,
                        FireDispatchEnv.N_ACTIONS, float(rate))
        for name in ("W1", "b1", "W2", "b2"):
            weights = data[name]
            if (weights.shape != getattr(net, name).shape
                    or weights.dtype.kind not in "iuf" or not np.isfinite(weights).all()):
                raise ValueError(f"Invalid checkpoint array: {name}")
            setattr(net, name, weights.copy())
    return net


def summarize_training_history(hist, diagnostics):
    no_service = diagnostics["Served"].eq(0)
    below_reference = diagnostics["ReportedReturn"] <= diagnostics["NoServiceUpperBound"] + 1e-8
    action_cost = diagnostics.loc[no_service, "ActionCost"]
    second_half = hist.iloc[len(hist)//2:]
    return {
        "episodes": len(hist), "no_service_count": int(no_service.sum()),
        "at_or_below_reference_count": int(below_reference.sum()),
        "served_but_at_or_below_reference_count": int((below_reference & ~no_service).sum()),
        "only_retarget_or_return_no_service_count": int((no_service & diagnostics.FireActions.eq(0) & diagnostics.ChargeActions.eq(0)).sum()),
        "no_service_action_cost_median": float(action_cost.median()),
        "no_service_action_cost_p90": float(action_cost.quantile(0.90)),
        "return_iqr": float(hist["Cumulative Reward"].quantile(0.75)-hist["Cumulative Reward"].quantile(0.25)),
        "final_moving_average": float(hist["Moving Average"].iloc[-1]),
        "second_half_moving_average_slope": float(np.polyfit(second_half.Episode, second_half["Moving Average"], 1)[0]),
        "moving_average_window": DRL_MOVING_AVG,
        "interpretation": "No service is determined from Served==0, not by a return threshold. The threshold is an upper bound for zero-service returns. These descriptive summaries do not establish convergence or a cause of the policy performance.",
    }


def plot_drl_figure(hist, diagnostics=None):
    plt.figure(figsize=(7,5.0))
    plt.plot(hist["Episode"], hist["Cumulative Reward"], color="#9ecae1", linewidth=.8, label="Reported episode return")
    plt.plot(hist["Episode"], hist["Moving Average"], color="#d62728", linewidth=2, label=f"Moving average ({DRL_MOVING_AVG} episodes)")
    if diagnostics is not None:
        bounds = diagnostics["NoServiceUpperBound"].to_numpy()
        if np.allclose(bounds, bounds[0], rtol=0, atol=1e-8):
            plt.axhline(bounds[0], color="0.3", linestyle="--", linewidth=1.1,
                        label=f"Zero-service upper bound ({bounds[0]:.0f})")
        else:
            plt.plot(hist["Episode"], bounds, color="0.3", linestyle="--", linewidth=1.1,
                     label="Episode-specific zero-service upper bound")
    plt.xlabel("Training Episode"); plt.ylabel("Reported Episode Return")
    plt.title("Training Returns of the DQN Dispatching Policy")
    plt.grid(True, linestyle="--", color="0.8")
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.19), fontsize=9)
    plt.tight_layout()
    save_fig("fig7_drl_training_convergence.png"); plt.close()

def heuristic_action(env, mode, *, action_rng=None):
    # Rule based sequential policies, compared against the trained DQN in the same environment.
    if mode == "Random":
        if action_rng is None:
            raise ValueError("The random policy requires a separate action generator.")
        return int(action_rng.integers(env.N_ACTIONS))
    i = env._target()
    feas = [k for k in env.pending if env._feasible(k)]
    if feas:
        if mode == "Nearest First":
            dists = {k: float(haversine_km(env.lat, env.lon, env.f_lat[k], env.f_lon[k])) for k in feas}
            best = min(dists, key=dists.get)
            return 0 if i == best else 2
        return 0 if i in feas else 2  # Risk First: pending is sorted by risk
    if env.battery < BATTERY_CAPACITY_WH*0.99 and env.pending:
        return 1
    return 3

def make_sequential_schedule(f_lat, f_lon, risk, base_lat, base_lon,
                             *, seed=DRL_EVAL_ENV_SEED, episodes=DRL_EVAL_EPISODES):
    # Preserve the legacy deterministic-policy base/wind sequence, without using
    # or modifying global RNG state. Generate it before any policy acts.
    environment_rng = np.random.RandomState(seed)
    schedule = []
    for episode in range(1, episodes + 1):
        base = int(environment_rng.randint(len(base_lat)))
        distance = haversine_km(base_lat[base], base_lon[base], f_lat, f_lon)
        indices = np.argsort(distance)[:DRL_FIRES_PER_EPISODE].tolist()
        schedule.append({"Episode": episode, "BaseIndex": base,
                         "FireIndices": json.dumps(indices),
                         "Wind": float(environment_rng.uniform(0.0, WIND_PENALTY_MAX))})
    return schedule


def evaluate_sequential_policy(select, f_lat, f_lon, risk, base_lat, base_lon,
                               *, schedule=None, return_episodes=False):
    if schedule is None:
        schedule = make_sequential_schedule(f_lat, f_lon, risk, base_lat, base_lon)
    served, total, rets = 0, 0, []
    records = []
    for case in schedule:
        base, indices = int(case["BaseIndex"]), np.asarray(json.loads(case["FireIndices"]), dtype=int)
        env = FireDispatchEnv(f_lat[indices], f_lon[indices], risk[indices],
                              base_lat[base], base_lon[base])
        s = env.reset(wind=case["Wind"]); ep_ret, done = 0.0, False
        action_counts = np.zeros(env.N_ACTIONS, dtype=int)
        while not done:
            a = select(env, s)
            action_counts[a] += 1
            s, r, done = env.step(a)
            ep_ret += r
        served += env.served; total += len(indices)
        episode_return = ep_ret - C_PENALTY_RL*env.total_risk
        rets.append(episode_return)
        records.append({**case, "BaseLatitude": float(base_lat[base]),
                        "BaseLongitude": float(base_lon[base]), "FireCount": len(indices),
                        "InitialBatteryWh": BATTERY_CAPACITY_WH, "TotalFireRisk": env.total_risk,
                        "Served": env.served, "Steps": env.steps,
                        "Return": episode_return, "MissionCost": -episode_return,
                        "FireActions": int(action_counts[0]), "ChargeActions": int(action_counts[1]),
                        "RetargetActions": int(action_counts[2]), "ReturnActions": int(action_counts[3])})
    result = (served/total*100, float(np.mean(rets)))
    return (*result, pd.DataFrame(records)) if return_episodes else result

def compare_sequential_policies(net, f_lat, f_lon, risk, base_lat, base_lon, *, policy_order=None):
    names = ["Random", "Nearest First", "Risk First", "DQN Policy"]
    if policy_order is not None:
        if sorted(policy_order) != sorted(names):
            raise ValueError("policy_order must include each of the four policies exactly once.")
        names = list(policy_order)
    schedule = make_sequential_schedule(f_lat, f_lon, risk, base_lat, base_lon)
    save_df(pd.DataFrame(schedule), "sequential_episode_schedule.csv")
    rows, all_episodes = [], []
    for name in names:
        action_rng = np.random.default_rng(DRL_EVAL_ACTION_SEED)
        if name == "DQN Policy":
            select = lambda env, state: int(np.argmax(net.predict(state)[0]))
        else:
            select = lambda env, state: heuristic_action(env, name, action_rng=action_rng)
        sr, ret, episodes = evaluate_sequential_policy(
            select, f_lat, f_lon, risk, base_lat, base_lon,
            schedule=schedule, return_episodes=True,
        )
        episodes.insert(0, "Policy", name)
        all_episodes.append(episodes)
        rows.append({"Policy": name, "Served Ratio (%)": sr, "Average Return": ret, "Average Mission Cost": -ret})
    df = pd.DataFrame(rows); save_df(df, "sequential_policy_comparison.csv")
    save_df(pd.concat(all_episodes, ignore_index=True), "sequential_policy_episodes.csv")
    save_dqn_network(net, Path(OUTPUT_DIR) / "dqn_model.npz")
    config = {"episodes_per_policy": len(schedule), "environment_seed": DRL_EVAL_ENV_SEED,
              "environment_rng": "NumPy RandomState MT19937; base draw then wind draw per episode",
              "random_action_seed": DRL_EVAL_ACTION_SEED, "action_rng": "NumPy default_rng PCG64",
              "protocol": "Precomputed common schedule; all policies share the base, ordered fire indices, wind and initial battery in each episode.",
              "training_seed": RANDOM_SEED, "training_episodes": DRL_EPISODES,
              "metric": "Pooled served count / pooled fire count; mean undiscounted mission cost including unserved-risk penalty.",
              "input_arrays_sha256": {
                  name: hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()
                  for name, values in [("fire_lat", f_lat), ("fire_lon", f_lon), ("risk", risk),
                                       ("base_lat", base_lat), ("base_lon", base_lon)]},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "checkpoint_sha256": hashlib.sha256((Path(OUTPUT_DIR) / "dqn_model.npz").read_bytes()).hexdigest()}
    (Path(OUTPUT_DIR) / "sequential_config.json").write_text(json.dumps(config, indent=2) + "\n")
    return df

def plot_policy_figures(df):
    names = df["Policy"].tolist(); colors = ["#7f7f7f", "#1f77b4", "#ff7f0e", "#d62728"]
    specs = [
        (df["Served Ratio (%)"].values, "Served Fire Ratio (%)", "Served Fire Ratio of the Sequential Policies", "fig8_sequential_policy_served_ratio.png"),
        (df["Average Mission Cost"].values, "Average Mission Cost", "Average Mission Cost of the Sequential Policies", "fig9_sequential_policy_cost.png"),
    ]
    for vals, ylabel, title, fname in specs:
        plt.figure(figsize=(7,4.2))
        bars = plt.bar(names, vals, color=colors, edgecolor="black", linewidth=.7); bars[-1].set_hatch("//")
        mx = np.nanmax(vals)
        for b in bars:
            h = b.get_height()
            plt.text(b.get_x()+b.get_width()/2, h+mx*.015, f"{h:.1f}", ha="center", va="bottom", fontsize=9)
        plt.ylim(0, max(mx*1.15, 1.0))
        plt.ylabel(ylabel); plt.title(title); plt.grid(axis="y", linestyle="--", color="0.8")
        plt.tight_layout(); save_fig(fname); plt.close()

def run_single_round(sample, cols, blat, blon):
    flat,flon=sample[cols["lat"]].values,sample[cols["lon"]].values; risk=sample["FireRiskScore"].values
    matrices=compute_matrices(flat,flon,risk,blat,blon)
    print(f"Feasible assignment ratio: {matrices['feasible'].mean()*100:.2f}%")
    assigns=run_models(risk,matrices)
    results=evaluate(assigns,risk,matrices)
    save_assignments(assigns,sample,cols)
    batt=battery_sensitivity(risk,matrices)
    make_figures(sample,cols,blat,blon,results,batt)
    hist,drl_eval,net=train_drl(sample,cols,blat,blon)
    plot_drl_figure(hist, pd.read_csv(Path(OUTPUT_DIR) / "drl_training_diagnostics.csv"))
    comp=compare_sequential_policies(net,flat,flon,risk,blat,blon)
    plot_policy_figures(comp)
    print("\nDRL greedy policy evaluation:", drl_eval)
    print("\nSequential policy comparison:")
    print(comp.to_string(index=False))
    print("\nCompleted. Outputs saved in:", OUTPUT_DIR)
    print("Figures saved in:", FIGURES_DIR)
    print(results.to_string(index=False))


def load_saved_inputs(input_dir):
    """Load the incident and base snapshots exported by the single-round run."""
    sample = pd.read_csv(input_dir / "preprocessed_fire_sample.csv", float_precision="round_trip")
    bases = pd.read_csv(input_dir / "uav_base_locations.csv", float_precision="round_trip")
    cols = find_fire_columns(sample)
    if (
        not all(cols.values())
        or "FireRiskScore" not in sample
        or not {"UAV_ID", "BaseLatitude", "BaseLongitude", "BatteryWh"} <= set(bases)
    ):
        raise ValueError("The input directory must contain both exported incident and UAV base snapshots.")
    if sample.empty or bases.empty:
        raise ValueError("The incident and UAV base snapshots must contain at least one row each.")
    for frame, names in (
        (sample, [*cols.values(), "FireRiskScore"]),
        (bases, ["BaseLatitude", "BaseLongitude", "BatteryWh"]),
    ):
        for name in names:
            frame[name] = pd.to_numeric(frame[name], errors="coerce")
            if not np.isfinite(frame[name]).all():
                raise ValueError(f"Snapshot field {name} must contain finite numbers.")
    if not (
        sample[cols["lat"]].between(-90, 90).all()
        and sample[cols["lon"]].between(-180, 180).all()
        and bases["BaseLatitude"].between(-90, 90).all()
        and bases["BaseLongitude"].between(-180, 180).all()
    ):
        raise ValueError("Snapshot coordinates must be within valid latitude and longitude ranges.")
    if not sample[cols["acres"]].ge(0).all():
        raise ValueError("Snapshot acreage must be nonnegative.")
    if not sample["FireRiskScore"].between(0, 1).all():
        raise ValueError("Snapshot risk scores must be between 0 and 1.")
    if bases["UAV_ID"].isna().any() or bases["UAV_ID"].duplicated().any():
        raise ValueError("Snapshot UAV identifiers must be present and unique.")
    if not bases["BatteryWh"].eq(BATTERY_CAPACITY_WH).all():
        raise ValueError(f"The snapshots require a nominal battery capacity of {BATTERY_CAPACITY_WH:g} Wh.")
    if input_dir.resolve() != Path(OUTPUT_DIR).resolve():
        save_df(sample, "preprocessed_fire_sample.csv")
        save_df(bases, "uav_base_locations.csv")
    return sample, cols, bases["BaseLatitude"].to_numpy(), bases["BaseLongitude"].to_numpy()


def evaluate_rard_variant(risk, matrices, capacity, *, protection=ENERGY_UNCERTAINTY_RATIO,
                          reserve=SAFETY_RESERVE_RATIO, scale_deadlines=True,
                          alpha=ALPHA_DELAY, delta=DELTA_RISK_REWARD):
    """Change admission or cost while keeping the reference scoring criteria fixed."""
    reference_deadline = matrices["deadline"]
    admission_deadline = reference_deadline if scale_deadlines else np.full(len(risk), BASE_DEADLINE_MIN)
    scenario = configure_missions(
        matrices, capacity, protection=protection, reserve=reserve, deadline=admission_deadline,
    )
    assignment = rard_assignment(risk, scenario, alpha=alpha, delta=delta)
    score = score_assignment(assignment, risk, scenario, capacity, evaluation_deadline=reference_deadline)
    score["Covered Missions"] = sum(
        1 for i, j in assignment if scenario["actual"][i, j] <= scenario["robust"][i, j]
    )
    return score


def draw_dispatch_rounds(sample, cols, base_lat, base_lon, n_rounds, seed):
    """Replay each random draw across the baselines and the RARD variants."""
    rng = np.random.default_rng(seed)
    fire_lat = sample[cols["lat"]].to_numpy()
    fire_lon = sample[cols["lon"]].to_numpy()
    fire_risk = sample["FireRiskScore"].to_numpy()
    variants = {
        "Complete RARD": {},
        "Without risk-weighted delay": {"alpha": 0.0},
        "Without risk service reward": {"delta": 0.0},
        "Without risk-scaled deadlines": {"scale_deadlines": False},
        "Without protection level": {"protection": 0.0},
        "Without safety reserve": {"reserve": 0.0},
    }
    draws, comparisons, protection_sweep, ablations = [], [], [], []
    for round_id in range(1, n_rounds + 1):
        fires = rng.choice(len(sample), size=N_ROUND_FIRES, replace=False)
        bases = rng.choice(len(base_lat), size=N_ROUND_UAVS, replace=False)
        draws.append({
            "Round": round_id,
            "Incident Row Indices": json.dumps(fires.tolist()),
            "UAV Base Row Indices": json.dumps(bases.tolist()),
        })
        risk = fire_risk[fires]
        matrices = compute_matrices(
            fire_lat[fires], fire_lon[fires], risk, base_lat[bases], base_lon[bases],
            rng=rng, export=False,
        )
        for model, assignment in run_models(risk, matrices).items():
            comparisons.append({
                "Round": round_id, "Model": model, "Battery Capacity (Wh)": BATTERY_CAPACITY_WH,
                **score_assignment(assignment, risk, matrices, BATTERY_CAPACITY_WH),
            })
        for margin in PROTECTION_LEVELS:
            protection_sweep.append({
                "Round": round_id, "Protection Level": margin, "Battery Capacity (Wh)": CONSTRAINED_BATTERY_WH,
                **evaluate_rard_variant(risk, matrices, CONSTRAINED_BATTERY_WH, protection=margin),
            })
        for capacity in (BATTERY_CAPACITY_WH, CONSTRAINED_BATTERY_WH):
            for name, settings in variants.items():
                ablations.append({
                    "Round": round_id, "Variant": name, "Battery Capacity (Wh)": capacity,
                    **evaluate_rard_variant(risk, matrices, capacity, **settings),
                })
    return tuple(pd.DataFrame(rows) for rows in (draws, comparisons, protection_sweep, ablations))


def bootstrap_means(values, rng, n_resamples):
    """Resample paired differences by round and return their bootstrap means."""
    means = np.empty(n_resamples)
    batch_size = max(1, 1_000_000 // len(values))
    for start in range(0, n_resamples, batch_size):
        stop = min(start + batch_size, n_resamples)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    return means


def summarize_comparisons(comparisons, rng, n_resamples):
    metrics = ["Dispatched Missions", "Average Response Delay (min)", "Risk-Weighted Delay", "Unserved Fire Risk"]
    summary = comparisons.groupby("Model", sort=False)[metrics].agg(["mean", "std"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]

    paired_rows = []
    for metric in ("Risk-Weighted Delay", "Unserved Fire Risk"):
        paired = comparisons.pivot(index="Round", columns="Model", values=metric)
        for baseline in comparisons["Model"].unique():
            if baseline == "Proposed":
                continue
            differences = (paired["Proposed"] - paired[baseline]).to_numpy()
            bootstrap = bootstrap_means(differences, rng, n_resamples)
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
            baseline_mean = paired[baseline].mean()
            paired_rows.append({
                "Metric": metric, "Baseline": baseline, "Paired Rounds": len(differences),
                "Baseline Mean": baseline_mean, "RARD Mean": paired["Proposed"].mean(),
                "Mean Difference (RARD - Baseline)": differences.mean(),
                "95% CI Lower": lower, "95% CI Upper": upper,
                "Rounds Improved (%)": 100 * np.mean(differences < 0),
                "Reduction (%)": -100 * differences.mean() / baseline_mean if baseline_mean else np.nan,
            })
            if metric == "Risk-Weighted Delay" and baseline == "Nearest Fire":
                nearest_bootstrap = bootstrap
    return summary.reset_index(), pd.DataFrame(paired_rows), nearest_bootstrap


def summarize_robustness(rounds, group_columns):
    summary = rounds.groupby(group_columns, sort=False).agg(
        mean_dispatched=("Dispatched Missions", "mean"),
        total_dispatched=("Dispatched Missions", "sum"),
        total_safe_returns=("Safe Returns", "sum"),
        total_covered=("Covered Missions", "sum"),
        mean_risk_delay=("Risk-Weighted Delay", "mean"),
        mean_unserved_risk=("Unserved Fire Risk", "mean"),
    ).reset_index()
    summary["Safe Returns (%)"] = 100 * summary["total_safe_returns"] / summary["total_dispatched"]
    summary["Measured Coverage"] = summary["total_covered"] / summary["total_dispatched"]
    return summary


def summarize_paired_safety(ablations, rng, n_resamples):
    constrained = ablations[ablations["Battery Capacity (Wh)"] == CONSTRAINED_BATTERY_WH]
    ratios = constrained.assign(
        ratio=100 * constrained["Safe Returns"] / constrained["Dispatched Missions"],
    ).pivot(index="Round", columns="Variant", values="ratio")
    rows = []
    for variant in ("Without protection level", "Without safety reserve"):
        differences = (ratios["Complete RARD"] - ratios[variant]).dropna().to_numpy()
        mean, lower, upper = np.nan, np.nan, np.nan
        if differences.size:
            bootstrap = bootstrap_means(differences, rng, n_resamples)
            mean = differences.mean()
            lower, upper = np.quantile(bootstrap, [0.025, 0.975])
        rows.append({
            "Variant": variant, "Paired Rounds With Dispatch": len(differences),
            "Mean Safety Gain (percentage points)": mean,
            "95% CI Lower": lower, "95% CI Upper": upper,
        })
    return pd.DataFrame(rows)


def plot_validation(comparisons, nearest_bootstrap, ablation_summary, protection_summary):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), layout="constrained")
    risk_delays = comparisons.pivot(index="Round", columns="Model", values="Risk-Weighted Delay")
    model_order = ["Nearest Fire", "Minimum Delay", "Minimum Energy", "Energy-Delay", "Proposed"]
    axes[0].boxplot(risk_delays[model_order].to_numpy(), tick_labels=["Nearest", "Delay", "Energy", "E-D", "RARD"])
    axes[0].set_title(f"(a) Risk-weighted delay at {BATTERY_CAPACITY_WH:g} Wh")
    axes[0].set_ylabel("Risk-weighted delay")
    axes[1].hist(nearest_bootstrap, bins=30, color="#407da5", edgecolor="white")
    for limit in np.quantile(nearest_bootstrap, [0.025, 0.975]):
        axes[1].axvline(limit, color="#9f2929", linestyle="--")
    axes[1].set_title("(b) Paired bootstrap")
    axes[1].set_xlabel("RARD minus nearest fire")
    axes[1].set_ylabel("Resamples")
    constrained = ablation_summary[ablation_summary["Battery Capacity (Wh)"] == CONSTRAINED_BATTERY_WH]
    short_names = {
        "Complete RARD": "Full", "Without risk-weighted delay": "No risk delay",
        "Without risk service reward": "No risk reward", "Without risk-scaled deadlines": "Fixed deadline",
        "Without protection level": "No margin", "Without safety reserve": "No reserve",
    }
    axes[2].barh(constrained["Variant"].map(short_names), constrained["Safe Returns (%)"], color="#407da5")
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 105)
    axes[2].set_title(f"(c) Safe returns at {CONSTRAINED_BATTERY_WH:g} Wh")
    axes[2].set_xlabel("Safe returns (%)")
    save_fig("fig10_statistical_validation.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), layout="constrained")
    margins = protection_summary["Protection Level"]
    axes[0].plot(margins, protection_summary["Predicted Coverage"], "o-", label="Predicted")
    axes[0].plot(margins, protection_summary["Measured Coverage"], "s--", label="Measured")
    axes[0].set_title("(a) Energy coverage")
    axes[0].set_ylabel("Coverage probability")
    axes[0].legend()
    service, = axes[1].plot(margins, protection_summary["mean_dispatched"], "o-", label="Dispatched")
    safe_axis = axes[1].twinx()
    safety, = safe_axis.plot(margins, protection_summary["Safe Returns (%)"], "s--", color="#9f2929", label="Safe returns")
    axes[1].set_title(f"(b) Service and safety at {CONSTRAINED_BATTERY_WH:g} Wh")
    axes[1].set_ylabel("Dispatched missions per round")
    safe_axis.set_ylabel("Safe returns (%)")
    safe_axis.set_ylim(0, 105)
    axes[1].legend(handles=[service, safety], loc="lower left")
    for axis in axes:
        axis.set_xlabel("Protection level")
        axis.set_xticks(PROTECTION_LEVELS)
    save_fig("fig11_protection_level_sweep.png")
    plt.close(fig)


def save_randomized_config(n_rounds, seed, n_resamples):
    output = Path(OUTPUT_DIR)
    config = {
        "rounds": n_rounds,
        "incidents_per_round": N_ROUND_FIRES,
        "uavs_per_round": N_ROUND_UAVS,
        "sampling": "without replacement in draw order using saved coordinates and risk scores",
        "round_seed": seed,
        "bootstrap_seed": seed + 1,
        "bootstrap_resamples": n_resamples,
        "bootstrap_confidence_level": 0.95,
        "nominal_battery_wh": BATTERY_CAPACITY_WH,
        "constrained_battery_wh": CONSTRAINED_BATTERY_WH,
        "uav_speed_mps": UAV_SPEED_MPS,
        "flight_power_w": FLIGHT_POWER_W,
        "hover_power_w": HOVER_POWER_W,
        "hover_time_min": HOVER_TIME_MIN,
        "wind_penalty_max": WIND_PENALTY_MAX,
        "energy_disturbance_max": ENERGY_DISTURBANCE_MAX,
        "protection_level": ENERGY_UNCERTAINTY_RATIO,
        "protection_sweep": PROTECTION_LEVELS,
        "safety_reserve_ratio": SAFETY_RESERVE_RATIO,
        "base_deadline_min": BASE_DEADLINE_MIN,
        "minimum_deadline_min": MIN_DEADLINE_MIN,
        "deadline_sensitivity": DEADLINE_SENSITIVITY,
        "alpha_delay": ALPHA_DELAY,
        "beta_energy": BETA_ENERGY,
        "delta_risk_reward": DELTA_RISK_REWARD,
        "input_sha256": {
            name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("preprocessed_fire_sample.csv", "uav_base_locations.csv")
        },
    }
    (output / "randomized_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


def run_randomized_experiment(sample, cols, base_lat, base_lon, n_rounds, seed, n_resamples):
    draws, comparisons, protection, ablations = draw_dispatch_rounds(
        sample, cols, base_lat, base_lon, n_rounds, seed,
    )
    bootstrap_rng = np.random.default_rng(seed + 1)
    summary, paired, nearest_bootstrap = summarize_comparisons(comparisons, bootstrap_rng, n_resamples)
    protection_summary = summarize_robustness(protection, ["Protection Level"])
    protection_summary["Predicted Coverage"] = (
        protection_summary["Protection Level"] / ENERGY_DISTURBANCE_MAX
    ).clip(upper=1.0)
    ablation_summary = summarize_robustness(ablations, ["Battery Capacity (Wh)", "Variant"])
    paired_safety = summarize_paired_safety(ablations, bootstrap_rng, n_resamples)
    for frame, filename in [
        (draws, "randomized_round_inputs.csv"),
        (comparisons, "randomized_rounds.csv"),
        (summary, "randomized_summary.csv"),
        (paired, "paired_bootstrap.csv"),
        (protection, "protection_rounds.csv"),
        (protection_summary, "protection_summary.csv"),
        (ablations, "ablation_rounds.csv"),
        (ablation_summary, "ablation_summary.csv"),
        (paired_safety, "ablation_paired_safety.csv"),
    ]:
        save_df(frame, filename)
    save_randomized_config(n_rounds, seed, n_resamples)
    plot_validation(comparisons, nearest_bootstrap, ablation_summary, protection_summary)
    print(f"\nCompleted {n_rounds} paired dispatch rounds with {n_resamples:,} bootstrap resamples.")
    print(summary.to_string(index=False))


def main(argv=None):
    global DATASET_PATH, OUTPUT_DIR, FIGURES_DIR
    parser = argparse.ArgumentParser(description="Reproduce the RARD dispatching experiments.")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--input-dir", type=Path, help="Directory containing the saved incident and base CSVs.")
    inputs.add_argument("--dataset", type=Path, help="Original NIFC wildfire records CSV.")
    parser.add_argument("--output-dir", type=Path, help="Directory for results and figures.")
    parser.add_argument("--experiment", choices=("all", "single", "randomized", "sequential"), default="all")
    parser.add_argument("--dqn-model", type=Path, help="Saved .npz checkpoint required for --experiment sequential.")
    parser.add_argument("--rounds", type=int, default=N_RANDOMIZED_ROUNDS)
    parser.add_argument("--round-seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--bootstrap-resamples", type=int, default=N_BOOTSTRAP_RESAMPLES)
    args = parser.parse_args(argv)
    if args.experiment == "sequential" and args.dqn_model is None:
        parser.error("--experiment sequential requires --dqn-model PATH.")
    if args.rounds < 2 or args.round_seed < 0 or args.bootstrap_resamples < 2:
        parser.error("Use at least two rounds and bootstrap resamples and a nonnegative round seed.")
    if args.output_dir is not None:
        OUTPUT_DIR = str(args.output_dir)
        FIGURES_DIR = str(args.output_dir / "figures")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    if args.dataset is not None:
        DATASET_PATH = str(args.dataset)

    input_dir = args.input_dir
    if input_dir is None and args.dataset is None and not Path(DATASET_PATH).is_file():
        bundled = Path(__file__).resolve().parent / "data"
        if (bundled / "preprocessed_fire_sample.csv").is_file() and (bundled / "uav_base_locations.csv").is_file():
            input_dir = bundled
    try:
        if input_dir is not None:
            sample, cols, base_lat, base_lon = load_saved_inputs(input_dir)
        else:
            sample, cols, historical = load_dataset()
            base_lat, base_lon = create_bases(historical[:, 0], historical[:, 1])
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    if args.experiment in ("all", "randomized") and (len(sample) < N_ROUND_FIRES or len(base_lat) < N_ROUND_UAVS):
        parser.error(f"The randomized experiment requires at least {N_ROUND_FIRES} incidents and {N_ROUND_UAVS} bases.")
    if args.experiment == "sequential":
        try:
            net = load_dqn_network(args.dqn_model)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            parser.error(str(exc))
        flat, flon, risk = sample[cols["lat"]].values, sample[cols["lon"]].values, sample["FireRiskScore"].values
        comp = compare_sequential_policies(net, flat, flon, risk, base_lat, base_lon)
        plot_policy_figures(comp)
        print(comp.to_string(index=False))
    if args.experiment in ("all", "single"):
        np.random.seed(RANDOM_SEED)
        run_single_round(sample, cols, base_lat, base_lon)
    if args.experiment in ("all", "randomized"):
        run_randomized_experiment(
            sample, cols, base_lat, base_lon, args.rounds, args.round_seed, args.bootstrap_resamples,
        )


if __name__ == "__main__":
    main()
