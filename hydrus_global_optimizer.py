"""
hydrus_global_optimizer.py
=============================
Batch Global Optimization for HYDRUS 2D.
Features:
- Dynamic Header Parser (Immune to module toggling and shifting columns)
- Automated batch processing via batch_runs.csv (Dynamic Columns)
- Log-space optimization for kinetic/capacity parameters
- PHYSICS-GUIDED LOSS: Log-NRMSE + KGE + Weak-Form Mass Balance
- Local Hessian approximation for Standard Errors (Delta method applied)
- Stateful saving (prevents overwriting previous batches on restart)
"""

import sys
import time
import subprocess
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import interp1d
from scipy.optimize import differential_evolution, minimize

try:
    from scipy.integrate import trapezoid as _trapz
except ImportError:
    from scipy.integrate import trapz as _trapz

warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROJECT PATHS
# ─────────────────────────────────────────────────────────────────────────────

PROJECT_DIR    = Path(r"YOUR HYDRUS PERMANENT DIRECTORY")
HYDRUS_EXE     = PROJECT_DIR / "H2D_Calc64.exe"
SELECTOR_IN    = PROJECT_DIR / "Selector.in"
OBSNOD_OUT     = PROJECT_DIR / "ObsNod.out"
BATCH_FILE     = PROJECT_DIR / "batch_runs.csv"
OUTPUT_CSV     = PROJECT_DIR / "optim_master_results.csv"
ERROR_LOG      = PROJECT_DIR / "patch_errors.log"

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

RANDOM_SEED  = 42
_NO_WINDOW   = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)

VALID_PARAMS = {
    "Bulk.d", "DisperL", "DisperT", "Frac", "ThImob", "DifW", "DifG",
    "Ks", "Nu", "Beta", "Henry", "SnkL1", "SnkS1", "SnkG1", 
    "SnkL1p", "SnkS1p", "SnkG1p", "SnkL0", "SnkS0", "SnkG0", "Alfa",
    "Thr", "Ths", "Hyd_Alfa", "Hyd_n", "Hyd_Ks", "Hyd_l"
}

MODEL_CONFIGS = {
    "params": [
        dict(name="DisperL", active=False, fixed_value=0.5),
        dict(name="DisperT", active=False, fixed_value=0.1),
        dict(name="DifW",    active=False, fixed_value=0.0),
        
        dict(name="Ks",    active=True, is_log=True, lo=0.01, hi=5.0),
        dict(name="Beta",  active=False, fixed_value=0.5), # is_log=False, lo=0.4, hi=0.6
        
        dict(name="Frac",  active=True,is_log=False, lo=0.01, hi=0.99), #  fixed_value=1.00
        dict(name="Alfa",  active=True, is_log=True,  lo=1e-5, hi=1.0),
        
        dict(name="SnkL1", active=False, fixed_value=0.0),
        dict(name="SnkS1", active=True,  is_log=False, lo=1e-4, hi=0.02),
        
        dict(name="ThImob", active=False, fixed_value=0.0),
    ]
}

OPTIM_CONFIG = dict(
    popsize        = 10,
    maxiter        = 30,
    mutation       = (0.5, 1.2),
    recombination  = 0.7,
    hydrus_timeout = 300,
)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — INITIALIZATION & GLOBALS
# ─────────────────────────────────────────────────────────────────────────────

ACTIVE = [p for p in MODEL_CONFIGS["params"] if p.get("active", False)]
ACTIVE_NAMES = [p["name"] for p in ACTIVE]
BASE_FIXED = {p["name"]: p["fixed_value"] for p in MODEL_CONFIGS["params"] if not p.get("active", False)}

BOUNDS = []
for p in ACTIVE:
    if p.get("is_log", False): BOUNDS.append((np.log10(p["lo"]), np.log10(p["hi"])))
    else: BOUNDS.append((p["lo"], p["hi"]))

_current_obs_times = None
_current_obs_concs = None
_current_t_max = 0.0
_current_t_pulse = 0.0
_current_obs_mass = 0.0
_current_obs_mean = 0.0
_current_obs_std = 0.0
_current_fixed_params = {}
_run_count = 0
_best_loss = np.inf
_BASE_LINES = []

def get_tpulse(lines: list) -> float:
    """Extracts tPulse from Selector.in Block D for mass balance checking."""
    for i, line in enumerate(lines):
        if "tpulse" in line.lower():
            if i + 1 < len(lines):
                return float(lines[i+1].split()[0])
    return 0.0

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — DYNAMIC HYDRUS PARSER
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    return "0" if v == 0.0 else f"{v:.6g}"

def modify_selector(base_lines: list, params: dict) -> list:
    lines = base_lines.copy()
    for i, line in enumerate(lines):
        tokens = line.split()
        if not tokens: continue
        
        # Block C: Hydraulics
        if "thr" in tokens and "ths" in tokens and "n" in tokens:
            if i + 1 < len(lines):
                val_tokens = lines[i+1].split()
                for p_name, h_name in [("Thr", "thr"), ("Ths", "ths"), ("Hyd_Alfa", "Alfa"), ("Hyd_n", "n"), ("Hyd_Ks", "Ks"), ("Hyd_l", "l")]:
                    if p_name in params and h_name in tokens:
                        idx = tokens.index(h_name)
                        if idx < len(val_tokens): val_tokens[idx] = _fmt(params[p_name])
                lines[i+1] = "    ".join(val_tokens) + "\n"
                
        # Block G: Bulk and Dispersion
        elif "Bulk.d." in tokens and "DisperL" in tokens:
            if i + 1 < len(lines):
                val_tokens = lines[i+1].split()
                for p_name, h_name in [("Bulk.d", "Bulk.d."), ("DisperL", "DisperL"), ("DisperT", "DisperT"), ("Frac", "Frac"), ("ThImob", "ThImob")]:
                    if p_name in params and h_name in tokens:
                        idx = tokens.index(h_name)
                        if idx < len(val_tokens): val_tokens[idx] = _fmt(params[p_name])
                lines[i+1] = "    ".join(val_tokens) + "\n"
        
        # Block G: Reactive
        elif "Ks" in tokens and "Beta" in tokens and "Nu" in tokens:
            if i + 1 < len(lines):
                val_tokens = lines[i+1].split()
                for p_name, h_name in [("Ks", "Ks"), ("Beta", "Beta"), ("SnkL1", "SnkL1"), ("SnkS1", "SnkS1"), ("Alfa", "Alfa")]:
                    if p_name in params and h_name in tokens:
                        idx = tokens.index(h_name)
                        if idx < len(val_tokens): val_tokens[idx] = _fmt(params[p_name])
                lines[i+1] = "    ".join(val_tokens) + "\n"
                
    return lines

def _verify_patch(params: dict) -> list:
    try:
        with open(SELECTOR_IN, "r") as f: disk = f.readlines()
    except OSError as e:
        return [f"File read error: {e}"]
    
    errors = []
    for i, line in enumerate(disk):
        tokens = line.split()
        if not tokens: continue
        
        if "thr" in tokens and "ths" in tokens and "n" in tokens and i + 1 < len(disk):
            val_tokens = disk[i+1].split()
            for p_name, h_name in [("Thr", "thr"), ("Ths", "ths"), ("Hyd_Alfa", "Alfa"), ("Hyd_n", "n"), ("Hyd_Ks", "Ks"), ("Hyd_l", "l")]:
                if p_name in params and h_name in tokens:
                    idx = tokens.index(h_name)
                    if idx < len(val_tokens):
                        try:
                            written, expected = float(val_tokens[idx]), float(params[p_name])
                            if abs(written - expected) > abs(expected) * 1e-4 + 1e-10:
                                errors.append(f"{p_name}: expected {expected:.6g} got {written:.6g}")
                        except ValueError: errors.append(f"{p_name} non-numeric")

        elif "Bulk.d." in tokens and "DisperL" in tokens and i + 1 < len(disk):
            val_tokens = disk[i+1].split()
            for p_name, h_name in [("Bulk.d", "Bulk.d."), ("DisperL", "DisperL"), ("DisperT", "DisperT"), ("Frac", "Frac"), ("ThImob", "ThImob")]:
                if p_name in params and h_name in tokens:
                    idx = tokens.index(h_name)
                    if idx < len(val_tokens):
                        try:
                            written, expected = float(val_tokens[idx]), float(params[p_name])
                            if abs(written - expected) > abs(expected) * 1e-4 + 1e-10:
                                errors.append(f"{p_name}: expected {expected:.6g} got {written:.6g}")
                        except ValueError: errors.append(f"{p_name} non-numeric")

        elif "Ks" in tokens and "Beta" in tokens and "Nu" in tokens and i + 1 < len(disk):
            val_tokens = disk[i+1].split()
            for p_name, h_name in [("Ks", "Ks"), ("Beta", "Beta"), ("SnkL1", "SnkL1"), ("SnkS1", "SnkS1"), ("Alfa", "Alfa")]:
                if p_name in params and h_name in tokens:
                    idx = tokens.index(h_name)
                    if idx < len(val_tokens):
                        try:
                            written, expected = float(val_tokens[idx]), float(params[p_name])
                            if abs(written - expected) > abs(expected) * 1e-4 + 1e-10:
                                errors.append(f"{p_name}: expected {expected:.6g} got {written:.6g}")
                        except ValueError: errors.append(f"{p_name} non-numeric")
    return errors

def write_and_verify(params: dict) -> bool:
    lines = modify_selector(_BASE_LINES, params)
    with open(SELECTOR_IN, "w") as f: 
        f.writelines(lines)
        
    errs = _verify_patch(params)
    if errs:
        with open(ERROR_LOG, "a") as f:
            f.write(f"Run {_run_count} Errors: {errs}\n")
        return False
    return True

def read_obsnod() -> "pd.DataFrame | None":
    rows = []; in_data = False
    try:
        with open(OBSNOD_OUT, "r") as fh:
            for line in fh:
                if not in_data:
                    if line.strip().lower().startswith("time"): in_data = True
                    continue
                parts = line.split()
                if len(parts) < 5: continue
                try: rows.append((float(parts[0]), float(parts[4])))
                except ValueError: continue
    except FileNotFoundError: return None
    
    if not rows: return None
    return pd.DataFrame(rows, columns=["time","Conc"]).drop_duplicates("time").sort_values("time").reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — PHYSICS-GUIDED OBJECTIVE FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def objective_function(theta: np.ndarray) -> float:
    global _run_count, _best_loss
    _run_count += 1 

    params = dict(_current_fixed_params)
    for name, val in zip(ACTIVE_NAMES, theta):
        p_dict = next(p for p in ACTIVE if p["name"] == name)
        params[name] = float(10**val if p_dict.get("is_log", False) else val)
    
    if not write_and_verify(params): 
        sys.stdout.write(f"\r  Running... [{_run_count:04d}] Last Loss: 1e6 (Patch Error)      ")
        sys.stdout.flush()
        return 1e6
    
    try:
        p = subprocess.run([str(HYDRUS_EXE)], cwd=str(PROJECT_DIR), 
                           timeout=OPTIM_CONFIG["hydrus_timeout"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                           creationflags=_NO_WINDOW)
        if p.returncode != 0: 
            sys.stdout.write(f"\r  Running... [{_run_count:04d}] Last Loss: 1e6 (Crash)      ")
            sys.stdout.flush()
            return 1e6
    except subprocess.TimeoutExpired:
        sys.stdout.write(f"\r  Running... [{_run_count:04d}] Last Loss: 1e6 (Timeout)      ")
        sys.stdout.flush()
        return 1e6

    sim_df = read_obsnod()
    if sim_df is None: return 1e6
    
    st, sc = sim_df["time"].values, sim_df["Conc"].values
    if len(st) < 2 or st[-1] < _current_t_max: return 1e6
    
    # Fill value set to 0.0 to prevent NaNs during log transform
    fn = interp1d(st, sc, kind="linear", bounds_error=False, fill_value=0.0)
    sim_at_obs = fn(_current_obs_times)

    # 1. PHYSICS-GUIDED PENALTY: Log-NRMSE 
    eps = 1e-5
    log_obs = np.log10(np.maximum(_current_obs_concs, 0) + eps)
    # Clip sim_at_obs to 0 to prevent negative logs from numerical oscillations
    log_sim = np.log10(np.maximum(sim_at_obs, 0) + eps)
    log_rmse = np.sqrt(np.mean((log_obs - log_sim)**2))

    # 2. KGE Shape Penalty
    sim_mean = np.mean(sim_at_obs)
    sim_std = np.std(sim_at_obs)
    r = np.corrcoef(_current_obs_concs, sim_at_obs)[0, 1] if (_current_obs_std > 0 and sim_std > 0) else 0.0
    beta = sim_mean / _current_obs_mean if _current_obs_mean > 0 else 1.0
    gamma = (sim_std * _current_obs_mean) / (sim_mean * _current_obs_std) if (_current_obs_std > 0 and sim_mean > 1e-12) else 1.0
    kge = 1.0 - np.sqrt((r - 1.0)**2 + (beta - 1.0)**2 + (gamma - 1.0)**2)

    # 3. PHYSICS-GUIDED PENALTY: Weak-Form Mass Balance
    sim_mass = float(_trapz(y=sc, x=st))
    mass_penalty = 0.0
    if sim_mass < (_current_obs_mass * 0.7):
        # Heavy penalty if simulated mass is significantly lower than observed mass
        mass_penalty = 2.0 * (_current_obs_mass / (sim_mass + eps))

    total_loss = float(log_rmse + (1.0 - kge) + mass_penalty)

    sys.stdout.write(f"\r  Running... [{_run_count:04d}] Last Loss: {total_loss:.5f}      ")
    sys.stdout.flush()

    if total_loss < _best_loss and total_loss < 999999:
        _best_loss = total_loss
        tag = "  ".join(f"{n}={params[n]:.4g}" for n in ACTIVE_NAMES)
        print(f"\n  >>> [{_run_count:04d}] NEW BEST LOSS: {total_loss:.5f} (MassErr: {abs(sim_mass-_current_obs_mass):.3f}) | {tag}")

    return total_loss

def direct_fit_check(best_params: dict, obs_times: np.ndarray, obs_concs: np.ndarray, filename: str, final_loss: float, run_idx: int) -> None:
    try: 
        import matplotlib.pyplot as plt
    except ImportError: 
        return
    
    if not write_and_verify(best_params): return
    
    subprocess.run([str(HYDRUS_EXE)], cwd=str(PROJECT_DIR), 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                   creationflags=_NO_WINDOW)
    
    sim_df = read_obsnod()
    if sim_df is None: return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(obs_times, obs_concs, "o", color="#D85A30", markersize=5, zorder=5, label="Observed", alpha=0.9)
    ax.plot(sim_df["time"], sim_df["Conc"], color="#1D9E75", linewidth=2.0, label=f"Best Fit (Loss={final_loss:.4f})")
    
    ax.set_xlabel("Time (min)"); ax.set_ylabel("C/C0")
    ax.set_title(f"Optimized Fit: {filename} (Batch Row {run_idx})")
    ax.legend(fontsize=9); plt.tight_layout()
    
    out_name = f"optim_fit_{Path(filename).stem}_batch{run_idx}.png"
    plt.savefig(PROJECT_DIR / out_name, dpi=150); plt.close()
    print(f"  Plot saved to {out_name}")

# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — BATCH MANAGER
# ─────────────────────────────────────────────────────────────────────────────

def get_batch_runs() -> pd.DataFrame:
    if not BATCH_FILE.exists():
        print(f"\n[INFO] {BATCH_FILE.name} not found. Generating dynamic template...")
        df = pd.DataFrame([{
            "filename": "observed.csv", 
            "Bulk.d": 1.5,
            "Ths": 0.43,
            "Hyd_Ks": 29.8,
            "DisperL": 0.5, 
            "DisperT": 0.1, 
            "DifW": 0.0
        }])
        df.to_csv(BATCH_FILE, index=False)
        print(f"[ACTION REQUIRED] Please edit {BATCH_FILE.name} with your file names and specific material properties, then run this script again.")
        sys.exit(0)
    return pd.read_csv(BATCH_FILE)

def main() -> None:
    global _BASE_LINES, _current_obs_times, _current_obs_concs
    global _current_t_max, _current_t_pulse, _current_obs_mass
    global _current_obs_mean, _current_obs_std
    global _current_fixed_params, _run_count, _best_loss

    if RANDOM_SEED is not None: np.random.seed(RANDOM_SEED)

    with open(SELECTOR_IN, "r") as f: _BASE_LINES = f.readlines()
    _current_t_pulse = get_tpulse(_BASE_LINES)
    
    batch_df = get_batch_runs()
    
    # Load historical results to prevent overwriting
    master_results = []
    if OUTPUT_CSV.exists():
        try:
            master_results = pd.read_csv(OUTPUT_CSV).to_dict('records')
            print(f"  [INFO] Loaded {len(master_results)} previous results from {OUTPUT_CSV.name}.")
        except Exception as e:
            print(f"  [WARNING] Could not read existing results: {e}. Output will be overwritten.")

    print(f"\n{'='*60}")
    print(f"  HYDRUS Batch Optimizer (DE + Hessian)")
    print(f"  Found {len(batch_df)} experiments in batch list.")
    print(f"  Optimizing {len(ACTIVE_NAMES)} parameters: {ACTIVE_NAMES}")
    print(f"{'='*60}\n")

    for idx, row in batch_df.iterrows():
        obs_file = PROJECT_DIR / str(row['filename']).strip()
        if not obs_file.exists():
            print(f"\n[SKIPPED] {row['filename']} not found.")
            continue

        print(f"\n>>> PROCESSING: {row['filename']} (Batch Row {idx} | tPulse={_current_t_pulse})")
        
        df = pd.read_csv(obs_file); df.columns = df.columns.str.strip()
        df = df[["Time","Conc"]].dropna().sort_values("Time").reset_index(drop=True)
        _current_obs_times = df["Time"].values.astype(float)
        _current_obs_concs = df["Conc"].values.astype(float)
        _current_t_max = float(_current_obs_times.max())
        _current_obs_mass = float(_trapz(y=_current_obs_concs, x=_current_obs_times))
        _current_obs_mean = np.mean(_current_obs_concs)
        _current_obs_std = np.std(_current_obs_concs)

        _current_fixed_params = dict(BASE_FIXED)
        applied_overrides = []
        for col in row.index:
            col_clean = str(col).strip()
            if col_clean in VALID_PARAMS and col_clean not in ACTIVE_NAMES and not pd.isna(row[col]):
                try:
                    val = float(row[col])
                    _current_fixed_params[col_clean] = val
                    applied_overrides.append(f"{col_clean}={val}")
                except ValueError:
                    pass
        
        if applied_overrides:
            print(f"  Applied Fixed Properties: {', '.join(applied_overrides)}")

        _run_count = 0
        _best_loss = np.inf
        t0 = time.time()

        result = differential_evolution(
            objective_function, 
            bounds=BOUNDS,
            strategy='best1bin',
            maxiter=OPTIM_CONFIG["maxiter"],
            popsize=OPTIM_CONFIG["popsize"],
            mutation=OPTIM_CONFIG["mutation"],
            recombination=OPTIM_CONFIG["recombination"],
            seed=RANDOM_SEED,
            disp=False
        )

        print(f"\n  Polishing fit and calculating uncertainty (Hessian)...")
        polish_res = minimize(objective_function, result.x, method='L-BFGS-B', bounds=BOUNDS)
        
        try:
            hess_inv_op = polish_res.hess_inv
            if hasattr(hess_inv_op, 'todense'):
                dense_hess = hess_inv_op.todense()
            else:
                dense_hess = hess_inv_op @ np.eye(len(polish_res.x))
            std_errors = np.sqrt(np.maximum(0, np.diag(dense_hess)))
        except Exception as e:
            print(f"  [WARNING] Hessian extraction failed: {e}")
            std_errors = np.zeros_like(polish_res.x)

        final_x = polish_res.x
        final_loss = polish_res.fun

        print(f"\n  Experiment Complete: {row['filename']}")
        print(f"  Runs: {_run_count} | Time: {(time.time() - t0) / 60:.1f} min | Final Loss: {final_loss:.5f}")
        
        res_row = {"filename": row['filename'], "Loss": final_loss}
        
        for col in row.index:
            col_clean = str(col).strip()
            if col_clean in VALID_PARAMS and col_clean not in ACTIVE_NAMES and not pd.isna(row[col]):
                try:
                    res_row[f"fixed_{col_clean}"] = float(row[col])
                except ValueError:
                    pass

        for i, name in enumerate(ACTIVE_NAMES):
            p_dict = next(p for p in ACTIVE if p["name"] == name)
            is_log = p_dict.get("is_log", False)
            
            val = float(10**final_x[i] if is_log else final_x[i])
            if is_log:
                se = float(val * np.log(10) * std_errors[i])
            else:
                se = float(std_errors[i])
            
            print(f"    {name:<10} = {val:.5g} ± {se:.3g}")
            res_row[name] = val
            res_row[f"{name}_SE"] = se

        # Append new row to historical data and overwrite CSV
        master_results.append(res_row)
        pd.DataFrame(master_results).to_csv(OUTPUT_CSV, index=False)

        best_params_dict = dict(_current_fixed_params)
        for i, name in enumerate(ACTIVE_NAMES):
            p_dict = next(p for p in ACTIVE if p["name"] == name)
            best_params_dict[name] = float(10**final_x[i] if p_dict.get("is_log", False) else final_x[i])
            
        direct_fit_check(best_params_dict, _current_obs_times, _current_obs_concs, str(row['filename']), final_loss, idx)

    print(f"\n{'='*60}")
    print(f"  ALL BATCHES COMPLETE.")
    print(f"  Results saved to: {OUTPUT_CSV.name}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()