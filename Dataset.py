import os
import copy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw

# ============================================================
# Reproducibility
# ============================================================
SEED = 42
rng = np.random.default_rng(SEED)

# ============================================================
# Settings
# ============================================================
out_dir = r"C:\Users\mahao\gnn_case118_dataset"
os.makedirs(out_dir, exist_ok=True)

# Number of samples per outage per split
n_per_cont_train = 200
n_per_cont_val   = 60
n_per_cont_test  = 60

# Fixed outage branch sets 
train_outage_lines = [36, 33, 8, 93, 94]
val_outage_lines   = [36, 33, 8, 93, 94]
test_outage_lines  = [38, 33, 142, 31, 93]

# Renewable buses in 0-based pandapower indexing
# MATPOWER bus 10 -> pp index 9
# MATPOWER bus 57 -> pp index 56
pv_bus = 9
wt_bus = 56

pv_cap_mw = 40.0
wt_cap_mw = 50.0

load_scale_min = 0.75
load_scale_max = 1.20

pv_scale_min = 0.0
pv_scale_max = 1.0

wt_scale_min = 0.2
wt_scale_max = 1.0

max_attempt_factor = 20  # max attempts = requested_samples * this

# ============================================================
# Helper functions
# ============================================================
def assign_rateA_from_voltage(net):
    rateA = np.zeros(len(net.line), dtype=float)
    for i, row in net.line.iterrows():
        fb = int(row.from_bus)
        vn = float(net.bus.loc[fb, "vn_kv"])
        if abs(vn - 345.0) < 1e-3:
            rateA[i] = 700.0
        elif abs(vn - 161.0) < 1e-3:
            rateA[i] = 220.0
        else:
            rateA[i] = 180.0
    return rateA


def export_branch_data_csv(net, path_csv):
    rateA = assign_rateA_from_voltage(net)
    rows = []

    for i, row in net.line.iterrows():
        r_total = float(row.r_ohm_per_km * row.length_km)
        x_total = float(row.x_ohm_per_km * row.length_km)
        b_total = float(row.c_nf_per_km * row.length_km)
        ratio = 1.0

        rows.append({
            "from_bus": int(row.from_bus) + 1,  # save as 1-based
            "to_bus": int(row.to_bus) + 1,
            "r": r_total,
            "x": x_total,
            "b": b_total,
            "rateA": float(rateA[i]),
            "ratio": ratio
        })

    df = pd.DataFrame(rows)
    df.to_csv(path_csv, index=False)
    print(f"Saved branch data: {path_csv}")


def add_renewables(net, pv_bus, wt_bus):
    pp.create_sgen(net, bus=pv_bus, p_mw=0.0, q_mvar=0.0, name="PV")
    pp.create_sgen(net, bus=wt_bus, p_mw=0.0, q_mvar=0.0, name="Wind")
    return net


def apply_random_operating_point(net):
    load_scale = rng.uniform(load_scale_min, load_scale_max)
    pv_scale   = rng.uniform(pv_scale_min, pv_scale_max)
    wt_scale   = rng.uniform(wt_scale_min, wt_scale_max)

    net.load["p_mw"] *= load_scale
    if "q_mvar" in net.load.columns:
        net.load["q_mvar"] *= load_scale

    net.sgen.loc[net.sgen["name"] == "PV", "p_mw"] = pv_scale * pv_cap_mw
    net.sgen.loc[net.sgen["name"] == "PV", "q_mvar"] = 0.0

    net.sgen.loc[net.sgen["name"] == "Wind", "p_mw"] = wt_scale * wt_cap_mw
    net.sgen.loc[net.sgen["name"] == "Wind", "q_mvar"] = 0.0


def make_one_row_from_results(net, sample_id):
    """
    Row format:
    [sample_id, P1, Q1, V1, d1, P2, Q2, V2, d2, ..., P118, Q118, V118, d118]
    """
    row = [sample_id]
    for b in net.bus.index:
        P = float(net.res_bus.loc[b, "p_mw"])
        Q = float(net.res_bus.loc[b, "q_mvar"])
        V = float(net.res_bus.loc[b, "vm_pu"])
        d = float(net.res_bus.loc[b, "va_degree"])
        row.extend([P, Q, V, d])
    return row


def save_dataset(rows, contingency_rows, prefix):
    cols = ["sample_id"]
    n_bus = (len(rows[0]) - 1) // 4
    for b in range(1, n_bus + 1):
        cols.extend([f"P_{b}", f"Q_{b}", f"V_{b}", f"delta_{b}"])

    df = pd.DataFrame(rows, columns=cols)
    xlsx_path = os.path.join(out_dir, f"{prefix}.xlsx")
    df.to_excel(xlsx_path, index=False)

    cont_df = pd.DataFrame(contingency_rows, columns=["outage_from", "outage_to", "outage_line_idx"])
    cont_path = os.path.join(out_dir, f"{prefix}_contingency_info.csv")
    cont_df.to_csv(cont_path, index=False)

    print(f"Saved {prefix}:")
    print(f"  {xlsx_path}")
    print(f"  {cont_path}")


def generate_split(net_base, outage_lines, n_per_cont, prefix, start_sample_id=0):
    rows = []
    contingency_rows = []
    sample_id = start_sample_id

    total_target = len(outage_lines) * n_per_cont
    max_attempts = total_target * max_attempt_factor
    attempts = 0

    counts = {line_idx: 0 for line_idx in outage_lines}

    while sum(counts.values()) < total_target and attempts < max_attempts:
        attempts += 1

        for line_idx in outage_lines:
            if counts[line_idx] >= n_per_cont:
                continue

            net = copy.deepcopy(net_base)
            apply_random_operating_point(net)

            # outage line
            if line_idx not in net.line.index:
                print(f"Warning: line index {line_idx} not found in pandapower net.line")
                continue

            outage_from = int(net.line.loc[line_idx, "from_bus"]) + 1
            outage_to   = int(net.line.loc[line_idx, "to_bus"]) + 1

            net.line.at[line_idx, "in_service"] = False

            try:
                pp.runpp(
                    net,
                    algorithm="nr",
                    calculate_voltage_angles=True,
                    init="dc",
                    enforce_q_lims=True
                )
            except Exception:
                continue

            if not net.converged:
                continue

            row = make_one_row_from_results(net, sample_id)
            rows.append(row)
            contingency_rows.append([outage_from, outage_to, line_idx])

            counts[line_idx] += 1
            sample_id += 1

            if sample_id % 100 == 0:
                print(f"{prefix}: generated {len(rows)} / {total_target}")

    print(f"{prefix}: final counts per outage = {counts}")
    save_dataset(rows, contingency_rows, prefix)
    return sample_id


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Loading IEEE 118-bus network...")
    net_base = nw.case118()

    # Add PV and wind generators
    net_base = add_renewables(net_base, pv_bus, wt_bus)

    # Save branch data file needed by GNN code
    export_branch_data_csv(net_base, os.path.join(out_dir, "case118_branch_data.csv"))

    next_id = 0
    next_id = generate_split(net_base, train_outage_lines, n_per_cont_train, "train", start_sample_id=next_id)
    next_id = generate_split(net_base, val_outage_lines,   n_per_cont_val,   "val",   start_sample_id=next_id)
    next_id = generate_split(net_base, test_outage_lines,  n_per_cont_test,  "test",  start_sample_id=next_id)

    print("\nAll dataset files generated.")
