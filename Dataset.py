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

# Number of successful OPF operating points per split
# Each successful OPF point generates one sample for EACH outage in the split
n_base_train = 200
n_base_val   = 60
n_base_test  = 60

# Outage branch sets (verify these indices against pandapower net.line if needed)
train_outage_lines = [36, 33, 8, 93, 94]
val_outage_lines   = [36, 33, 8, 93, 94]
test_outage_lines  = [38, 33, 142, 31, 93]

# Renewable buses in 0-based pandapower indexing
# MATPOWER bus 10 -> pp bus 9
# MATPOWER bus 57 -> pp bus 56
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

max_attempt_factor = 30   # max attempts = target successful OPF points * factor

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
            "from_bus": int(row.from_bus) + 1,
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


def configure_acopf(net):
    """
    Add/adjust OPF constraints and costs so runopp can work.
    """
    # Voltage limits
    if "min_vm_pu" not in net.bus.columns:
        net.bus["min_vm_pu"] = 0.94
    else:
        net.bus["min_vm_pu"] = 0.94

    if "max_vm_pu" not in net.bus.columns:
        net.bus["max_vm_pu"] = 1.06
    else:
        net.bus["max_vm_pu"] = 1.06

    # Line loading limits for OPF
    if "max_loading_percent" not in net.line.columns:
        net.line["max_loading_percent"] = 100.0
    else:
        net.line["max_loading_percent"] = 100.0

    if len(net.trafo) > 0:
        if "max_loading_percent" not in net.trafo.columns:
            net.trafo["max_loading_percent"] = 100.0
        else:
            net.trafo["max_loading_percent"] = 100.0

    # ext_grid OPF bounds
    if len(net.ext_grid) > 0:
        net.ext_grid["min_p_mw"] = -1e4
        net.ext_grid["max_p_mw"] =  1e4
        net.ext_grid["min_q_mvar"] = -1e4
        net.ext_grid["max_q_mvar"] =  1e4
        net.ext_grid["controllable"] = True

    # generator OPF bounds
    if len(net.gen) > 0:
        if "min_p_mw" not in net.gen.columns:
            net.gen["min_p_mw"] = 0.0
        if "max_p_mw" not in net.gen.columns:
            net.gen["max_p_mw"] = net.gen["p_mw"] + 200.0

        if "min_q_mvar" not in net.gen.columns:
            net.gen["min_q_mvar"] = -500.0
        if "max_q_mvar" not in net.gen.columns:
            net.gen["max_q_mvar"] =  500.0

        net.gen["controllable"] = True

    # Keep renewable sgens fixed
    if len(net.sgen) > 0:
        net.sgen["controllable"] = False

    # Remove old poly costs if any
    if len(net.poly_cost) > 0:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    # Add simple linear costs
    if len(net.ext_grid) > 0:
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, "ext_grid", cp1_eur_per_mw=30.0)

    if len(net.gen) > 0:
        for idx in net.gen.index:
            pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=20.0)


def apply_random_operating_point(net):
    """
    Randomize load scale and renewable outputs.
    """
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


def aggregate_load_by_bus(net):
    """
    Aggregate actual demand by bus from net.load.
    """
    n_bus = len(net.bus)
    Pload = np.zeros(n_bus)
    Qload = np.zeros(n_bus)

    if len(net.load) > 0:
        for _, row in net.load.iterrows():
            b = int(row.bus)
            Pload[b] += float(row.p_mw)
            Qload[b] += float(row.q_mvar)

    return Pload, Qload


def aggregate_generation_by_bus_from_opf(net):
    """
    Aggregate pre-contingency ACOPF generation results by bus:
    ext_grid + gen + sgen
    """
    n_bus = len(net.bus)
    Pgen = np.zeros(n_bus)
    Qgen = np.zeros(n_bus)

    # ext_grid
    if len(net.ext_grid) > 0 and hasattr(net, "res_ext_grid"):
        for idx, row in net.ext_grid.iterrows():
            b = int(row.bus)
            Pgen[b] += float(net.res_ext_grid.loc[idx, "p_mw"])
            Qgen[b] += float(net.res_ext_grid.loc[idx, "q_mvar"])

    # gen
    if len(net.gen) > 0 and hasattr(net, "res_gen"):
        for idx, row in net.gen.iterrows():
            b = int(row.bus)
            Pgen[b] += float(net.res_gen.loc[idx, "p_mw"])
            Qgen[b] += float(net.res_gen.loc[idx, "q_mvar"])

    # sgen (PV / wind included here)
    if len(net.sgen) > 0 and hasattr(net, "res_sgen"):
        for idx, row in net.sgen.iterrows():
            b = int(row.bus)
            Pgen[b] += float(net.res_sgen.loc[idx, "p_mw"])
            Qgen[b] += float(net.res_sgen.loc[idx, "q_mvar"])

    return Pgen, Qgen


def make_pf_case_from_opf(net_opf):
    """
    Create a PF case whose generator dispatch/setpoints are initialized from ACOPF results.
    """
    net_pf = copy.deepcopy(net_opf)

    # Fix generator active powers to OPF results
    if len(net_pf.gen) > 0:
        net_pf.gen["p_mw"] = net_opf.res_gen["p_mw"].values

        # Fix voltage setpoints at generator buses to OPF solved voltages
        gen_buses = net_pf.gen["bus"].values.astype(int)
        net_pf.gen["vm_pu"] = net_opf.res_bus.loc[gen_buses, "vm_pu"].values

    # Fix ext_grid voltage magnitude to OPF solved voltage
    if len(net_pf.ext_grid) > 0:
        ext_buses = net_pf.ext_grid["bus"].values.astype(int)
        net_pf.ext_grid["vm_pu"] = net_opf.res_bus.loc[ext_buses, "vm_pu"].values

    return net_pf


def make_main_row(net_pf, sample_id):
    """
    Main dataset row format:
    [sample_id, Pload1, Qload1, V1, d1, ..., Pload118, Qload118, V118, d118]
    """
    Pload, Qload = aggregate_load_by_bus(net_pf)

    row = [sample_id]
    for b in net_pf.bus.index:
        V = float(net_pf.res_bus.loc[b, "vm_pu"])
        d = float(net_pf.res_bus.loc[b, "va_degree"])
        row.extend([float(Pload[b]), float(Qload[b]), V, d])
    return row


def make_dispatch_row(net_opf, sample_id):
    """
    Dispatch dataset row format:
    [sample_id, Pg1, Qg1, Pg2, Qg2, ..., Pg118, Qg118]
    """
    Pgen, Qgen = aggregate_generation_by_bus_from_opf(net_opf)

    row = [sample_id]
    for b in net_opf.bus.index:
        row.extend([float(Pgen[b]), float(Qgen[b])])
    return row


def save_main_dataset(rows, contingency_rows, prefix):
    cols = ["sample_id"]
    n_bus = (len(rows[0]) - 1) // 4
    for b in range(1, n_bus + 1):
        cols.extend([f"Pload_{b}", f"Qload_{b}", f"V_{b}", f"delta_{b}"])

    df = pd.DataFrame(rows, columns=cols)
    xlsx_path = os.path.join(out_dir, f"{prefix}.xlsx")
    df.to_excel(xlsx_path, index=False)

    cont_df = pd.DataFrame(contingency_rows, columns=["outage_from", "outage_to", "outage_line_idx"])
    cont_path = os.path.join(out_dir, f"{prefix}_contingency_info.csv")
    cont_df.to_csv(cont_path, index=False)

    print(f"Saved {prefix}:")
    print(f"  {xlsx_path}")
    print(f"  {cont_path}")


def save_dispatch_dataset(rows, prefix):
    cols = ["sample_id"]
    n_bus = (len(rows[0]) - 1) // 2
    for b in range(1, n_bus + 1):
        cols.extend([f"Pgen_{b}", f"Qgen_{b}"])

    df = pd.DataFrame(rows, columns=cols)
    xlsx_path = os.path.join(out_dir, f"{prefix}_dispatch.xlsx")
    df.to_excel(xlsx_path, index=False)
    print(f"  {xlsx_path}")


def generate_split(net_base, outage_lines, n_base_cases, prefix, start_sample_id=0):
    """
    For each successful random ACOPF base case, run ACPF for ALL outage lines in outage_lines.
    Therefore:
      final sample count = n_base_cases * len(outage_lines)
    """
    main_rows = []
    dispatch_rows = []
    contingency_rows = []
    sample_id = start_sample_id

    success_base_cases = 0
    max_attempts = n_base_cases * max_attempt_factor
    attempts = 0

    while success_base_cases < n_base_cases and attempts < max_attempts:
        attempts += 1

        # 1) random operating point
        net_opf = copy.deepcopy(net_base)
        apply_random_operating_point(net_opf)

        # 2) configure and run ACOPF
        configure_acopf(net_opf)

        try:
            pp.runopp(
                net_opf,
                calculate_voltage_angles=True,
                init="pf",
                delta=1e-8,
                verbose=False
            )
        except Exception:
            continue

        if not getattr(net_opf, "OPF_converged", False):
            continue

        # 3) pre-contingency dispatch row (same for all outages of this base case)
        #    We will repeat it once per contingency sample so sample_id aligns 1-to-1.
        dispatch_base_row = make_dispatch_row(net_opf, sample_id=None)

        # 4) for each outage, run ACPF with fixed OPF dispatch
        all_outages_success = True
        temp_main_rows = []
        temp_dispatch_rows = []
        temp_cont_rows = []

        for line_idx in outage_lines:
            if line_idx not in net_opf.line.index:
                print(f"Warning: line index {line_idx} not found in net.line")
                all_outages_success = False
                break

            net_pf = make_pf_case_from_opf(net_opf)

            outage_from = int(net_pf.line.loc[line_idx, "from_bus"]) + 1
            outage_to   = int(net_pf.line.loc[line_idx, "to_bus"]) + 1

            net_pf.line.at[line_idx, "in_service"] = False

            try:
                pp.runpp(
                    net_pf,
                    algorithm="nr",
                    calculate_voltage_angles=True,
                    init="dc",
                    enforce_q_lims=True
                )
            except Exception:
                all_outages_success = False
                break

            if not net_pf.converged:
                all_outages_success = False
                break

            main_row = make_main_row(net_pf, sample_id)

            # repeat dispatch row with this sample_id
            dispatch_row = [sample_id] + dispatch_base_row[1:]

            temp_main_rows.append(main_row)
            temp_dispatch_rows.append(dispatch_row)
            temp_cont_rows.append([outage_from, outage_to, line_idx])

            sample_id += 1

        if not all_outages_success:
            continue

        # 5) commit this ACOPF base case
        main_rows.extend(temp_main_rows)
        dispatch_rows.extend(temp_dispatch_rows)
        contingency_rows.extend(temp_cont_rows)

        success_base_cases += 1
        if success_base_cases % 20 == 0:
            print(f"{prefix}: successful ACOPF base cases = {success_base_cases} / {n_base_cases}")

    print(f"{prefix}: final successful base cases = {success_base_cases}")
    print(f"{prefix}: final contingency samples   = {len(main_rows)}")

    save_main_dataset(main_rows, contingency_rows, prefix)
    save_dispatch_dataset(dispatch_rows, prefix)

    return sample_id


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    print("Loading IEEE 118-bus network...")
    net_base = nw.case118()

    # Add PV and wind generators
    net_base = add_renewables(net_base, pv_bus, wt_bus)

    # Save branch data file
    export_branch_data_csv(net_base, os.path.join(out_dir, "case118_branch_data.csv"))

    next_id = 0
    next_id = generate_split(net_base, train_outage_lines, n_base_train, "train", start_sample_id=next_id)
    next_id = generate_split(net_base, val_outage_lines,   n_base_val,   "val",   start_sample_id=next_id)
    next_id = generate_split(net_base, test_outage_lines,  n_base_test,  "test",  start_sample_id=next_id)

    print("\nAll ACOPF->ACPF dataset files generated.")