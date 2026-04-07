import os
import copy
import numpy as np
import pandas as pd
import pandapower as pp
import pandapower.networks as nw

SEED = 42
rng = np.random.default_rng(SEED)

out_dir = r"C:\Users\mahao\gnn_case118_dataset"
os.makedirs(out_dir, exist_ok=True)

n_base_train = 200
n_base_val   = 60
n_base_test  = 60

train_outage_lines = [36, 33, 8, 93, 94]
val_outage_lines   = [36, 33, 8, 93, 94]
test_outage_lines  = [38, 33, 142, 31, 93]

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

max_attempt_factor = 30
VMIN = 0.94
VMAX = 1.06
MAX_LOADING = 100.0


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

    pd.DataFrame(rows).to_csv(path_csv, index=False)


def add_renewables(net, pv_bus, wt_bus):
    pp.create_sgen(net, bus=pv_bus, p_mw=0.0, q_mvar=0.0, name="PV")
    pp.create_sgen(net, bus=wt_bus, p_mw=0.0, q_mvar=0.0, name="Wind")
    return net


def configure_acopf(net):
    net.bus["min_vm_pu"] = VMIN
    net.bus["max_vm_pu"] = VMAX
    net.line["max_loading_percent"] = MAX_LOADING

    if len(net.trafo) > 0:
        net.trafo["max_loading_percent"] = MAX_LOADING

    if len(net.ext_grid) > 0:
        net.ext_grid["min_p_mw"] = -1e4
        net.ext_grid["max_p_mw"] =  1e4
        net.ext_grid["min_q_mvar"] = -1e4
        net.ext_grid["max_q_mvar"] =  1e4
        net.ext_grid["controllable"] = True

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

    if len(net.sgen) > 0:
        net.sgen["controllable"] = False

    if len(net.poly_cost) > 0:
        net.poly_cost.drop(net.poly_cost.index, inplace=True)

    if len(net.ext_grid) > 0:
        for idx in net.ext_grid.index:
            pp.create_poly_cost(net, idx, "ext_grid", cp1_eur_per_mw=30.0)

    if len(net.gen) > 0:
        for idx in net.gen.index:
            pp.create_poly_cost(net, idx, "gen", cp1_eur_per_mw=20.0)


def apply_random_operating_point(net):
    load_scale = rng.uniform(load_scale_min, load_scale_max)
    pv_scale   = rng.uniform(pv_scale_min, pv_scale_max)
    wt_scale   = rng.uniform(wt_scale_min, wt_scale_max)

    net.load["p_mw"] *= load_scale
    net.load["q_mvar"] *= load_scale

    net.sgen.loc[net.sgen["name"] == "PV", "p_mw"] = pv_scale * pv_cap_mw
    net.sgen.loc[net.sgen["name"] == "PV", "q_mvar"] = 0.0

    net.sgen.loc[net.sgen["name"] == "Wind", "p_mw"] = wt_scale * wt_cap_mw
    net.sgen.loc[net.sgen["name"] == "Wind", "q_mvar"] = 0.0


def aggregate_load_by_bus(net):
    n_bus = len(net.bus)
    Pload = np.zeros(n_bus)
    Qload = np.zeros(n_bus)
    for _, row in net.load.iterrows():
        b = int(row.bus)
        Pload[b] += float(row.p_mw)
        Qload[b] += float(row.q_mvar)
    return Pload, Qload


def aggregate_generation_by_bus_from_opf(net):
    n_bus = len(net.bus)
    Pgen = np.zeros(n_bus)
    Qgen = np.zeros(n_bus)

    if len(net.ext_grid) > 0:
        for idx, row in net.ext_grid.iterrows():
            b = int(row.bus)
            Pgen[b] += float(net.res_ext_grid.loc[idx, "p_mw"])
            Qgen[b] += float(net.res_ext_grid.loc[idx, "q_mvar"])

    if len(net.gen) > 0:
        for idx, row in net.gen.iterrows():
            b = int(row.bus)
            Pgen[b] += float(net.res_gen.loc[idx, "p_mw"])
            Qgen[b] += float(net.res_gen.loc[idx, "q_mvar"])

    if len(net.sgen) > 0:
        for idx, row in net.sgen.iterrows():
            b = int(row.bus)
            Pgen[b] += float(net.res_sgen.loc[idx, "p_mw"])
            Qgen[b] += float(net.res_sgen.loc[idx, "q_mvar"])

    return Pgen, Qgen


def make_pf_case_from_opf(net_opf):
    net_pf = copy.deepcopy(net_opf)

    if len(net_pf.gen) > 0:
        net_pf.gen["p_mw"] = net_opf.res_gen["p_mw"].values
        gen_buses = net_pf.gen["bus"].values.astype(int)
        net_pf.gen["vm_pu"] = net_opf.res_bus.loc[gen_buses, "vm_pu"].values

    if len(net_pf.ext_grid) > 0:
        ext_buses = net_pf.ext_grid["bus"].values.astype(int)
        net_pf.ext_grid["vm_pu"] = net_opf.res_bus.loc[ext_buses, "vm_pu"].values

    return net_pf


def make_main_row(net_pf, sample_id):
    Pload, Qload = aggregate_load_by_bus(net_pf)
    row = [sample_id]
    for b in net_pf.bus.index:
        V = float(net_pf.res_bus.loc[b, "vm_pu"])
        d = float(net_pf.res_bus.loc[b, "va_degree"])
        row.extend([float(Pload[b]), float(Qload[b]), V, d])
    return row


def make_dispatch_row(net_opf, sample_id):
    Pgen, Qgen = aggregate_generation_by_bus_from_opf(net_opf)
    row = [sample_id]
    for b in net_opf.bus.index:
        row.extend([float(Pgen[b]), float(Qgen[b])])
    return row


def make_dynamic_edge_row(net_opf, sample_id):
    """
    Pre-contingency line loading from ACOPF for every original line.
    """
    row = [sample_id]
    for idx in net_opf.line.index:
        row.append(float(net_opf.res_line.loc[idx, "loading_percent"]))
    return row


def make_classification_label(net_pf, sample_id):
    """
    0: no violation
    1: branch only
    2: voltage only
    3: both
    """
    has_v = bool(((net_pf.res_bus["vm_pu"] < VMIN) | (net_pf.res_bus["vm_pu"] > VMAX)).any())
    has_f = bool((net_pf.res_line["loading_percent"] > MAX_LOADING).any())

    if not has_f and not has_v:
        cls = 0
    elif has_f and not has_v:
        cls = 1
    elif not has_f and has_v:
        cls = 2
    else:
        cls = 3

    return [sample_id, int(has_f), int(has_v), cls]


def save_main_dataset(rows, contingency_rows, prefix):
    cols = ["sample_id"]
    n_bus = (len(rows[0]) - 1) // 4
    for b in range(1, n_bus + 1):
        cols.extend([f"Pload_{b}", f"Qload_{b}", f"V_{b}", f"delta_{b}"])

    pd.DataFrame(rows, columns=cols).to_excel(os.path.join(out_dir, f"{prefix}.xlsx"), index=False)
    pd.DataFrame(contingency_rows, columns=["outage_from", "outage_to", "outage_line_idx"]).to_csv(
        os.path.join(out_dir, f"{prefix}_contingency_info.csv"), index=False
    )


def save_dispatch_dataset(rows, prefix):
    cols = ["sample_id"]
    n_bus = (len(rows[0]) - 1) // 2
    for b in range(1, n_bus + 1):
        cols.extend([f"Pgen_{b}", f"Qgen_{b}"])
    pd.DataFrame(rows, columns=cols).to_excel(os.path.join(out_dir, f"{prefix}_dispatch.xlsx"), index=False)


def save_edge_dynamic_dataset(rows, n_line, prefix):
    cols = ["sample_id"] + [f"loading_{l+1}" for l in range(n_line)]
    pd.DataFrame(rows, columns=cols).to_excel(os.path.join(out_dir, f"{prefix}_edge_dynamic.xlsx"), index=False)


def save_label_dataset(rows, prefix):
    cols = ["sample_id", "has_branch_violation", "has_voltage_violation", "class_id"]
    pd.DataFrame(rows, columns=cols).to_csv(os.path.join(out_dir, f"{prefix}_labels.csv"), index=False)


def generate_split(net_base, outage_lines, n_base_cases, prefix, start_sample_id=0):
    main_rows = []
    dispatch_rows = []
    edge_rows = []
    label_rows = []
    contingency_rows = []
    sample_id = start_sample_id

    success_base_cases = 0
    max_attempts = n_base_cases * max_attempt_factor
    attempts = 0
    n_line = len(net_base.line)

    while success_base_cases < n_base_cases and attempts < max_attempts:
        attempts += 1

        net_opf = copy.deepcopy(net_base)
        apply_random_operating_point(net_opf)
        configure_acopf(net_opf)

        try:
            pp.runopp(net_opf, calculate_voltage_angles=True, init="pf", verbose=False)
        except Exception:
            continue

        if not getattr(net_opf, "OPF_converged", False):
            continue

        dispatch_base = make_dispatch_row(net_opf, sample_id=None)
        edge_base = make_dynamic_edge_row(net_opf, sample_id=None)

        temp_main = []
        temp_disp = []
        temp_edge = []
        temp_lab  = []
        temp_cont = []

        all_ok = True

        for line_idx in outage_lines:
            if line_idx not in net_opf.line.index:
                all_ok = False
                break

            net_pf = make_pf_case_from_opf(net_opf)
            outage_from = int(net_pf.line.loc[line_idx, "from_bus"]) + 1
            outage_to   = int(net_pf.line.loc[line_idx, "to_bus"]) + 1

            net_pf.line.at[line_idx, "in_service"] = False

            try:
                pp.runpp(net_pf, algorithm="nr", calculate_voltage_angles=True, init="dc", enforce_q_lims=True)
            except Exception:
                all_ok = False
                break

            if not net_pf.converged:
                all_ok = False
                break

            temp_main.append(make_main_row(net_pf, sample_id))
            temp_disp.append([sample_id] + dispatch_base[1:])
            temp_edge.append([sample_id] + edge_base[1:])
            temp_lab.append(make_classification_label(net_pf, sample_id))
            temp_cont.append([outage_from, outage_to, line_idx])

            sample_id += 1

        if not all_ok:
            continue

        main_rows.extend(temp_main)
        dispatch_rows.extend(temp_disp)
        edge_rows.extend(temp_edge)
        label_rows.extend(temp_lab)
        contingency_rows.extend(temp_cont)

        success_base_cases += 1
        if success_base_cases % 20 == 0:
            print(f"{prefix}: successful ACOPF base cases = {success_base_cases}/{n_base_cases}")

    save_main_dataset(main_rows, contingency_rows, prefix)
    save_dispatch_dataset(dispatch_rows, prefix)
    save_edge_dynamic_dataset(edge_rows, n_line, prefix)
    save_label_dataset(label_rows, prefix)

    return sample_id


if __name__ == "__main__":
    net_base = nw.case118()
    net_base = add_renewables(net_base, 9, 56)

    export_branch_data_csv(net_base, os.path.join(out_dir, "case118_branch_data.csv"))

    next_id = 0
    next_id = generate_split(net_base, train_outage_lines, n_base_train, "train", start_sample_id=next_id)
    next_id = generate_split(net_base, val_outage_lines,   n_base_val,   "val",   start_sample_id=next_id)
    next_id = generate_split(net_base, test_outage_lines,  n_base_test,  "test",  start_sample_id=next_id)

    print("All classification datasets generated.")