import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ============================================================
# Reproducibility
# ============================================================
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ============================================================
# Config
# ============================================================
n_bus = 118
batch_size = 16
lr = 5e-4
weight_decay = 1e-6
max_epochs = 800
patience = 30
val_ratio = 0.2

# ============================================================
# Early stopping config
# ============================================================
# Improvement is judged by both absolute and relative thresholds.
min_delta_abs = 1e-6
min_delta_rel = 1e-4

baseMVA = 100.0

# Supervised loss weights
w_v_pv = 100.0
w_v_pq = 100.0
w_d = 10.0
w_q = 10.0

# Physics-informed nodal-balance loss weights.
# Mismatch is calculated in per-unit.
lambda_p_mis = 10.0
lambda_q_mis = 10.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# Data file paths
# ============================================================
train_book = r"C:\Users\mahao\OneDrive - Washington State University (email.wsu.edu)\Research\BaseCase Study\case118_pf_dataset_nocont_125noseasonalcurves.xlsx"
test_book  = r"C:\Users\mahao\OneDrive - Washington State University (email.wsu.edu)\Research\BaseCase Study\case118_pf_dataset_nocont_125scen.xlsx"

model_path = f"[{n_bus}bus]_Best_NNConv_VDeltaQgen_Physics_PVPQV.pt"
stats_path = f"[{n_bus}bus]_NormStats_NNConv_VDeltaQgen_Physics_PVPQV.pt"
summary_csv = f"[{n_bus}bus]_TestSummary_NNConv_VDeltaQgen_Physics_PVPQV.csv"
loss_fig_path = f"[{n_bus}bus]_TrainValLoss_NNConv_VDeltaQgen_Physics_PVPQV.png"
loss_sep_fig_path = f"[{n_bus}bus]_TrainValLoss_Separate_NNConv_VDeltaQgen_Physics_PVPQV.png"
out_excel = f"[{n_bus}bus]_GNN_Prediction_Dataset_VDeltaQgen_Physics.xlsx"


# ============================================================
# Utility functions
# ============================================================
def read_excel_sheet(book_path, sheet_name):
    return pd.read_excel(book_path, sheet_name=sheet_name)


def load_branch_data_from_excel(book_path):
    df = read_excel_sheet(book_path, "case118_branch_data")

    from_bus = (df["from_bus"].values.astype(int) - 1).tolist()
    to_bus   = (df["to_bus"].values.astype(int) - 1).tolist()

    edge_attr_np = df[["r", "x", "b", "rateA", "ratio"]].values.astype(np.float32)

    # normalize each edge feature column
    edge_mean = edge_attr_np.mean(axis=0, keepdims=True)
    edge_std = edge_attr_np.std(axis=0, keepdims=True)
    edge_std[edge_std == 0] = 1.0

    edge_attr_np = (edge_attr_np - edge_mean) / edge_std
    edge_attr = edge_attr_np.tolist()

    return df, from_bus, to_bus, edge_attr


def load_bus_type_from_excel(book_path):
    df = read_excel_sheet(book_path, "case118_bus_type")
    df = df.sort_values("bus").reset_index(drop=True)

    is_slack = df["is_slack"].values.astype(np.float32)
    is_pv    = df["is_pv"].values.astype(np.float32)
    is_pq    = df["is_pq"].values.astype(np.float32)

    return is_slack, is_pv, is_pq


def read_branch_power_flow_from_workbook(book_path):
    xls = pd.ExcelFile(book_path)
    print("Available sheets:", [repr(s) for s in xls.sheet_names])

    target = None
    for s in xls.sheet_names:
        if s.strip().lower() == "branch_power_flow":
            target = s
            break

    if target is None:
        raise ValueError(
            f"'branch_power_flow' not found in workbook. Available sheets: {xls.sheet_names}"
        )

    df = pd.read_excel(book_path, sheet_name=target)
    df = df.sort_values("sample_id").reset_index(drop=True)
    return df


def extract_true_branch_flow_matrix(branchflow_df):
    s_cols = [c for c in branchflow_df.columns if c.startswith("Sbranch_")]
    s_cols = sorted(s_cols, key=lambda x: int(x.split("_")[1]))
    true_branch_mat = branchflow_df[s_cols].values.astype(np.float64)
    return true_branch_mat, s_cols


def read_dataset_from_workbook(book_path):
    main_df = read_excel_sheet(book_path, "train")
    disp_df = read_excel_sheet(book_path, "train_dispatch")
    meta_df = read_excel_sheet(book_path, "sample_meta")

    main_df = main_df.sort_values("sample_id").reset_index(drop=True)
    disp_df = disp_df.sort_values("sample_id").reset_index(drop=True)
    meta_df = meta_df.sort_values("sample_id").reset_index(drop=True)

    if not np.array_equal(main_df["sample_id"].values, disp_df["sample_id"].values):
        raise ValueError("sample_id mismatch between train and train_dispatch sheets")

    if not np.array_equal(main_df["sample_id"].values, meta_df["sample_id"].values):
        raise ValueError("sample_id mismatch between train and sample_meta sheets")

    # base-case workbook may not have contingency_index
    if "contingency_index" not in main_df.columns:
        main_df.insert(1, "contingency_index", 0)

    if "contingency_index" not in disp_df.columns:
        disp_df.insert(1, "contingency_index", 0)

    if "contingency_index" not in meta_df.columns:
        insert_pos = 3 if "hour" in meta_df.columns else len(meta_df.columns)
        meta_df.insert(insert_pos, "contingency_index", 0)

    return main_df, disp_df, meta_df


def split_features_targets_with_genbus(main_df, dispatch_df, n_bus,
                                       is_slack_vec, is_pv_vec, is_pq_vec, gen_bus_vec):
    """
    Input x per bus:
      [Pload, Qload, Pgen_total_at_bus, is_pv, is_pq, is_slack]

    Target y per bus:
      [V, delta, Qgen_total_at_bus]

    Qgen is removed from input and added to output to avoid data leakage.
    """
    main_np = main_df.values
    disp_np = dispatch_df.values

    x_list, y_list = [], []
    ng = len(gen_bus_vec)

    # after insertion, columns are:
    # main: sample_id, contingency_index, Pload_1, Qload_1, V_1, delta_1, ...
    # disp: sample_id, contingency_index, Pgen_1, Qgen_1, ...
    main_offset = 2
    disp_offset = 2

    for i in range(len(main_np)):
        x_sample, y_sample = [], []

        Pgen_bus = np.zeros(n_bus, dtype=np.float32)
        Qgen_bus = np.zeros(n_bus, dtype=np.float32)

        for g in range(ng):
            bus_idx = int(gen_bus_vec[g]) - 1
            Pgen = disp_np[i, 2 * g + disp_offset]
            Qgen = disp_np[i, 2 * g + disp_offset + 1]
            Pgen_bus[bus_idx] += Pgen
            Qgen_bus[bus_idx] += Qgen

        for n in range(n_bus):
            Pload = main_np[i, 4 * n + main_offset]
            Qload = main_np[i, 4 * n + main_offset + 1]
            V     = main_np[i, 4 * n + main_offset + 2]
            d     = main_np[i, 4 * n + main_offset + 3]

            x_sample.append([
                Pload,
                Qload,
                Pgen_bus[n],
                is_pv_vec[n],
                is_pq_vec[n],
                is_slack_vec[n]
            ])

            y_sample.append([
                V,
                d,
                Qgen_bus[n]
            ])

        x_list.append(x_sample)
        y_list.append(y_sample)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    return x, y


def fit_normalization(x_train, y_train):
    x_mean = x_train.mean(dim=0)
    x_std  = x_train.std(dim=0)
    y_mean = y_train.mean(dim=0)
    y_std  = y_train.std(dim=0)

    x_std[x_std == 0] = 1.0
    y_std[y_std == 0] = 1.0

    return x_mean, x_std, y_mean, y_std


def apply_normalization(x, y, x_mean, x_std, y_mean, y_std):
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std

    # preserve bus-type flags
    # input columns are now:
    # 0 Pload, 1 Qload, 2 Pgen, 3 is_pv, 4 is_pq, 5 is_slack
    x_norm[:, :, 3:] = x[:, :, 3:]
    return x_norm, y_norm


def denormalize(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def build_dynamic_dataset(x_tensor, y_tensor, meta_df, branch_df,
                          from_bus_orig, to_bus_orig, edge_attr_orig, n_bus):
    """
    If contingency_index > 0:
        remove the outaged branch.
    If contingency_index == 0:
        keep full intact topology.

    Also attaches Ybus to each graph for physics-informed nodal-balance loss.
    Topology-dependent objects are cached by contingency_index.
    """
    data_list = []
    cont_idx_list = meta_df["contingency_index"].values.astype(int)

    topology_cache = {}

    for i in range(x_tensor.size(0)):
        x = x_tensor[i]
        y = y_tensor[i]

        cont_idx = int(cont_idx_list[i])

        if cont_idx not in topology_cache:
            filtered_edges = []
            filtered_attr = []

            if cont_idx <= 0:
                filtered_edges = list(zip(from_bus_orig, to_bus_orig))
                filtered_attr = list(edge_attr_orig)
            else:
                out_idx0 = cont_idx - 1

                f_out = int(branch_df.iloc[out_idx0]["from_bus"]) - 1
                t_out = int(branch_df.iloc[out_idx0]["to_bus"]) - 1

                for f, t, attr in zip(from_bus_orig, to_bus_orig, edge_attr_orig):
                    if not ((f == f_out and t == t_out) or (f == t_out and t == f_out)):
                        filtered_edges.append((f, t))
                        filtered_attr.append(attr)

            from_buses_temp = [f for f, t in filtered_edges]
            to_buses_temp   = [t for f, t in filtered_edges]

            edge_index = torch.tensor(
                [
                    from_buses_temp + to_buses_temp,
                    to_buses_temp + from_buses_temp
                ],
                dtype=torch.long
            )

            edge_attr = torch.tensor(filtered_attr + filtered_attr, dtype=torch.float32)

            active_branch_df, _ = get_active_branch_df(branch_df, cont_idx)
            Ybus = build_Ybus_from_active_branch_df(active_branch_df, n_bus)

            ybus_real = torch.tensor(np.real(Ybus), dtype=torch.float32)
            ybus_imag = torch.tensor(np.imag(Ybus), dtype=torch.float32)

            topology_cache[cont_idx] = {
                "edge_index": edge_index,
                "edge_attr": edge_attr,
                "ybus_real": ybus_real,
                "ybus_imag": ybus_imag
            }

        topo = topology_cache[cont_idx]

        data = Data(
            x=x,
            y=y,
            edge_index=topo["edge_index"],
            edge_attr=topo["edge_attr"],
            ybus_real=topo["ybus_real"],
            ybus_imag=topo["ybus_imag"]
        )

        data_list.append(data)

    return data_list


# ============================================================
# Dynamic branch-flow helpers
# ============================================================
def get_active_branch_df(branch_df, contingency_index):
    n_branch = len(branch_df)
    active_mask = np.ones(n_branch, dtype=bool)

    # base case: no outage
    if contingency_index <= 0:
        active_branch_df = branch_df.copy().reset_index(drop=True)
        return active_branch_df, active_mask

    out_idx0 = contingency_index - 1
    if 0 <= out_idx0 < n_branch:
        active_mask[out_idx0] = False

    active_branch_df = branch_df.loc[active_mask].reset_index(drop=True)
    return active_branch_df, active_mask



def build_Ybus_from_active_branch_df(active_branch_df, n_bus):
    """
    Build MATPOWER-style Ybus from the active branch table.

    Uses:
      r, x, b, ratio

    Assumptions:
      - branch r/x/b are in per unit
      - ratio = 0 means tap = 1
      - no phase-shifter angle because exported branch table does not include SHIFT
    """
    Ybus = np.zeros((n_bus, n_bus), dtype=np.complex128)

    for k in range(len(active_branch_df)):
        fb = int(active_branch_df.iloc[k]["from_bus"]) - 1
        tb = int(active_branch_df.iloc[k]["to_bus"]) - 1

        r = float(active_branch_df.iloc[k]["r"])
        x = float(active_branch_df.iloc[k]["x"])
        b = float(active_branch_df.iloc[k]["b"])
        ratio = float(active_branch_df.iloc[k]["ratio"])

        tap = ratio if abs(ratio) > 1e-12 else 1.0

        z = complex(r, x)
        if abs(z) < 1e-12:
            y = 0.0 + 0.0j
        else:
            y = 1.0 / z

        ysh = 1j * b / 2.0

        yff = (y + ysh) / (tap * np.conj(tap))
        yft = -y / np.conj(tap)
        ytf = -y / tap
        ytt = y + ysh

        Ybus[fb, fb] += yff
        Ybus[fb, tb] += yft
        Ybus[tb, fb] += ytf
        Ybus[tb, tb] += ytt

    return Ybus


def build_Yf_Yt_from_active_branch_df(active_branch_df, n_bus):
    n_branch = len(active_branch_df)

    Yf = np.zeros((n_branch, n_bus), dtype=np.complex128)
    Yt = np.zeros((n_branch, n_bus), dtype=np.complex128)

    for k in range(n_branch):
        fb = int(active_branch_df.iloc[k]["from_bus"]) - 1
        tb = int(active_branch_df.iloc[k]["to_bus"]) - 1

        r = float(active_branch_df.iloc[k]["r"])
        x = float(active_branch_df.iloc[k]["x"])
        b = float(active_branch_df.iloc[k]["b"])
        ratio = float(active_branch_df.iloc[k]["ratio"])

        tap = ratio if abs(ratio) > 1e-12 else 1.0

        z = complex(r, x)
        if abs(z) < 1e-12:
            y = 0.0 + 0.0j
        else:
            y = 1.0 / z

        ysh = 1j * b / 2.0

        yff = (y + ysh) / (tap * np.conj(tap))
        yft = -y / np.conj(tap)
        ytf = -y / tap
        ytt = y + ysh

        Yf[k, fb] += yff
        Yf[k, tb] += yft

        Yt[k, fb] += ytf
        Yt[k, tb] += ytt

    return Yf, Yt


def compute_branch_flows_from_active_topology(vm, va_deg, active_branch_df, n_bus, baseMVA=100.0):
    va_rad = np.deg2rad(va_deg)
    V = vm * np.exp(1j * va_rad)

    fb = active_branch_df["from_bus"].values.astype(int) - 1
    tb = active_branch_df["to_bus"].values.astype(int) - 1

    Yf, Yt = build_Yf_Yt_from_active_branch_df(active_branch_df, n_bus)

    If = Yf @ V
    It = Yt @ V

    Vf = V[fb]
    Vt = V[tb]

    Sf = Vf * np.conj(If) * baseMVA
    St = Vt * np.conj(It) * baseMVA

    Sbr_active = np.maximum(np.abs(Sf), np.abs(St))
    return Sbr_active


def build_full_branch_flow_vector(Sbr_active, active_mask, n_branch):
    Sbr_full = np.full(n_branch, np.nan, dtype=np.float64)
    Sbr_full[active_mask] = Sbr_active
    return Sbr_full


# ============================================================
# Model
# ============================================================
class EdgeGCN(nn.Module):
    def __init__(self, in_channels=6, hidden_channels=64, out_channels=3, edge_dim=5, dropout=0.1):
        super().__init__()

        # Edge network for conv1:
        # maps edge_attr -> weight matrix of shape [in_channels, hidden_channels]
        self.edge_mlp1 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, in_channels * hidden_channels)
        )

        # Edge network for conv2:
        # maps edge_attr -> weight matrix of shape [hidden_channels, hidden_channels]
        self.edge_mlp2 = nn.Sequential(
            nn.Linear(edge_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hidden_channels * hidden_channels)
        )

        self.conv1 = NNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            nn=self.edge_mlp1,
            aggr='mean'
        )

        self.conv2 = NNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            nn=self.edge_mlp2,
            aggr='mean'
        )

        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        out = self.lin(x)
        return out


# ============================================================
# Loss: supervised + physics-informed nodal balance
# ============================================================
def masked_mse(err, mask):
    """
    Mean squared error over selected nodes only.
    If the mask is empty, return zero safely.
    """
    mask = mask.bool()
    if torch.sum(mask) == 0:
        return torch.tensor(0.0, dtype=err.dtype, device=err.device)
    return torch.mean(err[mask] ** 2)


def loss_mse_denorm_physics(pred_norm, target_norm, batch,
                            y_mean, y_std,
                            x_mean, x_std,
                            n_bus,
                            baseMVA=100.0,
                            w_v_pv=100.0,
                            w_v_pq=100.0,
                            w_d=10.0,
                            w_q=10.0,
                            lambda_p_mis=10.0,
                            lambda_q_mis=10.0):
    """
    Total loss:

        L_total = L_supervised + L_physics

    where:

        L_supervised =
            w_v_pv * MSE(V_error on PV buses)
          + w_v_pq * MSE(V_error on PQ buses)
          + w_d    * MSE(delta_error on all buses)
          + w_q    * MSE(Qgen_error on all buses)

        L_physics =
            lambda_p_mis * MSE(P_mismatch_pu)
          + lambda_q_mis * MSE(Q_mismatch_pu)

    y columns:
      0 V
      1 delta in degrees
      2 Qgen in MVAr

    x columns:
      0 Pload in MW
      1 Qload in MVAr
      2 Pgen in MW
      3 is_pv
      4 is_pq
      5 is_slack
    """
    total_nodes = pred_norm.size(0)
    batch_size_local = total_nodes // n_bus

    # --------------------------------------------------------
    # Denormalize predicted and target outputs
    # --------------------------------------------------------
    y_mean_exp = y_mean.repeat(batch_size_local, 1).to(pred_norm.device)
    y_std_exp  = y_std.repeat(batch_size_local, 1).to(pred_norm.device)

    pred   = denormalize(pred_norm, y_mean_exp, y_std_exp)
    target = denormalize(target_norm, y_mean_exp, y_std_exp)

    err = pred - target

    err_v = err[:, 0]
    err_d = err[:, 1]
    err_q = err[:, 2]

    # --------------------------------------------------------
    # Bus-type masks
    # Bus-type flags are preserved during normalization:
    #   batch.x[:, 3] = is_pv
    #   batch.x[:, 4] = is_pq
    #   batch.x[:, 5] = is_slack
    # --------------------------------------------------------
    pv_mask = batch.x[:, 3] > 0.5
    pq_mask = batch.x[:, 4] > 0.5

    # --------------------------------------------------------
    # Raw supervised component losses
    # --------------------------------------------------------
    loss_v_pv  = masked_mse(err_v, pv_mask)
    loss_v_pq  = masked_mse(err_v, pq_mask)
    loss_v_all = torch.mean(err_v ** 2)

    loss_d = torch.mean(err_d ** 2)
    loss_q = torch.mean(err_q ** 2)

    # --------------------------------------------------------
    # Weighted supervised contributions
    # --------------------------------------------------------
    loss_v_pv_w = w_v_pv * loss_v_pv
    loss_v_pq_w = w_v_pq * loss_v_pq

    # For reporting only; not directly used in total loss.
    loss_v_all_w = 100.0 * loss_v_all

    loss_d_w = w_d * loss_d
    loss_q_w = w_q * loss_q

    loss_supervised = (
        loss_v_pv_w +
        loss_v_pq_w +
        loss_d_w +
        loss_q_w
    )

    # --------------------------------------------------------
    # Denormalize input Pload, Qload, Pgen
    # --------------------------------------------------------
    x_mean_exp = x_mean.repeat(batch_size_local, 1).to(pred_norm.device)
    x_std_exp  = x_std.repeat(batch_size_local, 1).to(pred_norm.device)

    x_denorm_first3 = batch.x[:, 0:3] * x_std_exp[:, 0:3] + x_mean_exp[:, 0:3]

    x_3d = x_denorm_first3.reshape(batch_size_local, n_bus, 3)
    pred_3d = pred.reshape(batch_size_local, n_bus, 3)

    Pload = x_3d[:, :, 0]
    Qload = x_3d[:, :, 1]
    Pgen  = x_3d[:, :, 2]

    Vmag      = pred_3d[:, :, 0]
    delta_deg = pred_3d[:, :, 1]
    Qgen_pred = pred_3d[:, :, 2]

    # --------------------------------------------------------
    # Complex voltage
    # --------------------------------------------------------
    delta_rad = delta_deg * np.pi / 180.0

    Vreal = Vmag * torch.cos(delta_rad)
    Vimag = Vmag * torch.sin(delta_rad)
    Vcomplex = torch.complex(Vreal, Vimag)

    # --------------------------------------------------------
    # Batched Ybus
    # PyG batches ybus_real as [batch_size*n_bus, n_bus]
    # so reshape to [batch_size, n_bus, n_bus].
    # --------------------------------------------------------
    Yreal = batch.ybus_real.reshape(batch_size_local, n_bus, n_bus).to(pred_norm.device)
    Yimag = batch.ybus_imag.reshape(batch_size_local, n_bus, n_bus).to(pred_norm.device)
    Ybus = torch.complex(Yreal, Yimag)

    Ibus = torch.bmm(Ybus, Vcomplex.unsqueeze(-1)).squeeze(-1)

    # S_calc in MVA: S = V * conj(I) * baseMVA
    S_calc = Vcomplex * torch.conj(Ibus) * baseMVA

    Pcalc = torch.real(S_calc)
    Qcalc = torch.imag(S_calc)

    # --------------------------------------------------------
    # Nodal power-balance mismatch, in p.u.
    # --------------------------------------------------------
    P_mis_pu = (Pgen - Pload - Pcalc) / baseMVA
    Q_mis_pu = (Qgen_pred - Qload - Qcalc) / baseMVA

    # Raw physics losses
    loss_p_mis = torch.mean(P_mis_pu ** 2)
    loss_q_mis = torch.mean(Q_mis_pu ** 2)

    # Weighted physics contributions
    loss_p_mis_w = lambda_p_mis * loss_p_mis
    loss_q_mis_w = lambda_q_mis * loss_q_mis

    loss_physics = loss_p_mis_w + loss_q_mis_w

    loss_total = loss_supervised + loss_physics

    loss_dict = {
        # group losses
        "total": loss_total,
        "supervised": loss_supervised,
        "physics": loss_physics,

        # raw supervised losses
        "v_pv": loss_v_pv,
        "v_pq": loss_v_pq,
        "v_all": loss_v_all,
        "delta": loss_d,
        "qgen": loss_q,

        # raw physics losses
        "pmis": loss_p_mis,
        "qmis": loss_q_mis,

        # weighted supervised contributions
        "v_pv_w": loss_v_pv_w,
        "v_pq_w": loss_v_pq_w,
        "v_all_w": loss_v_all_w,
        "delta_w": loss_d_w,
        "qgen_w": loss_q_w,

        # weighted physics contributions
        "pmis_w": loss_p_mis_w,
        "qmis_w": loss_q_mis_w,
    }

    return loss_total, loss_dict


# ============================================================
# Loss meter and early-stopping helpers
# ============================================================
LOSS_KEYS = [
    # group losses
    "total",
    "supervised",
    "physics",

    # raw supervised component losses
    "v_pv",
    "v_pq",
    "v_all",
    "delta",
    "qgen",

    # raw physics component losses
    "pmis",
    "qmis",

    # weighted supervised contributions
    "v_pv_w",
    "v_pq_w",
    "v_all_w",
    "delta_w",
    "qgen_w",

    # weighted physics contributions
    "pmis_w",
    "qmis_w",
]


def init_loss_meter():
    return {k: 0.0 for k in LOSS_KEYS}


def update_loss_meter(meter, loss_dict, num_graphs):
    for k in LOSS_KEYS:
        meter[k] += float(loss_dict[k].detach().cpu()) * num_graphs


def finalize_loss_meter(meter, dataset_size):
    return {k: meter[k] / dataset_size for k in LOSS_KEYS}


def has_improved(new_value, best_value, min_delta_abs=1e-6, min_delta_rel=1e-4):
    """
    Improvement test using both absolute and relative thresholds.
    """
    if np.isinf(best_value):
        return True

    threshold = max(min_delta_abs, min_delta_rel * abs(best_value))
    return new_value < best_value - threshold


# ============================================================
# Load graph / bus metadata
# ============================================================
print("Loading branch data...")
branch_df, from_bus_orig, to_bus_orig, edge_attr_orig = load_branch_data_from_excel(train_book)

print("Loading bus type data...")
is_slack_vec, is_pv_vec, is_pq_vec = load_bus_type_from_excel(train_book)

gen_bus_vec = [
    1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40,
    42, 46, 49, 54, 55, 56, 59, 61, 62, 65, 66, 69, 70, 72, 73, 74, 76,
    77, 80, 85, 87, 89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110, 111,
    112, 113, 116
]

print("Loading training workbook...")
train_main_df, train_disp_df, train_meta_df = read_dataset_from_workbook(train_book)

print("Loading testing workbook...")
test_main_df, test_disp_df, test_meta_df = read_dataset_from_workbook(test_book)

# Build raw tensors
x_all_raw, y_all_raw = split_features_targets_with_genbus(
    train_main_df, train_disp_df, n_bus,
    is_slack_vec, is_pv_vec, is_pq_vec, gen_bus_vec
)

x_test_raw, y_test_raw = split_features_targets_with_genbus(
    test_main_df, test_disp_df, n_bus,
    is_slack_vec, is_pv_vec, is_pq_vec, gen_bus_vec
)

# Split train/val
idx_all = np.arange(x_all_raw.size(0))
idx_train, idx_val = train_test_split(
    idx_all, test_size=val_ratio, random_state=SEED, shuffle=True
)

x_train_raw = x_all_raw[idx_train]
y_train_raw = y_all_raw[idx_train]
meta_train_df = train_meta_df.iloc[idx_train].reset_index(drop=True)

x_val_raw = x_all_raw[idx_val]
y_val_raw = y_all_raw[idx_val]
meta_val_df = train_meta_df.iloc[idx_val].reset_index(drop=True)

# Fit normalization
x_mean, x_std, y_mean, y_std = fit_normalization(x_train_raw, y_train_raw)

x_train, y_train = apply_normalization(x_train_raw, y_train_raw, x_mean, x_std, y_mean, y_std)
x_val, y_val     = apply_normalization(x_val_raw, y_val_raw, x_mean, x_std, y_mean, y_std)
x_test, y_test   = apply_normalization(x_test_raw, y_test_raw, x_mean, x_std, y_mean, y_std)

# Build dynamic graph datasets
train_list = build_dynamic_dataset(x_train, y_train, meta_train_df, branch_df,
                                   from_bus_orig, to_bus_orig, edge_attr_orig, n_bus)
val_list = build_dynamic_dataset(x_val, y_val, meta_val_df, branch_df,
                                 from_bus_orig, to_bus_orig, edge_attr_orig, n_bus)
test_list = build_dynamic_dataset(x_test, y_test, test_meta_df, branch_df,
                                  from_bus_orig, to_bus_orig, edge_attr_orig, n_bus)

train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_list, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_list, batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_list)}")
print(f"Val samples:   {len(val_list)}")
print(f"Test samples:  {len(test_list)}")

torch.save({
    "x_mean": x_mean,
    "x_std": x_std,
    "y_mean": y_mean,
    "y_std": y_std
}, stats_path)


# ============================================================
# Train
# ============================================================
model = EdgeGCN(
    in_channels=6,
    hidden_channels=64,
    out_channels=3,
    edge_dim=5,
    dropout=0.1
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

y_mean_dev = y_mean.to(device)
y_std_dev  = y_std.to(device)
x_mean_dev = x_mean.to(device)
x_std_dev  = x_std.to(device)

# Best values for grouped early stopping
best_total = float("inf")
best_supervised = float("inf")
best_physics = float("inf")
best_v_pv = float("inf")
best_v_pq = float("inf")

best_epoch = -1
pat_count = 0

epoch_hist = []
hist = {f"train_{k}": [] for k in LOSS_KEYS}
hist.update({f"val_{k}": [] for k in LOSS_KEYS})

print("Training...")
for epoch in range(1, max_epochs + 1):

    # ========================================================
    # Training
    # ========================================================
    model.train()
    train_meter = init_loss_meter()

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y

        loss, loss_dict = loss_mse_denorm_physics(
            pred,
            target,
            batch,
            y_mean_dev,
            y_std_dev,
            x_mean_dev,
            x_std_dev,
            n_bus,
            baseMVA=baseMVA,
            w_v_pv=w_v_pv,
            w_v_pq=w_v_pq,
            w_d=w_d,
            w_q=w_q,
            lambda_p_mis=lambda_p_mis,
            lambda_q_mis=lambda_q_mis
        )

        loss.backward()
        optimizer.step()

        update_loss_meter(train_meter, loss_dict, batch.num_graphs)

    train_avg = finalize_loss_meter(train_meter, len(train_loader.dataset))

    # ========================================================
    # Validation
    # ========================================================
    model.eval()
    val_meter = init_loss_meter()

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            pred = model(batch)
            target = batch.y

            loss, loss_dict = loss_mse_denorm_physics(
                pred,
                target,
                batch,
                y_mean_dev,
                y_std_dev,
                x_mean_dev,
                x_std_dev,
                n_bus,
                baseMVA=baseMVA,
                w_v_pv=w_v_pv,
                w_v_pq=w_v_pq,
                w_d=w_d,
                w_q=w_q,
                lambda_p_mis=lambda_p_mis,
                lambda_q_mis=lambda_q_mis
            )

            update_loss_meter(val_meter, loss_dict, batch.num_graphs)

    val_avg = finalize_loss_meter(val_meter, len(val_loader.dataset))

    # ========================================================
    # Scheduler uses total validation loss
    # ========================================================
    scheduler.step(val_avg["total"])

    # ========================================================
    # Save history
    # ========================================================
    epoch_hist.append(epoch)

    for k in LOSS_KEYS:
        hist[f"train_{k}"].append(train_avg[k])
        hist[f"val_{k}"].append(val_avg[k])

    # ========================================================
    # Grouped early stopping
    #
    # Save model only when total validation loss improves.
    # Reset patience if any important group improves:
    #   total, supervised, physics, PV voltage loss, PQ voltage loss
    # ========================================================
    improved_total = has_improved(
        val_avg["total"],
        best_total,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel
    )

    improved_supervised = has_improved(
        val_avg["supervised"],
        best_supervised,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel
    )

    improved_physics = has_improved(
        val_avg["physics"],
        best_physics,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel
    )

    improved_v_pv = has_improved(
        val_avg["v_pv"],
        best_v_pv,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel
    )

    improved_v_pq = has_improved(
        val_avg["v_pq"],
        best_v_pq,
        min_delta_abs=min_delta_abs,
        min_delta_rel=min_delta_rel
    )

    if improved_total:
        best_total = val_avg["total"]
        best_epoch = epoch
        torch.save(model.state_dict(), model_path)

    if improved_supervised:
        best_supervised = val_avg["supervised"]

    if improved_physics:
        best_physics = val_avg["physics"]

    if improved_v_pv:
        best_v_pv = val_avg["v_pv"]

    if improved_v_pq:
        best_v_pq = val_avg["v_pq"]

    if improved_total or improved_supervised or improved_physics or improved_v_pv or improved_v_pq:
        pat_count = 0
    else:
        pat_count += 1

    # ========================================================
    # Print
    # Raw losses are printed first.
    # Weighted contributions are printed in parentheses.
    # ========================================================
    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "

            f"TrainTotal {train_avg['total']:.6e} | "
            f"TrainSup {train_avg['supervised']:.6e} | "
            f"TrainPhy {train_avg['physics']:.6e} | "

            f"TrainPVV {train_avg['v_pv']:.6e} ({train_avg['v_pv_w']:.6e}) | "
            f"TrainPQV {train_avg['v_pq']:.6e} ({train_avg['v_pq_w']:.6e}) | "
            f"TrainVall {train_avg['v_all']:.6e} ({train_avg['v_all_w']:.6e}) | "
            f"TrainDelta {train_avg['delta']:.6e} ({train_avg['delta_w']:.6e}) | "
            f"TrainQgen {train_avg['qgen']:.6e} ({train_avg['qgen_w']:.6e}) | "
            f"TrainPmis {train_avg['pmis']:.6e} ({train_avg['pmis_w']:.6e}) | "
            f"TrainQmis {train_avg['qmis']:.6e} ({train_avg['qmis_w']:.6e}) | "

            f"ValTotal {val_avg['total']:.6e} | "
            f"ValSup {val_avg['supervised']:.6e} | "
            f"ValPhy {val_avg['physics']:.6e} | "

            f"ValPVV {val_avg['v_pv']:.6e} ({val_avg['v_pv_w']:.6e}) | "
            f"ValPQV {val_avg['v_pq']:.6e} ({val_avg['v_pq_w']:.6e}) | "
            f"ValVall {val_avg['v_all']:.6e} ({val_avg['v_all_w']:.6e}) | "
            f"ValDelta {val_avg['delta']:.6e} ({val_avg['delta_w']:.6e}) | "
            f"ValQgen {val_avg['qgen']:.6e} ({val_avg['qgen_w']:.6e}) | "
            f"ValPmis {val_avg['pmis']:.6e} ({val_avg['pmis_w']:.6e}) | "
            f"ValQmis {val_avg['qmis']:.6e} ({val_avg['qmis_w']:.6e}) | "

            f"Improve(T/S/P/PVV/PQV)=({int(improved_total)}/"
            f"{int(improved_supervised)}/"
            f"{int(improved_physics)}/"
            f"{int(improved_v_pv)}/"
            f"{int(improved_v_pq)}) | "
            f"Patience {pat_count}/{patience} | "
            f"BestTotal {best_total:.6e} @ {best_epoch}"
        )

    if pat_count >= patience:
        print(
            f"Early stopping at epoch {epoch}. "
            f"Best total validation loss: {best_total:.6e} @ epoch {best_epoch}. "
            f"Best supervised validation loss: {best_supervised:.6e}. "
            f"Best physics validation loss: {best_physics:.6e}. "
            f"Best PV voltage validation loss: {best_v_pv:.6e}. "
            f"Best PQ voltage validation loss: {best_v_pq:.6e}."
        )
        break

print("Best model saved to:", model_path)


# ============================================================
# Plot training history
# ============================================================

# ------------------------------------------------------------
# Total, supervised, and physics losses
# ------------------------------------------------------------
plt.figure(figsize=(9, 5))
plt.plot(epoch_hist, hist["train_total"], label="Train Total Loss", linewidth=2)
plt.plot(epoch_hist, hist["val_total"], label="Val Total Loss", linewidth=2)
plt.plot(epoch_hist, hist["train_supervised"], label="Train Supervised Loss", linewidth=2)
plt.plot(epoch_hist, hist["val_supervised"], label="Val Supervised Loss", linewidth=2)
plt.plot(epoch_hist, hist["train_physics"], label="Train Physics Loss", linewidth=2)
plt.plot(epoch_hist, hist["val_physics"], label="Val Physics Loss", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss, log scale")
plt.title("Total / Supervised / Physics Loss")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(loss_fig_path, dpi=300)
plt.show()


# ------------------------------------------------------------
# Raw component MSE losses
# ------------------------------------------------------------
plt.figure(figsize=(11, 6))
plt.plot(epoch_hist, hist["train_v_pv"], label="Train PV V MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_v_pv"], label="Val PV V MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_v_pq"], label="Train PQ V MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_v_pq"], label="Val PQ V MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_v_all"], label="Train All V MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_v_all"], label="Val All V MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_delta"], label="Train Delta MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_delta"], label="Val Delta MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_qgen"], label="Train Qgen MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_qgen"], label="Val Qgen MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_pmis"], label="Train Pmis PU MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_pmis"], label="Val Pmis PU MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_qmis"], label="Train Qmis PU MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_qmis"], label="Val Qmis PU MSE", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Raw Component MSE, log scale")
plt.title("Raw Loss Components")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(loss_sep_fig_path, dpi=300)
plt.show()


# ------------------------------------------------------------
# Weighted component contributions
# ------------------------------------------------------------
weighted_loss_fig_path = f"[{n_bus}bus]_WeightedLossContributions_NNConv_VDeltaQgen_Physics_PVPQV.png"

plt.figure(figsize=(11, 6))
plt.plot(epoch_hist, hist["train_v_pv_w"], label="Train 100*PV V MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_v_pv_w"], label="Val 100*PV V MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_v_pq_w"], label="Train 100*PQ V MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_v_pq_w"], label="Val 100*PQ V MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_delta_w"], label="Train 10*Delta MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_delta_w"], label="Val 10*Delta MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_qgen_w"], label="Train 10*Qgen MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_qgen_w"], label="Val 10*Qgen MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_pmis_w"], label="Train Lambda*Pmis MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_pmis_w"], label="Val Lambda*Pmis MSE", linewidth=2)
plt.plot(epoch_hist, hist["train_qmis_w"], label="Train Lambda*Qmis MSE", linewidth=2)
plt.plot(epoch_hist, hist["val_qmis_w"], label="Val Lambda*Qmis MSE", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Weighted Loss Contribution, log scale")
plt.title("Weighted Contributions to Total Loss")
plt.grid(True, which="both")
plt.legend()
plt.tight_layout()
plt.savefig(weighted_loss_fig_path, dpi=300)
plt.show()


# ============================================================
# Test evaluation
# ============================================================
def evaluate_on_loader_separate(model, loader, y_mean, y_std, n_bus, device):
    preds_denorm = []
    trues_denorm = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred_norm = model(batch)
            true_norm = batch.y

            bs = batch.num_graphs
            y_mean_exp = y_mean.repeat(bs, 1).to(device)
            y_std_exp  = y_std.repeat(bs, 1).to(device)

            pred = denormalize(pred_norm, y_mean_exp, y_std_exp)
            true = denormalize(true_norm, y_mean_exp, y_std_exp)

            preds_denorm.append(pred.cpu())
            trues_denorm.append(true.cpu())

    preds_denorm = torch.cat(preds_denorm, dim=0).numpy()
    trues_denorm = torch.cat(trues_denorm, dim=0).numpy()

    mse_total  = mean_squared_error(trues_denorm, preds_denorm)
    rmse_total = np.sqrt(mse_total)
    mae_total  = mean_absolute_error(trues_denorm, preds_denorm)

    mse_v  = mean_squared_error(trues_denorm[:, 0], preds_denorm[:, 0])
    rmse_v = np.sqrt(mse_v)
    mae_v  = mean_absolute_error(trues_denorm[:, 0], preds_denorm[:, 0])

    mse_d  = mean_squared_error(trues_denorm[:, 1], preds_denorm[:, 1])
    rmse_d = np.sqrt(mse_d)
    mae_d  = mean_absolute_error(trues_denorm[:, 1], preds_denorm[:, 1])

    mse_q  = mean_squared_error(trues_denorm[:, 2], preds_denorm[:, 2])
    rmse_q = np.sqrt(mse_q)
    mae_q  = mean_absolute_error(trues_denorm[:, 2], preds_denorm[:, 2])

    max_abs_v = np.max(np.abs(trues_denorm[:, 0] - preds_denorm[:, 0]))
    max_abs_d = np.max(np.abs(trues_denorm[:, 1] - preds_denorm[:, 1]))
    max_abs_q = np.max(np.abs(trues_denorm[:, 2] - preds_denorm[:, 2]))

    return {
        "MSE_total": mse_total,
        "RMSE_total": rmse_total,
        "MAE_total": mae_total,

        "MSE_V": mse_v,
        "RMSE_V": rmse_v,
        "MAE_V": mae_v,

        "MSE_delta": mse_d,
        "RMSE_delta": rmse_d,
        "MAE_delta": mae_d,

        "MSE_Qgen": mse_q,
        "RMSE_Qgen": rmse_q,
        "MAE_Qgen": mae_q,

        "MaxAbsErr_V": max_abs_v,
        "MaxAbsErr_delta": max_abs_d,
        "MaxAbsErr_Qgen": max_abs_q,

        "preds_denorm": preds_denorm,
        "trues_denorm": trues_denorm
    }


# ============================================================
# Test + export dataset-style workbook
# ============================================================
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

res = evaluate_on_loader_separate(model, test_loader, y_mean, y_std, n_bus, device)

preds_denorm = res.pop("preds_denorm")
trues_denorm = res.pop("trues_denorm")
res["file"] = test_book

print("\nTest file:", test_book)
print(res)

pred_3d = preds_denorm.reshape(-1, n_bus, 3)
true_3d = trues_denorm.reshape(-1, n_bus, 3)

# ------------------------------------------------------------
# Predicted V / delta / Qgen sheet
# ------------------------------------------------------------
pred_rows = []
for i in range(pred_3d.shape[0]):
    row = {
        "sample_id": int(test_meta_df.iloc[i]["sample_id"]),
        "scenario_name": test_meta_df.iloc[i]["scenario_name"],
        "hour": int(test_meta_df.iloc[i]["hour"]),
        "contingency_index": int(test_meta_df.iloc[i]["contingency_index"])
    }

    for b in range(n_bus):
        row[f"pred_V_{b+1}"] = pred_3d[i, b, 0]
        row[f"pred_delta_{b+1}"] = pred_3d[i, b, 1]
        row[f"pred_Qgen_{b+1}"] = pred_3d[i, b, 2]

    pred_rows.append(row)

pred_vdelta_df = pd.DataFrame(pred_rows)

# ------------------------------------------------------------
# True V / delta / Qgen sheet
# ------------------------------------------------------------
true_rows = []
for i in range(true_3d.shape[0]):
    row = {
        "sample_id": int(test_meta_df.iloc[i]["sample_id"]),
        "scenario_name": test_meta_df.iloc[i]["scenario_name"],
        "hour": int(test_meta_df.iloc[i]["hour"]),
        "contingency_index": int(test_meta_df.iloc[i]["contingency_index"])
    }

    for b in range(n_bus):
        row[f"true_V_{b+1}"] = true_3d[i, b, 0]
        row[f"true_delta_{b+1}"] = true_3d[i, b, 1]
        row[f"true_Qgen_{b+1}"] = true_3d[i, b, 2]

    true_rows.append(row)

true_vdelta_df = pd.DataFrame(true_rows)

# ------------------------------------------------------------
# Branch power flow reconstructed from PREDICTED V/delta
# ------------------------------------------------------------
branchflow_rows = []
n_branch = len(branch_df)

for i in range(pred_3d.shape[0]):
    vm = pred_3d[i, :, 0]
    va = pred_3d[i, :, 1]
    cont_idx = int(test_meta_df.iloc[i]["contingency_index"])

    active_branch_df, active_mask = get_active_branch_df(branch_df, cont_idx)

    Sbr_active = compute_branch_flows_from_active_topology(
        vm, va, active_branch_df, n_bus=n_bus, baseMVA=baseMVA
    )

    Sbr_full = build_full_branch_flow_vector(Sbr_active, active_mask, n_branch)

    row = {
        "sample_id": int(test_meta_df.iloc[i]["sample_id"]),
        "scenario_name": test_meta_df.iloc[i]["scenario_name"],
        "hour": int(test_meta_df.iloc[i]["hour"]),
        "contingency_index": cont_idx
    }

    for l in range(n_branch):
        row[f"Sbranch_{l+1}"] = Sbr_full[l]

    branchflow_rows.append(row)

pred_branchflow_df = pd.DataFrame(branchflow_rows)

# ------------------------------------------------------------
# Violation analysis from predicted V and predicted branch flow
# ------------------------------------------------------------
Vmin = 0.94
Vmax = 1.06

s_cols = [c for c in pred_branchflow_df.columns if c.startswith("Sbranch_")]
s_cols = sorted(s_cols, key=lambda x: int(x.split("_")[1]))

rateA = branch_df["rateA"].values.astype(np.float64)

violation_rows = []

for i in range(pred_3d.shape[0]):
    sample_id = int(test_meta_df.iloc[i]["sample_id"])
    scenario_name = test_meta_df.iloc[i]["scenario_name"]
    hour = int(test_meta_df.iloc[i]["hour"])
    contingency_index = int(test_meta_df.iloc[i]["contingency_index"])

    # ========================================================
    # Branch violation part
    # ========================================================
    s_pred = pred_branchflow_df.loc[i, s_cols].values.astype(np.float64)

    valid_branch_mask = np.isfinite(s_pred) & np.isfinite(rateA) & (rateA > 0)

    overload_pct = np.zeros_like(s_pred, dtype=np.float64)
    overload_pct[valid_branch_mask] = np.maximum(
        0.0,
        (s_pred[valid_branch_mask] - rateA[valid_branch_mask]) / rateA[valid_branch_mask] * 100.0
    )

    branch_viol_mask = overload_pct > 0
    num_branch_viol = int(np.sum(branch_viol_mask))

    worst_overload_pct = float(np.max(overload_pct)) if num_branch_viol > 0 else 0.0
    sum_branch_overload_pct = float(np.sum(overload_pct))

    if num_branch_viol > 0:
        worst_branch_idx = int(np.argmax(overload_pct) + 1)
        worst_flow_mva = float(s_pred[worst_branch_idx - 1])
        worst_limit_mva = float(rateA[worst_branch_idx - 1])
    else:
        worst_branch_idx = np.nan
        worst_flow_mva = np.nan
        worst_limit_mva = np.nan

    # ========================================================
    # Voltage violation part
    # ========================================================
    vm = pred_3d[i, :, 0]

    underv_mask = vm < Vmin
    overvol_mask = vm > Vmax
    volt_violation_mask = underv_mask | overvol_mask

    num_underv = int(np.sum(underv_mask))
    num_overv = int(np.sum(overvol_mask))
    num_volt_viol = int(np.sum(volt_violation_mask))

    v_dev = np.zeros_like(vm, dtype=np.float64)
    v_dev[underv_mask] = Vmin - vm[underv_mask]
    v_dev[overvol_mask] = vm[overvol_mask] - Vmax

    worst_v_dev = float(np.max(v_dev)) if num_volt_viol > 0 else 0.0
    sum_voltage_dev_pu = float(np.sum(v_dev))

    if num_volt_viol > 0:
        worst_v_bus = int(np.argmax(v_dev) + 1)
        worst_vm = float(vm[worst_v_bus - 1])
    else:
        worst_v_bus = np.nan
        worst_vm = np.nan

    total_violations = num_branch_viol + num_volt_viol

    violation_rows.append({
        "sample_id": sample_id,
        "scenario_name": scenario_name,
        "hour": hour,
        "contingency_index": contingency_index,

        "NumBranchViolations": num_branch_viol,
        "WorstOverload_pct": worst_overload_pct,
        "SumBranchOverload_pct": sum_branch_overload_pct,
        "WorstOverloadBranch": worst_branch_idx,
        "WorstFlow_MVA": worst_flow_mva,
        "Limit_MVA": worst_limit_mva,

        "NumVoltageViolations": num_volt_viol,
        "NumUndervoltage": num_underv,
        "NumOvervoltage": num_overv,
        "WorstVoltageDeviation_pu": worst_v_dev,
        "SumVoltageDeviation_pu": sum_voltage_dev_pu,
        "WorstVBus": worst_v_bus,
        "WorstVM_pu": worst_vm,

        "TotalViolations": total_violations
    })

violations_df = pd.DataFrame(violation_rows)

# ------------------------------------------------------------
# Add separate severity ranks
# ------------------------------------------------------------
violations_df["BranchSeverityRank"] = np.nan
violations_df["VoltageSeverityRank"] = np.nan

branch_nonzero_idx = violations_df.index[violations_df["NumBranchViolations"] > 0]

branch_rank_order = violations_df.loc[branch_nonzero_idx].sort_values(
    by=["SumBranchOverload_pct", "NumBranchViolations", "WorstOverload_pct"],
    ascending=[False, False, False]
).index

violations_df.loc[branch_rank_order, "BranchSeverityRank"] = np.arange(1, len(branch_rank_order) + 1)

voltage_nonzero_idx = violations_df.index[violations_df["NumVoltageViolations"] > 0]

voltage_rank_order = violations_df.loc[voltage_nonzero_idx].sort_values(
    by=["SumVoltageDeviation_pu", "NumVoltageViolations", "WorstVoltageDeviation_pu"],
    ascending=[False, False, False]
).index

violations_df.loc[voltage_rank_order, "VoltageSeverityRank"] = np.arange(1, len(voltage_rank_order) + 1)

severe_branch_df = violations_df[violations_df["NumBranchViolations"] > 0].sort_values(
    by=["BranchSeverityRank"],
    ascending=[True]
).reset_index(drop=True)

severe_voltage_df = violations_df[violations_df["NumVoltageViolations"] > 0].sort_values(
    by=["VoltageSeverityRank"],
    ascending=[True]
).reset_index(drop=True)

# ------------------------------------------------------------
# Read true branch power flow from test workbook
# ------------------------------------------------------------
test_branchflow_df = read_branch_power_flow_from_workbook(test_book)

if not np.array_equal(test_meta_df["sample_id"].values,
                      test_branchflow_df["sample_id"].values):
    raise ValueError("sample_id mismatch between sample_meta and branch_power_flow")

true_branch_mat, s_cols = extract_true_branch_flow_matrix(test_branchflow_df)
pred_branch_mat = pred_branchflow_df[s_cols].values.astype(np.float64)

# Ignore NaN entries such as outaged branch positions
mask = np.isfinite(true_branch_mat) & np.isfinite(pred_branch_mat)
true_branch_valid = true_branch_mat[mask]
pred_branch_valid = pred_branch_mat[mask]

mse_branch = mean_squared_error(true_branch_valid, pred_branch_valid)
rmse_branch = np.sqrt(mse_branch)
mae_branch = mean_absolute_error(true_branch_valid, pred_branch_valid)
max_abs_branch = np.max(np.abs(true_branch_valid - pred_branch_valid))

metrics_df = pd.DataFrame([res])
metrics_df["MSE_branch"] = mse_branch
metrics_df["RMSE_branch"] = rmse_branch
metrics_df["MAE_branch"] = mae_branch
metrics_df["MaxAbsErr_branch"] = max_abs_branch

metrics_df.to_csv(summary_csv, index=False)
print("Saved test summary to:", summary_csv)

print("Branch-flow metrics:")
print("MSE_branch =", mse_branch)
print("RMSE_branch =", rmse_branch)
print("MAE_branch =", mae_branch)
print("MaxAbsErr_branch =", max_abs_branch)

# ------------------------------------------------------------
# Find where MaxAbsErr_branch occurs
# ------------------------------------------------------------
abs_err_branch = np.abs(true_branch_mat - pred_branch_mat)
abs_err_branch_masked = np.where(mask, abs_err_branch, np.nan)

flat_idx = np.nanargmax(abs_err_branch_masked)
sample_idx, branch_col_idx = np.unravel_index(flat_idx, abs_err_branch_masked.shape)

max_abs_branch = abs_err_branch_masked[sample_idx, branch_col_idx]

branch_name = s_cols[branch_col_idx]
branch_index = int(branch_name.split("_")[1])

sample_id = int(test_meta_df.iloc[sample_idx]["sample_id"])
scenario_name = test_meta_df.iloc[sample_idx]["scenario_name"]
hour = int(test_meta_df.iloc[sample_idx]["hour"])
contingency_index = int(test_meta_df.iloc[sample_idx]["contingency_index"])

pred_val = pred_branch_mat[sample_idx, branch_col_idx]
true_val = true_branch_mat[sample_idx, branch_col_idx]

from_bus = int(branch_df.iloc[branch_index - 1]["from_bus"])
to_bus   = int(branch_df.iloc[branch_index - 1]["to_bus"])

print("\n========== MaxAbsErr_branch location ==========")
print("sample_idx =", sample_idx)
print("sample_id =", sample_id)
print("scenario_name =", scenario_name)
print("hour =", hour)
print("contingency_index =", contingency_index)
print("branch column =", branch_name)
print("branch index =", branch_index)
print("predicted Sbranch =", pred_val)
print("true Sbranch =", true_val)
print("absolute error =", max_abs_branch)
print("from_bus =", from_bus)
print("to_bus   =", to_bus)

# ------------------------------------------------------------
# Export workbook
# ------------------------------------------------------------
with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
    metrics_df.to_excel(writer, sheet_name="metrics_summary", index=False)
    pred_vdelta_df.to_excel(writer, sheet_name="pred_v_delta_qgen", index=False)
    true_vdelta_df.to_excel(writer, sheet_name="true_v_delta_qgen", index=False)
    pred_branchflow_df.to_excel(writer, sheet_name="reconstructed_from_pred", index=False)
    test_branchflow_df.to_excel(writer, sheet_name="true_branch_power_flow", index=False)
    violations_df.to_excel(writer, sheet_name="violations_summary", index=False)
    severe_branch_df.to_excel(writer, sheet_name="severe_branch_rank", index=False)
    severe_voltage_df.to_excel(writer, sheet_name="severe_voltage_rank", index=False)
    test_meta_df.to_excel(writer, sheet_name="sample_meta", index=False)

print("Saved full prediction workbook to:", out_excel)
print(pred_vdelta_df.head())
print(pred_branchflow_df.head())