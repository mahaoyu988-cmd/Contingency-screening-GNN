import os
import copy
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


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
max_epochs = 100
patience = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# -----------------------------
# Data file paths
# -----------------------------
train_xlsx = r"C:\Users\mahao\gnn_case118_dataset\train.xlsx"
val_xlsx   = r"C:\Users\mahao\gnn_case118_dataset\val.xlsx"

test_xlsx_list = [
    r"C:\Users\mahao\gnn_case118_dataset\test.xlsx"
   ]

# -----------------------------
# Network topology / branch data files
# You should prepare these from MATLAB or CSV
# -----------------------------
branch_csv = r"C:\Users\mahao\gnn_case118_dataset\case118_branch_data.csv"
# expected columns:
# from_bus, to_bus, r, x, b, rateA, ratio

cont_train_csv = r"C:\Users\mahao\gnn_case118_dataset\train_contingency_info.csv"
cont_val_csv   = r"C:\Users\mahao\gnn_case118_dataset\val_contingency_info.csv"

test_cont_csv_list = [
    r"C:\Users\mahao\gnn_case118_dataset\test_contingency_info.csv",
]

model_path = f"[{n_bus}bus]_Best_GCN_dynamic.pt"
stats_path = f"[{n_bus}bus]_NormStats_dynamic.pt"
summary_csv = f"[{n_bus}bus]_GCN_TestSummary_dynamic.csv"


# ============================================================
# Utility functions
# ============================================================
def load_branch_data(branch_csv):
    """
    Reads the original intact network branch list.

    Returns:
        from_bus_orig: list[int] zero-based
        to_bus_orig:   list[int] zero-based
        edge_attr_orig: list[list[float]]
    """
    df = pd.read_csv(branch_csv)

    # convert bus numbering from 1-based to 0-based if needed
    from_bus_orig = (df["from_bus"].values.astype(int) - 1).tolist()
    to_bus_orig   = (df["to_bus"].values.astype(int) - 1).tolist()

    # edge features
    edge_attr_orig = df[["r", "x", "b", "rateA", "ratio"]].values.astype(np.float32).tolist()

    return from_bus_orig, to_bus_orig, edge_attr_orig


def load_contingency_info(cont_csv):
    """
    Reads contingency line outages for each sample.
    One row per sample, with columns:
        outage_from, outage_to
    """
    df = pd.read_csv(cont_csv)
    f_out = (df["outage_from"].values.astype(int) - 1).tolist()
    t_out = (df["outage_to"].values.astype(int) - 1).tolist()
    return f_out, t_out


def read_excel_dataset(xlsx_path):
    return pd.read_excel(xlsx_path).values


def split_features_targets(dataset_np, n_bus):
    """
    Expected row structure:
    [optional_id, bus1_P, bus1_Q, bus1_V, bus1_delta, bus2_P, ...]

    We now use:
      Input per bus:  [P, Q, is_pv, is_pq, is_slack]  -> 5 features
      Target per bus: [V, delta]                      -> 2 targets

    Important:
      V and delta are NOT included in inputs anymore.
    """
    x_list, y_list = [], []

    for i in range(len(dataset_np)):
        x_sample, y_sample = [], []

        for n in range(n_bus):
            P = dataset_np[i, 4 * n + 1]
            Q = dataset_np[i, 4 * n + 2]
            V = dataset_np[i, 4 * n + 3]
            d = dataset_np[i, 4 * n + 4]

            # simple bus-type heuristic from your old code
            is_pv = 0
            is_pq = 0
            is_slack = 0

            if n == 0:
                is_slack = 1
            elif Q == 0:
                is_pv = 1
            else:
                is_pq = 1

            # input features: NO target leakage
            x_sample.append([P, Q, is_pv, is_pq, is_slack])

            # targets
            y_sample.append([V, d])

        x_list.append(x_sample)
        y_list.append(y_sample)

    x = torch.tensor(x_list, dtype=torch.float32)   # [N, 118, 5]
    y = torch.tensor(y_list, dtype=torch.float32)   # [N, 118, 2]
    return x, y


def fit_normalization(x_train, y_train):
    """
    Fit normalization stats using training set only.
    """
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

    # preserve one-hot bus type flags: last 3 columns
    x_norm[:, :, 2:] = x[:, :, 2:]
    return x_norm, y_norm


def denormalize(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def build_dynamic_dataset(
    x_tensor,
    y_tensor,
    from_bus_orig,
    to_bus_orig,
    edge_attr_orig,
    from_bus_contingency,
    to_bus_contingency,
):
    """
    Build a list of PyG Data objects with dynamic topology.
    Each sample removes one outage line.
    """
    data_list = []
    n_samples = x_tensor.size(0)

    for i in range(n_samples):
        x = x_tensor[i]
        y = y_tensor[i]

        f_out = from_bus_contingency[i]
        t_out = to_bus_contingency[i]

        filtered_edges = []
        filtered_attr = []

        for (f, t, attr) in zip(from_bus_orig, to_bus_orig, edge_attr_orig):
            # remove the outage line in either direction
            if not ((f == f_out and t == t_out) or (f == t_out and t == f_out)):
                filtered_edges.append((f, t))
                filtered_attr.append(attr)

        from_buses_temp = [f for f, t in filtered_edges]
        to_buses_temp   = [t for f, t in filtered_edges]

        # bidirectional edge index
        edge_index = torch.tensor(
            [
                from_buses_temp + to_buses_temp,
                to_buses_temp   + from_buses_temp
            ],
            dtype=torch.long
        )

        # duplicate edge attributes for reverse direction
        edge_attr = torch.tensor(
            filtered_attr + filtered_attr,
            dtype=torch.float32
        )

        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        data_list.append(data)

    return data_list


# ============================================================
# Model with edge attributes
# ============================================================
class EdgeGCN(nn.Module):
    """
    We concatenate aggregated edge information into node embeddings
    indirectly by using a simple edge-aware preprocessing trick:
    linearly project node features, then standard GCN layers.

    If you want fully explicit edge-conditioned message passing later,
    we can upgrade this to NNConv / GINEConv.
    """
    def __init__(self, in_channels=5, hidden_channels=64, out_channels=2, dropout=0.1):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        out = self.lin(x)
        return out


# ============================================================
# Loss in original units
# ============================================================
def loss_mse_denorm(pred_norm, target_norm, y_mean, y_std, n_bus):
    total_nodes = pred_norm.size(0)
    batch_size_local = total_nodes // n_bus

    y_mean_exp = y_mean.repeat(batch_size_local, 1).to(pred_norm.device)
    y_std_exp  = y_std.repeat(batch_size_local, 1).to(pred_norm.device)

    pred   = denormalize(pred_norm, y_mean_exp, y_std_exp)
    target = denormalize(target_norm, y_mean_exp, y_std_exp)
    return torch.mean((pred - target) ** 2)


# ============================================================
# Prepare train / val datasets
# ============================================================
print("Loading branch data...")
from_bus_orig, to_bus_orig, edge_attr_orig = load_branch_data(branch_csv)

print("Loading train/val datasets...")
train_np = read_excel_dataset(train_xlsx)
val_np   = read_excel_dataset(val_xlsx)

x_train_raw, y_train_raw = split_features_targets(train_np, n_bus)
x_val_raw,   y_val_raw   = split_features_targets(val_np, n_bus)

# Fit normalization on training only
x_mean, x_std, y_mean, y_std = fit_normalization(x_train_raw, y_train_raw)

x_train, y_train = apply_normalization(x_train_raw, y_train_raw, x_mean, x_std, y_mean, y_std)
x_val,   y_val   = apply_normalization(x_val_raw,   y_val_raw,   x_mean, x_std, y_mean, y_std)

# Load outage info
f_out_train, t_out_train = load_contingency_info(cont_train_csv)
f_out_val,   t_out_val   = load_contingency_info(cont_val_csv)

train_list = build_dynamic_dataset(
    x_train, y_train,
    from_bus_orig, to_bus_orig, edge_attr_orig,
    f_out_train, t_out_train
)

val_list = build_dynamic_dataset(
    x_val, y_val,
    from_bus_orig, to_bus_orig, edge_attr_orig,
    f_out_val, t_out_val
)

train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_list,   batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_list)}")
print(f"Val samples:   {len(val_list)}")

# save normalization stats
torch.save({
    "x_mean": x_mean,
    "x_std": x_std,
    "y_mean": y_mean,
    "y_std": y_std
}, stats_path)


# ============================================================
# Train
# ============================================================
model = EdgeGCN(in_channels=5, hidden_channels=64, out_channels=2, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

best_val = float("inf")
best_epoch = -1
pat_count = 0

print("Training...")
for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y

        loss = loss_mse_denorm(pred, target, y_mean.to(device), y_std.to(device), n_bus)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_loader.dataset)

    # validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            pred = model(batch)
            target = batch.y
            loss = loss_mse_denorm(pred, target, y_mean.to(device), y_std.to(device), n_bus)
            val_loss += loss.item() * batch.num_graphs

    val_loss /= len(val_loader.dataset)
    scheduler.step(val_loss)

    if val_loss < best_val - 1e-12:
        best_val = val_loss
        best_epoch = epoch
        pat_count = 0
        torch.save(model.state_dict(), model_path)
    else:
        pat_count += 1

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | TrainLoss {train_loss:.6e} | ValLoss {val_loss:.6e} | BestVal {best_val:.6e} @ {best_epoch}")

    if pat_count >= patience:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break

print("Best model saved to:", model_path)


# ============================================================
# Testing
# ============================================================
def evaluate_on_dataset(model, xlsx_path, cont_csv_path,
                        from_bus_orig, to_bus_orig, edge_attr_orig,
                        x_mean, x_std, y_mean, y_std,
                        n_bus, batch_size, device):
    data_np = read_excel_dataset(xlsx_path)
    x_raw, y_raw = split_features_targets(data_np, n_bus)
    x_norm, y_norm = apply_normalization(x_raw, y_raw, x_mean, x_std, y_mean, y_std)

    f_out, t_out = load_contingency_info(cont_csv_path)

    data_list = build_dynamic_dataset(
        x_norm, y_norm,
        from_bus_orig, to_bus_orig, edge_attr_orig,
        f_out, t_out
    )

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

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

    mse   = mean_squared_error(trues_denorm, preds_denorm)
    rmse  = np.sqrt(mse)
    nrmse = rmse / np.std(trues_denorm)
    mae   = mean_absolute_error(trues_denorm, preds_denorm)
    r2    = r2_score(trues_denorm, preds_denorm)

    return {"MSE": mse, "RMSE": rmse, "NRMSE": nrmse, "MAE": mae, "R2": r2}


# reload best model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_results = []
for xlsx_path, cont_csv_path in zip(test_xlsx_list, test_cont_csv_list):
    res = evaluate_on_dataset(
        model, xlsx_path, cont_csv_path,
        from_bus_orig, to_bus_orig, edge_attr_orig,
        x_mean, x_std, y_mean, y_std,
        n_bus, batch_size, device
    )
    res["file"] = xlsx_path
    all_results.append(res)

    print("\nTest file:", xlsx_path)
    print(res)

df = pd.DataFrame(all_results)
df.to_csv(summary_csv, index=False)
print("Saved test summary to:", summary_csv)
