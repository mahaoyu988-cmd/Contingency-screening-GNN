import os
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

# ============================================================
# Paths
# ============================================================
data_dir = r"C:\Users\mahao\gnn_case118_dataset"

train_xlsx = os.path.join(data_dir, "train.xlsx")
val_xlsx   = os.path.join(data_dir, "val.xlsx")

test_xlsx_list = [
    os.path.join(data_dir, "test.xlsx")
]

branch_csv   = os.path.join(data_dir, "case118_branch_data.csv")
bus_type_csv = os.path.join(data_dir, "case118_bus_types.csv")

model_path   = f"[{n_bus}bus]_Best_GCN_fixed.pt"
stats_path   = f"[{n_bus}bus]_NormStats_fixed.pt"
summary_csv  = f"[{n_bus}bus]_GCN_TestSummary_fixed.csv"


# ============================================================
# Utility functions
# ============================================================
def read_excel_dataset(xlsx_path):
    return pd.read_excel(xlsx_path).values


def load_branch_data(branch_csv):
    """
    Load intact network branch list and edge attributes.
    Returns zero-based bus indices.
    """
    df = pd.read_csv(branch_csv)

    from_bus = (df["from_bus"].values.astype(int) - 1).tolist()
    to_bus   = (df["to_bus"].values.astype(int) - 1).tolist()

    edge_attr = df[["r", "x", "b", "rateA", "ratio"]].values.astype(np.float32)
    return from_bus, to_bus, edge_attr


def load_bus_types(bus_type_csv):
    df = pd.read_csv(bus_type_csv)
    is_slack = df["is_slack"].values.astype(np.float32)
    is_pv    = df["is_pv"].values.astype(np.float32)
    is_pq    = df["is_pq"].values.astype(np.float32)
    return is_slack, is_pv, is_pq


def split_features_targets(dataset_np, n_bus, is_slack_arr, is_pv_arr, is_pq_arr):
    """
    Input per bus:
      [P, Q, is_pv, is_pq, is_slack]  -> 5 features

    Target per bus:
      [V, delta] -> 2 targets
    """
    x_list, y_list = [], []

    for i in range(len(dataset_np)):
        x_sample, y_sample = [], []

        for n in range(n_bus):
            P = dataset_np[i, 4 * n + 1]
            Q = dataset_np[i, 4 * n + 2]
            V = dataset_np[i, 4 * n + 3]
            d = dataset_np[i, 4 * n + 4]

            x_sample.append([
                P, Q,
                is_pv_arr[n],
                is_pq_arr[n],
                is_slack_arr[n]
            ])

            y_sample.append([V, d])

        x_list.append(x_sample)
        y_list.append(y_sample)

    x = torch.tensor(x_list, dtype=torch.float32)   # [N, 118, 5]
    y = torch.tensor(y_list, dtype=torch.float32)   # [N, 118, 2]
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

    # preserve bus type one-hot features
    x_norm[:, :, 2:] = x[:, :, 2:]
    return x_norm, y_norm


def denormalize(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def build_fixed_edge_index(from_bus_orig, to_bus_orig):
    """
    Build one intact, bidirectional edge_index for all samples.
    """
    edge_index = torch.tensor(
        [
            from_bus_orig + to_bus_orig,
            to_bus_orig   + from_bus_orig
        ],
        dtype=torch.long
    )
    return edge_index


def build_fixed_dataset(x_tensor, y_tensor, edge_index, edge_attr_orig):
    """
    Build Data list using the same edge_index for every sample.
    """
    data_list = []

    # duplicate edge_attr for reverse direction
    edge_attr = torch.tensor(
        np.vstack([edge_attr_orig, edge_attr_orig]),
        dtype=torch.float32
    )

    for i in range(x_tensor.size(0)):
        data = Data(
            x=x_tensor[i],
            y=y_tensor[i],
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        data_list.append(data)

    return data_list


# ============================================================
# Model
# ============================================================
class FixedEdgeGCN(nn.Module):
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
# Load data
# ============================================================
print("Loading branch and bus-type data...")
from_bus_orig, to_bus_orig, edge_attr_orig = load_branch_data(branch_csv)
is_slack_arr, is_pv_arr, is_pq_arr = load_bus_types(bus_type_csv)

print("Loading train/val datasets...")
train_np = read_excel_dataset(train_xlsx)
val_np   = read_excel_dataset(val_xlsx)

x_train_raw, y_train_raw = split_features_targets(train_np, n_bus, is_slack_arr, is_pv_arr, is_pq_arr)
x_val_raw,   y_val_raw   = split_features_targets(val_np,   n_bus, is_slack_arr, is_pv_arr, is_pq_arr)

# fit normalization using train only
x_mean, x_std, y_mean, y_std = fit_normalization(x_train_raw, y_train_raw)

x_train, y_train = apply_normalization(x_train_raw, y_train_raw, x_mean, x_std, y_mean, y_std)
x_val,   y_val   = apply_normalization(x_val_raw,   y_val_raw,   x_mean, x_std, y_mean, y_std)

# build fixed graph
edge_index = build_fixed_edge_index(from_bus_orig, to_bus_orig)

train_list = build_fixed_dataset(x_train, y_train, edge_index, edge_attr_orig)
val_list   = build_fixed_dataset(x_val,   y_val,   edge_index, edge_attr_orig)

train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_list,   batch_size=batch_size, shuffle=False)

print(f"Train samples: {len(train_list)}")
print(f"Val samples:   {len(val_list)}")
print("Fixed edge_index shape:", edge_index.shape)

torch.save({
    "x_mean": x_mean,
    "x_std": x_std,
    "y_mean": y_mean,
    "y_std": y_std
}, stats_path)


# ============================================================
# Train
# ============================================================
model = FixedEdgeGCN(in_channels=5, hidden_channels=64, out_channels=2, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

best_val = float("inf")
best_epoch = -1
pat_count = 0

print("Training fixed-edge baseline...")
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

print("Best fixed-edge model saved to:", model_path)


# ============================================================
# Testing
# ============================================================
def evaluate_on_dataset(model, xlsx_path,
                        edge_index, edge_attr_orig,
                        is_slack_arr, is_pv_arr, is_pq_arr,
                        x_mean, x_std, y_mean, y_std,
                        n_bus, batch_size, device):
    data_np = read_excel_dataset(xlsx_path)

    x_raw, y_raw = split_features_targets(data_np, n_bus, is_slack_arr, is_pv_arr, is_pq_arr)
    x_norm, y_norm = apply_normalization(x_raw, y_raw, x_mean, x_std, y_mean, y_std)

    data_list = build_fixed_dataset(x_norm, y_norm, edge_index, edge_attr_orig)
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


# load best model
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

all_results = []
for xlsx_path in test_xlsx_list:
    res = evaluate_on_dataset(
        model, xlsx_path,
        edge_index, edge_attr_orig,
        is_slack_arr, is_pv_arr, is_pq_arr,
        x_mean, x_std, y_mean, y_std,
        n_bus, batch_size, device
    )
    res["file"] = xlsx_path
    all_results.append(res)

    print("\nTest file:", xlsx_path)
    print(res)

df = pd.DataFrame(all_results)
df.to_csv(summary_csv, index=False)
print("Saved fixed-edge test summary to:", summary_csv)