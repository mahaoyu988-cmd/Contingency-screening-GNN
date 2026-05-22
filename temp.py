import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split
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
max_epochs = 800
patience = 30
val_ratio = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ============================================================
# Data file paths
# ============================================================
train_book = r"C:\Users\haoyu.ma\OneDrive - Washington State University (email.wsu.edu)\Research\Datasetgeneration\case118_scenario_acopf_dataset_125mixcases.xlsx"
test_book  = r"C:\Users\haoyu.ma\OneDrive - Washington State University (email.wsu.edu)\Research\Datasetgeneration\case118_random_training_dataset.xlsx"

model_path = f"[{n_bus}bus]_Best_GCN_static_125mixcases.pt"
stats_path = f"[{n_bus}bus]_NormStats_static_125mixcases.pt"
summary_csv = f"[{n_bus}bus]_GCN_TestSummary_static_125mixcases.csv"
loss_fig_path = f"[{n_bus}bus]_TrainValLoss_125mixcases.png"
loss_sep_fig_path = f"[{n_bus}bus]_TrainValLoss_Separate_125mixcases.png"


# ============================================================
# Utility functions
# ============================================================
def read_excel_sheet(book_path, sheet_name):
    return pd.read_excel(book_path, sheet_name=sheet_name)


def load_branch_data_from_excel(book_path):
    df = read_excel_sheet(book_path, "case118_branch_data")

    from_bus = (df["from_bus"].values.astype(int) - 1).tolist()
    to_bus   = (df["to_bus"].values.astype(int) - 1).tolist()

    edge_attr = df[["r", "x", "b", "rateA", "ratio"]].values.astype(np.float32).tolist()
    return from_bus, to_bus, edge_attr


def load_bus_type_from_excel(book_path):
    df = read_excel_sheet(book_path, "case118_bus_type")
    df = df.sort_values("bus").reset_index(drop=True)

    is_slack = df["is_slack"].values.astype(np.float32)
    is_pv    = df["is_pv"].values.astype(np.float32)
    is_pq    = df["is_pq"].values.astype(np.float32)

    return is_slack, is_pv, is_pq


def read_dataset_from_workbook(book_path):
    main_df = read_excel_sheet(book_path, "train")
    disp_df = read_excel_sheet(book_path, "train_dispatch")

    main_df = main_df.sort_values("sample_id").reset_index(drop=True)
    disp_df = disp_df.sort_values("sample_id").reset_index(drop=True)

    if not np.array_equal(main_df["sample_id"].values, disp_df["sample_id"].values):
        raise ValueError("sample_id mismatch between train and train_dispatch sheets")

    return main_df, disp_df


def split_features_targets_with_genbus(main_df, dispatch_df, n_bus,
                                       is_slack_vec, is_pv_vec, is_pq_vec, gen_bus_vec):
    """
    main_df row format:
      [sample_id, Pload_1, Qload_1, V_1, delta_1, ..., Pload_118, Qload_118, V_118, delta_118]

    dispatch_df row format:
      [sample_id, Pgen_1, Qgen_1, Pgen_2, Qgen_2, ..., Pgen_ng, Qgen_ng]

    Input per bus:
      [Pload, Qload, Pgen_total_at_bus, Qgen_total_at_bus, is_pv, is_pq, is_slack]

    Target per bus:
      [V, delta]
    """
    main_np = main_df.values
    disp_np = dispatch_df.values

    x_list, y_list = [], []
    ng = len(gen_bus_vec)

    for i in range(len(main_np)):
        x_sample, y_sample = [], []

        Pgen_bus = np.zeros(n_bus, dtype=np.float32)
        Qgen_bus = np.zeros(n_bus, dtype=np.float32)

        for g in range(ng):
            bus_idx = int(gen_bus_vec[g]) - 1
            Pgen = disp_np[i, 2 * g + 1]
            Qgen = disp_np[i, 2 * g + 2]
            Pgen_bus[bus_idx] += Pgen
            Qgen_bus[bus_idx] += Qgen

        for n in range(n_bus):
            Pload = main_np[i, 4 * n + 1]
            Qload = main_np[i, 4 * n + 2]
            V     = main_np[i, 4 * n + 3]
            d     = main_np[i, 4 * n + 4]

            x_sample.append([
                Pload,
                Qload,
                Pgen_bus[n],
                Qgen_bus[n],
                is_pv_vec[n],
                is_pq_vec[n],
                is_slack_vec[n]
            ])
            y_sample.append([V, d])

        x_list.append(x_sample)
        y_list.append(y_sample)

    x = torch.tensor(x_list, dtype=torch.float32)   # [N, 118, 7]
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

    # preserve bus-type flags
    x_norm[:, :, 4:] = x[:, :, 4:]
    return x_norm, y_norm


def denormalize(y_norm, y_mean, y_std):
    return y_norm * y_std + y_mean


def build_static_dataset(x_tensor, y_tensor, from_bus_orig, to_bus_orig, edge_attr_orig):
    data_list = []

    edge_index = torch.tensor(
        [
            from_bus_orig + to_bus_orig,
            to_bus_orig + from_bus_orig
        ],
        dtype=torch.long
    )

    edge_attr = torch.tensor(
        edge_attr_orig + edge_attr_orig,
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
class EdgeGCN(nn.Module):
    def __init__(self, in_channels=7, hidden_channels=64, out_channels=2, dropout=0.1):
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
# Loss in original units: separate total / V / delta
# ============================================================
def loss_mse_denorm_separate(pred_norm, target_norm, y_mean, y_std, n_bus):
    total_nodes = pred_norm.size(0)
    batch_size_local = total_nodes // n_bus

    y_mean_exp = y_mean.repeat(batch_size_local, 1).to(pred_norm.device)
    y_std_exp  = y_std.repeat(batch_size_local, 1).to(pred_norm.device)

    pred   = denormalize(pred_norm, y_mean_exp, y_std_exp)
    target = denormalize(target_norm, y_mean_exp, y_std_exp)

    err = pred - target

    # err[:,0] = V error in pu
    # err[:,1] = delta error in degree
    loss_v = torch.mean(err[:, 0] ** 2)
    loss_d = torch.mean(err[:, 1] ** 2)
    loss_total = torch.mean(err ** 2)

    return loss_total, loss_v, loss_d


# ============================================================
# Load graph / bus metadata
# ============================================================
print("Loading branch data...")
from_bus_orig, to_bus_orig, edge_attr_orig = load_branch_data_from_excel(train_book)

print("Loading bus type data...")
is_slack_vec, is_pv_vec, is_pq_vec = load_bus_type_from_excel(train_book)

# Must match generator order in MATLAB export
gen_bus_vec = [
    1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40,
    42, 46, 49, 54, 55, 56, 59, 61, 62, 65, 66, 69, 70, 72, 73, 74, 76,
    77, 80, 85, 87, 89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110, 111,
    112, 113, 116
]

print("Loading training workbook...")
train_main_df, train_disp_df = read_dataset_from_workbook(train_book)

print("Loading testing workbook...")
test_main_df, test_disp_df = read_dataset_from_workbook(test_book)

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

x_val_raw = x_all_raw[idx_val]
y_val_raw = y_all_raw[idx_val]

# Fit normalization on training only
x_mean, x_std, y_mean, y_std = fit_normalization(x_train_raw, y_train_raw)

x_train, y_train = apply_normalization(x_train_raw, y_train_raw, x_mean, x_std, y_mean, y_std)
x_val, y_val     = apply_normalization(x_val_raw, y_val_raw, x_mean, x_std, y_mean, y_std)
x_test, y_test   = apply_normalization(x_test_raw, y_test_raw, x_mean, x_std, y_mean, y_std)

train_list = build_static_dataset(x_train, y_train, from_bus_orig, to_bus_orig, edge_attr_orig)
val_list   = build_static_dataset(x_val,   y_val,   from_bus_orig, to_bus_orig, edge_attr_orig)
test_list  = build_static_dataset(x_test,  y_test,  from_bus_orig, to_bus_orig, edge_attr_orig)

train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_list,   batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_list,  batch_size=batch_size, shuffle=False)

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
model = EdgeGCN(in_channels=7, hidden_channels=64, out_channels=2, dropout=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

best_val = float("inf")
best_epoch = -1
pat_count = 0

epoch_hist = []

train_loss_hist = []
train_loss_v_hist = []
train_loss_d_hist = []

val_loss_hist = []
val_loss_v_hist = []
val_loss_d_hist = []

print("Training...")
for epoch in range(1, max_epochs + 1):
    model.train()

    train_loss = 0.0
    train_loss_v = 0.0
    train_loss_d = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)
        target = batch.y

        loss, loss_v, loss_d = loss_mse_denorm_separate(
            pred, target, y_mean.to(device), y_std.to(device), n_bus
        )

        loss.backward()
        optimizer.step()

        train_loss   += loss.item()   * batch.num_graphs
        train_loss_v += loss_v.item() * batch.num_graphs
        train_loss_d += loss_d.item() * batch.num_graphs

    train_loss   /= len(train_loader.dataset)
    train_loss_v /= len(train_loader.dataset)
    train_loss_d /= len(train_loader.dataset)

    model.eval()

    val_loss = 0.0
    val_loss_v = 0.0
    val_loss_d = 0.0

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            pred = model(batch)
            target = batch.y

            loss, loss_v, loss_d = loss_mse_denorm_separate(
                pred, target, y_mean.to(device), y_std.to(device), n_bus
            )

            val_loss   += loss.item()   * batch.num_graphs
            val_loss_v += loss_v.item() * batch.num_graphs
            val_loss_d += loss_d.item() * batch.num_graphs

    val_loss   /= len(val_loader.dataset)
    val_loss_v /= len(val_loader.dataset)
    val_loss_d /= len(val_loader.dataset)

    scheduler.step(val_loss)

    epoch_hist.append(epoch)

    train_loss_hist.append(train_loss)
    train_loss_v_hist.append(train_loss_v)
    train_loss_d_hist.append(train_loss_d)

    val_loss_hist.append(val_loss)
    val_loss_v_hist.append(val_loss_v)
    val_loss_d_hist.append(val_loss_d)

    if val_loss < best_val - 1e-12:
        best_val = val_loss
        best_epoch = epoch
        pat_count = 0
        torch.save(model.state_dict(), model_path)
    else:
        pat_count += 1

    if epoch == 1 or epoch % 10 == 0:
        print(
            f"Epoch {epoch:4d} | "
            f"TrainLoss {train_loss:.6e} | TrainV {train_loss_v:.6e} | TrainDelta {train_loss_d:.6e} | "
            f"ValLoss {val_loss:.6e} | ValV {val_loss_v:.6e} | ValDelta {val_loss_d:.6e} | "
            f"BestVal {best_val:.6e} @ {best_epoch}"
        )

    if pat_count >= patience:
        print(f"Early stopping at epoch {epoch}. Best epoch: {best_epoch}")
        break

print("Best model saved to:", model_path)

# ============================================================
# Plot training history: total
# ============================================================
plt.figure(figsize=(8, 5))
plt.plot(epoch_hist, train_loss_hist, label='Train Loss', linewidth=2)
plt.plot(epoch_hist, val_loss_hist, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Total Loss vs Epoch')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(loss_fig_path, dpi=300)
plt.show()

# ============================================================
# Plot training history: separate V / delta
# ============================================================
plt.figure(figsize=(10, 6))
plt.plot(epoch_hist, train_loss_v_hist, label='Train V Loss', linewidth=2)
plt.plot(epoch_hist, val_loss_v_hist, label='Val V Loss', linewidth=2)
plt.plot(epoch_hist, train_loss_d_hist, label='Train Delta Loss', linewidth=2)
plt.plot(epoch_hist, val_loss_d_hist, label='Val Delta Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Separate Losses')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(loss_sep_fig_path, dpi=300)
plt.show()


# ============================================================
# Testing: combined + separate V / delta metrics
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

    # Combined metrics
    mse   = mean_squared_error(trues_denorm, preds_denorm)
    rmse  = np.sqrt(mse)
    nrmse = rmse / np.std(trues_denorm)
    mae   = mean_absolute_error(trues_denorm, preds_denorm)
    r2    = r2_score(trues_denorm, preds_denorm)

    # V only
    mse_v  = mean_squared_error(trues_denorm[:, 0], preds_denorm[:, 0])
    rmse_v = np.sqrt(mse_v)
    mae_v  = mean_absolute_error(trues_denorm[:, 0], preds_denorm[:, 0])

    # Delta only
    mse_d  = mean_squared_error(trues_denorm[:, 1], preds_denorm[:, 1])
    rmse_d = np.sqrt(mse_d)
    mae_d  = mean_absolute_error(trues_denorm[:, 1], preds_denorm[:, 1])

    return {
        "MSE": mse,
        "RMSE": rmse,
        "NRMSE": nrmse,
        "MAE": mae,
        "R2": r2,
        "MSE_V": mse_v,
        "RMSE_V": rmse_v,
        "MAE_V": mae_v,
        "MSE_delta": mse_d,
        "RMSE_delta": rmse_d,
        "MAE_delta": mae_d
    }


model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

res = evaluate_on_loader_separate(model, test_loader, y_mean, y_std, n_bus, device)
res["file"] = test_book

print("\nTest file:", test_book)
print(res)

df = pd.DataFrame([res])
df.to_csv(summary_csv, index=False)
print("Saved test summary to:", summary_csv)