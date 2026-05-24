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

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ============================================================
# Config
# ============================================================
n_bus = 118
batch_size = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_book = r"C:\Users\haoyu.ma\OneDrive - Washington State University (email.wsu.edu)\Research\Datasetgeneration\case118_random_training_dataset.xlsx"

# Change folder if needed
data_dir = r"C:\Users\haoyu.ma\OneDrive - Washington State University (email.wsu.edu)\Research\Datasetgeneration"

# ============================================================
# Cases to evaluate
# ============================================================
cases = [
    {
        "label": "8cases",
        "train_book": os.path.join(data_dir, "case118_scenario_acopf_dataset_8cases.xlsx"),
        "model_path": f"[{n_bus}bus]_Best_GCN_static_8cases.pt",
        "stats_path": f"[{n_bus}bus]_NormStats_static_8cases.pt",
    },
    {
        "label": "27cases",
        "train_book": os.path.join(data_dir, "case118_scenario_acopf_dataset_27cases.xlsx"),
        "model_path": f"[{n_bus}bus]_Best_GCN_static_27cases.pt",
        "stats_path": f"[{n_bus}bus]_NormStats_static_27cases.pt",
    },
    {
        "label": "64mixcases",
        "train_book": os.path.join(data_dir, "case118_scenario_acopf_dataset_64mixcases.xlsx"),
        "model_path": f"[{n_bus}bus]_Best_GCN_static_64mixcases.pt",
        "stats_path": f"[{n_bus}bus]_NormStats_static_64mixcases.pt",
    },
    {
        "label": "125mixcases",
        "train_book": os.path.join(data_dir, "case118_scenario_acopf_dataset_125mixcases.xlsx"),
        "model_path": f"[{n_bus}bus]_Best_GCN_static_125mixcases.pt",
        "stats_path": f"[{n_bus}bus]_NormStats_static_125mixcases.pt",
    },
    {
        "label": "216cases",
        "train_book": os.path.join(data_dir, "case118_scenario_acopf_dataset_216cases.xlsx"),
        "model_path": f"[{n_bus}bus]_Best_GCN_static_216cases.pt",
        "stats_path": f"[{n_bus}bus]_NormStats_static_216cases.pt",
    },
    {
        "label": "343cases",
        "train_book": os.path.join(data_dir, "case118_scenario_acopf_dataset_343cases.xlsx"),
        "model_path": f"[{n_bus}bus]_Best_GCN_static_343cases.pt",
        "stats_path": f"[{n_bus}bus]_NormStats_static_343cases.pt",
    },
]

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
            Pgen = disp_np[i, 2*g + 1]
            Qgen = disp_np[i, 2*g + 2]
            Pgen_bus[bus_idx] += Pgen
            Qgen_bus[bus_idx] += Qgen

        for n in range(n_bus):
            Pload = main_np[i, 4*n + 1]
            Qload = main_np[i, 4*n + 2]
            V     = main_np[i, 4*n + 3]
            d     = main_np[i, 4*n + 4]

            x_sample.append([
                Pload, Qload, Pgen_bus[n], Qgen_bus[n],
                is_pv_vec[n], is_pq_vec[n], is_slack_vec[n]
            ])
            y_sample.append([V, d])

        x_list.append(x_sample)
        y_list.append(y_sample)

    x = torch.tensor(x_list, dtype=torch.float32)
    y = torch.tensor(y_list, dtype=torch.float32)
    return x, y

def apply_normalization(x, y, x_mean, x_std, y_mean, y_std):
    x_norm = (x - x_mean) / x_std
    y_norm = (y - y_mean) / y_std
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
# Generator bus order
# ============================================================
gen_bus_vec = [
    1, 4, 6, 8, 10, 12, 15, 18, 19, 24, 25, 26, 27, 31, 32, 34, 36, 40,
    42, 46, 49, 54, 55, 56, 59, 61, 62, 65, 66, 69, 70, 72, 73, 74, 76,
    77, 80, 85, 87, 89, 90, 91, 92, 99, 100, 103, 104, 105, 107, 110, 111,
    112, 113, 116
]

# ============================================================
# Evaluation function
# ============================================================
def evaluate_case(case_info):
    label = case_info["label"]
    train_book = case_info["train_book"]
    model_path = case_info["model_path"]
    stats_path = case_info["stats_path"]

    print(f"\n========== Evaluating {label} ==========")

    # metadata from training workbook
    from_bus_orig, to_bus_orig, edge_attr_orig = load_branch_data_from_excel(train_book)
    is_slack_vec, is_pv_vec, is_pq_vec = load_bus_type_from_excel(train_book)

    # load saved normalization stats
    stats = torch.load(stats_path, map_location=device)
    x_mean = stats["x_mean"]
    x_std  = stats["x_std"]
    y_mean = stats["y_mean"]
    y_std  = stats["y_std"]

    # load test set
    test_main_df, test_disp_df = read_dataset_from_workbook(test_book)
    x_test_raw, y_test_raw = split_features_targets_with_genbus(
        test_main_df, test_disp_df, n_bus,
        is_slack_vec, is_pv_vec, is_pq_vec, gen_bus_vec
    )

    x_test, y_test = apply_normalization(x_test_raw, y_test_raw, x_mean, x_std, y_mean, y_std)
    test_list = build_static_dataset(x_test, y_test, from_bus_orig, to_bus_orig, edge_attr_orig)
    test_loader = DataLoader(test_list, batch_size=batch_size, shuffle=False)

    # load model
    model = EdgeGCN(in_channels=7, hidden_channels=64, out_channels=2, dropout=0.1).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    preds_denorm = []
    trues_denorm = []

    with torch.no_grad():
        for batch in test_loader:
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

    # total metrics
    mse   = mean_squared_error(trues_denorm, preds_denorm)
    rmse  = np.sqrt(mse)
    mae   = mean_absolute_error(trues_denorm, preds_denorm)
    r2    = r2_score(trues_denorm, preds_denorm)

    # V metrics
    mse_v  = mean_squared_error(trues_denorm[:, 0], preds_denorm[:, 0])
    rmse_v = np.sqrt(mse_v)
    mae_v  = mean_absolute_error(trues_denorm[:, 0], preds_denorm[:, 0])

    # Delta metrics
    mse_d  = mean_squared_error(trues_denorm[:, 1], preds_denorm[:, 1])
    rmse_d = np.sqrt(mse_d)
    mae_d  = mean_absolute_error(trues_denorm[:, 1], preds_denorm[:, 1])

    # max absolute errors
    pred_3d = preds_denorm.reshape(-1, n_bus, 2)
    true_3d = trues_denorm.reshape(-1, n_bus, 2)
    abs_err = np.abs(pred_3d - true_3d)

    max_abs_v = abs_err[:, :, 0].max()
    max_abs_d = abs_err[:, :, 1].max()

    return {
        "case": label,
        "MSE_total": mse,
        "RMSE_total": rmse,
        "MAE_total": mae,
        "R2": r2,
        "MSE_V": mse_v,
        "RMSE_V": rmse_v,
        "MAE_V": mae_v,
        "MSE_delta": mse_d,
        "RMSE_delta": rmse_d,
        "MAE_delta": mae_d,
        "MaxAbs_V": max_abs_v,
        "MaxAbs_delta": max_abs_d
    }

# ============================================================
# Run all cases
# ============================================================
results = []
for case in cases:
    results.append(evaluate_case(case))

results_df = pd.DataFrame(results)
print("\nFinal summary:")
print(results_df)

results_df.to_csv("AllCases_ErrorSummary.csv", index=False)
print("\nSaved to AllCases_ErrorSummary.csv")





labels = results_df["case"].tolist()
n_cases = len(labels)

# ============================================================
# Figure 1: Max absolute V error
# ============================================================
plt.figure(figsize=(10, 5), dpi=150)
plt.bar(labels, results_df["MaxAbs_V"])
plt.ylabel("Max absolute error (p.u.)")
plt.title("Maximum Absolute Voltage Magnitude Error")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# ============================================================
# Figure 2: Max absolute delta error
# ============================================================
plt.figure(figsize=(10, 5), dpi=150)
plt.bar(labels, results_df["MaxAbs_delta"])
plt.ylabel("Max absolute error (deg)")
plt.title("Maximum Absolute Voltage Angle Error")
plt.grid(True, axis="y")
plt.tight_layout()
plt.show()

# ============================================================
# Helper for grouped bar figures like your screenshot
# x-axis = metric names
# legend = cases
# ============================================================
def grouped_metric_plot(metric_names, data_matrix, case_labels, title, ylabel):
    """
    metric_names: list like ['MSE', 'RMSE', 'MAE']
    data_matrix : shape (n_cases, n_metrics)
    """
    x = np.arange(len(metric_names))
    width = 0.8 / len(case_labels)

    plt.figure(figsize=(10, 5), dpi=150)

    for i, case in enumerate(case_labels):
        plt.bar(
            x - 0.4 + width/2 + i * width,
            data_matrix[i],
            width=width,
            label=case
        )

    plt.xticks(x, metric_names)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.show()

# ============================================================
# Figure 3: Total error metrics
# ============================================================
total_metric_names = ["MSE_total", "RMSE_total", "MAE_total"]
total_data = results_df[total_metric_names].values
grouped_metric_plot(
    metric_names=total_metric_names,
    data_matrix=total_data,
    case_labels=labels,
    title="Total Error Metrics",
    ylabel="Metric value"
)

# ============================================================
# Figure 4: Voltage magnitude error metrics
# ============================================================
v_metric_names = ["MSE_V", "RMSE_V", "MAE_V"]
v_data = results_df[v_metric_names].values
grouped_metric_plot(
    metric_names=v_metric_names,
    data_matrix=v_data,
    case_labels=labels,
    title="Voltage Magnitude Error Metrics",
    ylabel="Metric value"
)

# ============================================================
# Figure 5: Voltage angle error metrics
# ============================================================
d_metric_names = ["MSE_delta", "RMSE_delta", "MAE_delta"]
d_data = results_df[d_metric_names].values
grouped_metric_plot(
    metric_names=d_metric_names,
    data_matrix=d_data,
    case_labels=labels,
    title="Voltage Angle Error Metrics",
    ylabel="Metric value"
)