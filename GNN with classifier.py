import os
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import NNConv, global_mean_pool

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix


SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

n_bus = 118
batch_size = 16
lr = 1e-3
weight_decay = 1e-6
max_epochs = 100
patience = 20

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

data_dir = r"C:\Users\mahao\gnn_case118_dataset"

train_xlsx = os.path.join(data_dir, "train.xlsx")
val_xlsx   = os.path.join(data_dir, "val.xlsx")
test_xlsx  = os.path.join(data_dir, "test.xlsx")

train_dispatch_xlsx = os.path.join(data_dir, "train_dispatch.xlsx")
val_dispatch_xlsx   = os.path.join(data_dir, "val_dispatch.xlsx")
test_dispatch_xlsx  = os.path.join(data_dir, "test_dispatch.xlsx")

train_edge_xlsx = os.path.join(data_dir, "train_edge_dynamic.xlsx")
val_edge_xlsx   = os.path.join(data_dir, "val_edge_dynamic.xlsx")
test_edge_xlsx  = os.path.join(data_dir, "test_edge_dynamic.xlsx")

train_label_csv = os.path.join(data_dir, "train_labels.csv")
val_label_csv   = os.path.join(data_dir, "val_labels.csv")
test_label_csv  = os.path.join(data_dir, "test_labels.csv")

branch_csv = os.path.join(data_dir, "case118_branch_data.csv")
cont_train_csv = os.path.join(data_dir, "train_contingency_info.csv")
cont_val_csv   = os.path.join(data_dir, "val_contingency_info.csv")
cont_test_csv  = os.path.join(data_dir, "test_contingency_info.csv")

model_path = "[118bus]_NNConv_Classifier.pt"
stats_path = "[118bus]_NNConv_Classifier_stats.pt"


def read_excel(path):
    return pd.read_excel(path).values


def read_csv(path):
    return pd.read_csv(path)


def load_branch_data(branch_csv):
    df = pd.read_csv(branch_csv)
    from_bus = (df["from_bus"].values.astype(int) - 1).tolist()
    to_bus   = (df["to_bus"].values.astype(int) - 1).tolist()
    static_edge = df[["r", "x", "b", "rateA", "ratio"]].values.astype(np.float32)
    return from_bus, to_bus, static_edge


def load_contingency_info(cont_csv):
    df = pd.read_csv(cont_csv)
    f_out = (df["outage_from"].values.astype(int) - 1).tolist()
    t_out = (df["outage_to"].values.astype(int) - 1).tolist()
    return f_out, t_out


def split_node_features(main_np, dispatch_np, n_bus):
    x_list = []
    for i in range(len(main_np)):
        x_sample = []
        for n in range(n_bus):
            Pload = main_np[i, 4*n + 1]
            Qload = main_np[i, 4*n + 2]
            Pgen  = dispatch_np[i, 2*n + 1]
            Qgen  = dispatch_np[i, 2*n + 2]

            if n == 0:
                is_slack, is_pv, is_pq = 1, 0, 0
            elif Pgen > 1e-6:
                is_slack, is_pv, is_pq = 0, 1, 0
            else:
                is_slack, is_pv, is_pq = 0, 0, 1

            x_sample.append([Pload, Qload, Pgen, Qgen, is_pv, is_pq, is_slack])

        x_list.append(x_sample)

    return torch.tensor(x_list, dtype=torch.float32)


def fit_x_normalization(x_train):
    x_mean = x_train.mean(dim=0)
    x_std  = x_train.std(dim=0)
    x_std[x_std == 0] = 1.0
    return x_mean, x_std


def apply_x_normalization(x, x_mean, x_std):
    x_norm = (x - x_mean) / x_std
    x_norm[:, :, 4:] = x[:, :, 4:]
    return x_norm


def build_dynamic_classification_dataset(
    x_tensor,
    edge_dynamic_np,
    label_df,
    from_bus_orig,
    to_bus_orig,
    static_edge_attr,
    from_bus_cont,
    to_bus_cont
):
    data_list = []
    n_samples = x_tensor.size(0)

    for i in range(n_samples):
        x = x_tensor[i]

        f_out = from_bus_cont[i]
        t_out = to_bus_cont[i]

        # dynamic edge feature row: [sample_id, loading_1, ..., loading_n]
        line_loading = edge_dynamic_np[i, 1:].astype(np.float32)

        filtered_edges = []
        filtered_attr = []

        for line_idx, (f, t, static_attr) in enumerate(zip(from_bus_orig, to_bus_orig, static_edge_attr)):
            if not ((f == f_out and t == t_out) or (f == t_out and t == f_out)):
                dyn_loading = line_loading[line_idx]
                edge_feat = np.concatenate([static_attr, [dyn_loading]], axis=0)
                filtered_edges.append((f, t))
                filtered_attr.append(edge_feat)

        from_temp = [f for f, t in filtered_edges]
        to_temp   = [t for f, t in filtered_edges]

        edge_index = torch.tensor(
            [from_temp + to_temp, to_temp + from_temp],
            dtype=torch.long
        )

        edge_attr = torch.tensor(
            np.vstack([filtered_attr, filtered_attr]),
            dtype=torch.float32
        )

        y_cls = int(label_df.loc[i, "class_id"])

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.tensor(y_cls, dtype=torch.long)
        )
        data_list.append(data)

    return data_list


class EdgeClassifier(nn.Module):
    def __init__(self, node_in=7, edge_in=6, hidden=64, nclass=4):
        super().__init__()

        nn1 = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.ReLU(),
            nn.Linear(64, node_in * hidden)
        )
        self.conv1 = NNConv(node_in, hidden, nn1, aggr="mean")

        nn2 = nn.Sequential(
            nn.Linear(edge_in, 64),
            nn.ReLU(),
            nn.Linear(64, hidden * hidden)
        )
        self.conv2 = NNConv(hidden, hidden, nn2, aggr="mean")

        self.lin1 = nn.Linear(hidden, 64)
        self.lin2 = nn.Linear(64, nclass)
        self.dropout = nn.Dropout(0.2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.dropout(x)

        g = global_mean_pool(x, batch)
        g = F.relu(self.lin1(g))
        g = self.dropout(g)
        out = self.lin2(g)
        return out


# -----------------------------
# Load train/val/test data
# -----------------------------
from_bus_orig, to_bus_orig, static_edge_attr = load_branch_data(branch_csv)

train_main = read_excel(train_xlsx)
val_main   = read_excel(val_xlsx)
test_main  = read_excel(test_xlsx)

train_dispatch = read_excel(train_dispatch_xlsx)
val_dispatch   = read_excel(val_dispatch_xlsx)
test_dispatch  = read_excel(test_dispatch_xlsx)

train_edge = read_excel(train_edge_xlsx)
val_edge   = read_excel(val_edge_xlsx)
test_edge  = read_excel(test_edge_xlsx)

train_label = read_csv(train_label_csv)
val_label   = read_csv(val_label_csv)
test_label  = read_csv(test_label_csv)

ftr, ttr = load_contingency_info(cont_train_csv)
fva, tva = load_contingency_info(cont_val_csv)
fte, tte = load_contingency_info(cont_test_csv)

x_train_raw = split_node_features(train_main, train_dispatch, n_bus)
x_val_raw   = split_node_features(val_main, val_dispatch, n_bus)
x_test_raw  = split_node_features(test_main, test_dispatch, n_bus)

x_mean, x_std = fit_x_normalization(x_train_raw)
x_train = apply_x_normalization(x_train_raw, x_mean, x_std)
x_val   = apply_x_normalization(x_val_raw, x_mean, x_std)
x_test  = apply_x_normalization(x_test_raw, x_mean, x_std)

torch.save({"x_mean": x_mean, "x_std": x_std}, stats_path)

train_list = build_dynamic_classification_dataset(
    x_train, train_edge, train_label,
    from_bus_orig, to_bus_orig, static_edge_attr,
    ftr, ttr
)

val_list = build_dynamic_classification_dataset(
    x_val, val_edge, val_label,
    from_bus_orig, to_bus_orig, static_edge_attr,
    fva, tva
)

test_list = build_dynamic_classification_dataset(
    x_test, test_edge, test_label,
    from_bus_orig, to_bus_orig, static_edge_attr,
    fte, tte
)

train_loader = DataLoader(train_list, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_list, batch_size=batch_size, shuffle=False)
test_loader  = DataLoader(test_list, batch_size=batch_size, shuffle=False)

model = EdgeClassifier(node_in=7, edge_in=6, hidden=64, nclass=4).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

best_val = 1e9
pat_count = 0

for epoch in range(1, max_epochs + 1):
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = F.cross_entropy(logits, batch.y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch.num_graphs

    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss = 0.0
    val_true, val_pred = [], []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            logits = model(batch)
            loss = F.cross_entropy(logits, batch.y)
            val_loss += loss.item() * batch.num_graphs

            pred = torch.argmax(logits, dim=1)
            val_true.extend(batch.y.cpu().numpy())
            val_pred.extend(pred.cpu().numpy())

    val_loss /= len(val_loader.dataset)
    val_acc = accuracy_score(val_true, val_pred)

    if val_loss < best_val:
        best_val = val_loss
        pat_count = 0
        torch.save(model.state_dict(), model_path)
    else:
        pat_count += 1

    if epoch == 1 or epoch % 10 == 0:
        print(f"Epoch {epoch:4d} | TrainLoss {train_loss:.6e} | ValLoss {val_loss:.6e} | ValAcc {val_acc:.4f}")

    if pat_count >= patience:
        print(f"Early stopping at epoch {epoch}")
        break

# -----------------------------
# Test
# -----------------------------
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        logits = model(batch)
        pred = torch.argmax(logits, dim=1)
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(pred.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
f1m = f1_score(y_true, y_pred, average="macro")

print("\nTest Accuracy:", acc)
print("Test Macro-F1:", f1m)
print("\nClassification Report:")
print(classification_report(y_true, y_pred, digits=4))
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))