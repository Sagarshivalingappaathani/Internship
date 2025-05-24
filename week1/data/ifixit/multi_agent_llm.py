#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool, global_add_pool

# ─── Model with enriched pooling + stats ───────────────────────────────────────
class JCIG_GNN(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=128, mlp_hidden=64, dropout=0.5):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        # after pooling we have hidden_dim * 3, plus 3 graph_stats features
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3 + 3, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        stats = data.graph_stats  # shape [batch_size, 3]
        # GCN layers
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv3(x, edge_index))
        # pooling
        m1 = global_mean_pool(x, batch)
        m2 = global_max_pool(x, batch)
        m3 = global_add_pool(x, batch)
        h = torch.cat([m1, m2, m3, stats.to(x.dtype)], dim=1)
        return self.mlp(h).squeeze()

# ─── Utilities ────────────────────────────────────────────────────────────────
def load_graphs(cache_dir="jcig_graph_cache"):
    graphs = []
    for fname in sorted(os.listdir(cache_dir)):
        if fname.endswith(".pt"):
            path = os.path.join(cache_dir, fname)
            graphs.append(torch.load(path))
    return graphs

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch)
        loss = criterion(logits, batch.y.float())
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
    return total_loss / len(loader.dataset)

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch))
            preds = (probs > 0.5).long()
            correct += (preds == batch.y.long()).sum().item()
            total += batch.num_graphs
    return correct / total if total else 0

# ─── Main training loop with early stopping ──────────────────────────────────
def main():
    # hyperparams
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LR = 5e-4
    WD = 5e-4
    DROPOUT = 0.5
    BATCH_SIZE = 16
    MAX_EPOCHS = 100
    PATIENCE = 10  # early stop if no val improvement

    # load
    graphs = load_graphs()
    labels = [g.y.item() for g in graphs]
    print(f"Loaded {len(graphs)} graphs.")

    # stratified 70/15/15 split
    g_train, g_temp, y_train, y_temp = train_test_split(
        graphs, labels, test_size=0.3, stratify=labels, random_state=42
    )
    g_val, g_test, y_val, y_test = train_test_split(
        g_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )
    print(f"Train: {len(g_train)}, Val: {len(g_val)}, Test: {len(g_test)}")

    train_loader = DataLoader(g_train, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(g_val,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(g_test,  batch_size=BATCH_SIZE)

    # model, optimizer, loss
    model = JCIG_GNN(in_dim=graphs[0].x.size(1), hidden_dim=128, mlp_hidden=64, dropout=DROPOUT).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    criterion = nn.BCEWithLogitsLoss()

    best_val = 0.0
    wait = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        val_acc = evaluate(model, val_loader, DEVICE)
        print(f"Epoch {epoch:3d} — loss: {loss:.4f} | val acc: {val_acc:.4f}", end="")

        # early stopping
        if val_acc > best_val:
            best_val = val_acc
            wait = 0
            torch.save(model.state_dict(), "best_jcig_model.pth")
            print("  ← new best")
        else:
            wait += 1
            print(f"  ({wait}/{PATIENCE})")
            if wait >= PATIENCE:
                print("Early stopping.")
                break

    # load best and test
    model.load_state_dict(torch.load("best_jcig_model.pth"))
    test_acc = evaluate(model, test_loader, DEVICE)
    print(f"\n🏆 Final Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
