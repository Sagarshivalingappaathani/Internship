# jcig_pipeline_kfold.py
import os, json, torch, pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

# ───── import your helper “agent” classes exactly as before ──────
from jcig_agents.step_extractor import StepExtractorAgent
from jcig_agents.embed_cluster import EmbedClusterAgent
from jcig_agents.edge_builder import EdgeBuilderAgent
from jcig_agents.graph_builder import GraphBuilderAgent

# ═══════════════ 1. LOAD DATA ═══════════════════════════════════
with open("Appliance.json") as f:
    documents = [json.loads(l) for l in f if l.strip()]
pairs_df = pd.read_csv("labeled_pairs_rule_based.csv")

# ═══════════════ 2. BUILD ALL JCIG GRAPHS ONCE ═════════════════
graphs, labels = [], []
for row in tqdm(pairs_df.itertuples(), total=len(pairs_df)):
    try:
        steps    = StepExtractorAgent.run(documents[row.doc1_id],
                                          documents[row.doc2_id])
        clusters = EmbedClusterAgent.run(steps)
        edges    = EdgeBuilderAgent.run(clusters)
        g        = GraphBuilderAgent.run(clusters, edges)
        g.y      = torch.tensor([row.label], dtype=torch.float)
        graphs.append(g)
        labels.append(int(row.label))
    except Exception as e:
        print(f"⚠️  skipped pair ({row.doc1_id},{row.doc2_id}): {e}")

print(f"✔ Built {len(graphs)} graphs")

# ═══════════════ 3. DEFINE MODEL ════════════════════════════════
class GCN(torch.nn.Module):
    def __init__(self, in_dim, hid=64):
        super().__init__()
        self.c1 = GCNConv(in_dim, hid)
        self.c2 = GCNConv(hid, hid)
        self.cls = torch.nn.Linear(hid, 1)

    def forward(self, x, ei, ew, batch):
        x = F.relu(self.c1(x, ei, ew))
        x = F.relu(self.c2(x, ei, ew))
        x = global_mean_pool(x, batch)
        return self.cls(x).view(-1)

# ═══════════════ 4. K-FOLD CV ══════════════════════════════════
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
kfold      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
N_EPOCHS   = 30
BATCH_SIZE = 16
fold_acc   = []

for fold, (train_idx, test_idx) in enumerate(kfold.split(graphs, labels), 1):
    train_graphs = [graphs[i]  for i in train_idx]
    test_graphs  = [graphs[i]  for i in test_idx]

    train_loader = DataLoader(train_graphs, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_graphs,  batch_size=BATCH_SIZE)

    model = GCN(in_dim=train_graphs[0].x.shape[1]).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-2)
    lossf = torch.nn.BCEWithLogitsLoss()

    for epoch in range(1, N_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
            loss  = lossf(pred, batch.y)
            loss.backward(); opt.step(); opt.zero_grad()

    # ── evaluate on held-out test fold ─────────────────────
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out   = torch.sigmoid(model(batch.x, batch.edge_index,
                                        batch.edge_attr, batch.batch))
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend((out > 0.5).cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    fold_acc.append(acc)
    print(f"Fold {fold}:  test-acc = {acc*100:.2f}%")

# ═══════════════ 5. SUMMARY ════════════════════════════════════
print(f"\n→ 5-fold mean test accuracy: "
      f"{np.mean(fold_acc)*100:.2f}%  ±  {np.std(fold_acc)*100:.2f}%")
