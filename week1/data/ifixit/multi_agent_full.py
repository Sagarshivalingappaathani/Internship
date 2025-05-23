import os
import json
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, global_mean_pool
import torch.nn.functional as F

from jcig_agents.step_extractor import StepExtractorAgent
from jcig_agents.embed_cluster import EmbedClusterAgent
from jcig_agents.edge_builder import EdgeBuilderAgent
from jcig_agents.graph_builder import GraphBuilderAgent

# === Load data ===
with open("Appliance.json") as f:
    documents = [json.loads(line) for line in f if line.strip()]

pairs_df = pd.read_csv("labeled_pairs_rule_based.csv")

# === Build all graphs ===
graph_data = []

for row in tqdm(pairs_df.itertuples(), total=len(pairs_df)):
    try:
        steps = StepExtractorAgent.run(documents[row.doc1_id], documents[row.doc2_id])
        clusters = EmbedClusterAgent.run(steps)
        edges = EdgeBuilderAgent.run(clusters)
        graph = GraphBuilderAgent.run(clusters, edges)
        graph.y = torch.tensor([row.label], dtype=torch.float)
        graph_data.append(graph)
    except Exception as e:
        print(f"⚠️ Skipped pair ({row.doc1_id}, {row.doc2_id}) due to error: {e}")

# === Split ===
train_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# === Define GCN ===
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.lin = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = self.lin(x)
        return x

# === Train model ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=train_data[0].x.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(1, 50):
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
        loss = loss_fn(out, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch} - Loss: {total_loss:.4f}")

# === Evaluate ===
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
        preds = torch.sigmoid(out) > 0.5
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

accuracy = accuracy_score(y_true, y_pred)
print(f"✅ Final Test Accuracy: {accuracy * 100:.2f}%")
