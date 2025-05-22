import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm

# === Load dataset ===
with open("Appliance.json", "r") as f:
    documents = [json.loads(line) for line in f if line.strip()]

pairs_df = pd.read_csv("labeled_pairs_rule_based.csv")

# === Extract cleaned steps ===
def extract_steps(doc):
    steps = []
    for step in doc.get("Steps", []):
        if "Text_raw" in step:
            steps.append(step["Text_raw"].strip())
        elif "Lines" in step:
            text_lines = [line.get("Text", "") for line in step["Lines"] if "Text" in line]
            combined = " ".join(text_lines).strip()
            if combined:
                steps.append(combined)
    return [s for s in steps if len(s.split()) > 2]

# === Build a shared TF-IDF vectorizer ===
all_sentences = []
for doc in documents:
    steps = extract_steps(doc)
    if steps:
        all_sentences.extend(steps)

if not all_sentences:
    raise ValueError("No valid steps extracted from any document — check Appliance.json formatting.")

print(f"Total valid step sentences extracted: {len(all_sentences)}")

vectorizer = TfidfVectorizer(stop_words='english', min_df=1, token_pattern=r"(?u)\b\w+\b")
vectorizer.fit(all_sentences)

# === Create PyG Data object from doc pair ===
def create_graph_from_pair(doc1_id, doc2_id, label, vectorizer):
    doc1_steps = extract_steps(documents[doc1_id])
    doc2_steps = extract_steps(documents[doc2_id])
    sentences = doc1_steps + doc2_steps
    if not sentences:
        return None

    tfidf = vectorizer.transform(sentences).toarray()
    sim_matrix = cosine_similarity(tfidf)

    nodes = [f"doc1_s{i}" for i in range(len(doc1_steps))] + [f"doc2_s{i}" for i in range(len(doc2_steps))]

    edge_index = []
    edge_weight = []

    for i in range(len(nodes)):
        sims = list(enumerate(sim_matrix[i]))
        sims = sorted([(j, score) for j, score in sims if j != i], key=lambda x: -x[1])
        if sims:
            j, weight = sims[0]
            edge_index.append([i, j])
            edge_weight.append(weight)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor(tfidf, dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

# === Create dataset ===
graph_data = []
for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    g = create_graph_from_pair(row.doc1_id, row.doc2_id, row.label, vectorizer)
    if g:
        graph_data.append(g)

# === Split ===
train_data, test_data = train_test_split(graph_data, test_size=0.2, random_state=42)
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=16)

# === Define GCN ===
class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
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

# === Train and evaluate ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(input_dim=train_data[0].x.shape[1], hidden_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.BCEWithLogitsLoss()

for epoch in range(1, 100):
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
y_true = []
y_pred = []

with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
        preds = torch.sigmoid(out) > 0.5
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\n Model Accuracy on Test Set: {acc * 100:.2f}%")