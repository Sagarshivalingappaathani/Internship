import json
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel

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

# === Safe SBERT wrapper ===
class SBERTEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            model_output = self.model(**inputs)
        return self.mean_pooling(model_output, inputs['attention_mask']).cpu().numpy()

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

bert_model = SBERTEmbedder()

# === Create JCIG Graph with Concept Clustering and D³ via HP ===
def create_graph_from_pair(doc1_id, doc2_id, label):
    doc1_steps = extract_steps(documents[doc1_id])
    doc2_steps = extract_steps(documents[doc2_id])
    sentences = doc1_steps + doc2_steps
    if not sentences:
        return None

    embeddings = bert_model.encode(sentences)
    sim_matrix = cosine_similarity(embeddings)

    # === Concept Clustering using KMeans (simulate Louvain) ===
    num_clusters = min(5, len(sentences))
    kmeans = KMeans(n_clusters=num_clusters, n_init='auto').fit(embeddings)
    labels = kmeans.labels_

    # Group sentence indices into concept clusters
    concepts = defaultdict(list)
    for idx, cluster_id in enumerate(labels):
        concepts[cluster_id].append(idx)

    concept_nodes = list(concepts.values())
    concept_embeddings = [np.mean(embeddings[inds], axis=0) for inds in concept_nodes]

    # Build concept similarity matrix
    concept_sim = cosine_similarity(concept_embeddings)

    edge_index = []
    edge_weight = []
    num_nodes = len(concept_nodes)

    for i in range(num_nodes):
        sims = list(enumerate(concept_sim[i]))
        sims = sorted([(j, s) for j, s in sims if j != i], key=lambda x: -x[1])
        if sims:
            j, score = sims[0]  # Hamiltonian Path style
            edge_index.append([i, j])
            edge_weight.append(score)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor(np.array(concept_embeddings), dtype=torch.float)
    y = torch.tensor([label], dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

# === Create dataset ===
graph_data = []
for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    g = create_graph_from_pair(row.doc1_id, row.doc2_id, row.label)
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
print(f"\n✅ Model Accuracy on Test Set: {acc * 100:.2f}%")
