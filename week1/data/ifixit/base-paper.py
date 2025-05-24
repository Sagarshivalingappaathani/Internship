import json
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
import networkx as nx
import spacy
import pytextrank

# === Load spaCy with TextRank ===
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

# === Load Dataset ===
with open("Appliance.json", "r") as f:
    documents = [json.loads(line) for line in f if line.strip()]
pairs_df = pd.read_csv("labeled_pairs_rule_based.csv")

# === Extract steps ===
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

# === Build Global TF-IDF Model ===
all_sentences = []
for doc in documents:
    all_sentences.extend(extract_steps(doc))
global_tfidf = TfidfVectorizer().fit(all_sentences)

# === Extract keywords using TextRank ===
def extract_keywords(text, top_k=15):
    doc = nlp(text)
    return [phrase.text for phrase in doc._.phrases[:top_k]]

# === Build keyword graph ===
def build_keyword_graph(sentences):
    G = nx.Graph()
    for sent in sentences:
        keywords = extract_keywords(sent)
        for i in range(len(keywords)):
            for j in range(i + 1, len(keywords)):
                G.add_edge(keywords[i], keywords[j])
    return G

# === Community Detection for Concepts ===
def detect_communities(G):
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    return [list(c) for c in communities]

# === Assign sentences to concept clusters using TF-IDF ===
def assign_sentences_to_concepts(sentences, concepts, tfidf_model):
    concept_vectors = []
    for concept in concepts:
        joined = " ".join(concept)
        vec = tfidf_model.transform([joined])
        concept_vectors.append(vec)

    node_texts = [[] for _ in concepts]
    for sent in sentences:
        sent_vec = tfidf_model.transform([sent])
        best_match = -1
        best_sim = 0.0
        for idx, concept_vec in enumerate(concept_vectors):
            sim = cosine_similarity(sent_vec, concept_vec)[0][0]
            if sim > 0.2 and sim > best_sim:
                best_sim = sim
                best_match = idx
        if best_match != -1:
            node_texts[best_match].append(sent)

    return node_texts

# === Create JCIG with Direction ===
def create_keyword_jcig(doc1_id, doc2_id, label):
    doc1_steps = [s for s in extract_steps(documents[doc1_id])]
    doc2_steps = [s for s in extract_steps(documents[doc2_id])]
    sentences = doc1_steps + doc2_steps
    if not sentences:
        return None

    G = build_keyword_graph(sentences)
    communities = detect_communities(G)

    concept_sentences = assign_sentences_to_concepts(sentences, communities, global_tfidf)
    concept_vectors = [global_tfidf.transform([" ".join(sents)]) for sents in concept_sentences if sents]
    node_features = np.vstack([v.toarray() for v in concept_vectors])

    sim_matrix = cosine_similarity(node_features)
    edge_index = []
    edge_weight = []
    top_k = 3  # add top-3 strongest outgoing edges per node

    for i in range(len(node_features)):
        sims = list(enumerate(sim_matrix[i]))
        sims = sorted([(j, s) for j, s in sims if j != i], key=lambda x: -x[1])
        for j, score in sims[:top_k]:
            edge_index.append([i, j])  # i → j
            edge_weight.append(score)

    if not edge_index:
        return None

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float32)
    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor([float(label)], dtype=torch.float32)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_weight, y=y)

# === Dataset creation ===
jcig_data = []
for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    g = create_keyword_jcig(row.doc1_id, row.doc2_id, row.label)
    if g:
        jcig_data.append(g)

# === 60-20-20 Split ===
train_size = int(0.6 * len(jcig_data))
val_size = int(0.2 * len(jcig_data))
test_size = len(jcig_data) - train_size - val_size
train_data, val_data, test_data = torch.utils.data.random_split(jcig_data, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
val_loader = DataLoader(val_data, batch_size=4)
test_loader = DataLoader(test_data, batch_size=4)

# === GCN Model ===
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

# === Training and Evaluation ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
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

# === Evaluation ===
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
        preds = torch.sigmoid(out) > 0.5
        y_true.extend(batch.y.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\n✅ Directional JCIG Accuracy: {acc * 100:.2f}%")
