# jcig_pipeline.py
import json, spacy, pytextrank, networkx as nx, numpy as np, pandas as pd, torch
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import StratifiedKFold
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score
import numpy as np

# ─────────────────────────────── 1. DATA ──────────────────────────────────────
with open("Appliance.json", "r") as f:
    documents = [json.loads(line) for line in f if line.strip()]
pairs_df = pd.read_csv("labeled_pairs_rule_based.csv")

# ─────────────────────── 2. TEXT PROCESSING (SpaCy + TextRank) ───────────────
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract_steps(doc):
    steps = []
    for step in doc.get("Steps", []):
        if "Text_raw" in step:
            steps.append(step["Text_raw"].strip())
        elif "Lines" in step:
            txt = " ".join([l["Text"] for l in step["Lines"] if "Text" in l]).strip()
            if txt: steps.append(txt)
    return [s for s in steps if len(s.split()) > 2]

def extract_keywords(text, k=15):
    return [p.text.lower().strip() for p in nlp(text)._.phrases[:k]]

# ───────────────────────── 3. GLOBAL TF-IDF ───────────────────────────────────
all_sents = [s for d in documents for s in extract_steps(d)]
global_tfidf = TfidfVectorizer().fit(all_sents)

# ─────────────────────────── 4. SBERT WRAPPER ────────────────────────────────
class SBERT:
    def __init__(self, name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tok = AutoTokenizer.from_pretrained(name)
        self.mod = AutoModel.from_pretrained(name)
    def encode(self, sentences):
        if not sentences: return np.zeros((1,384))
        ip = self.tok(sentences, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad(): out = self.mod(**ip).last_hidden_state
        att = ip['attention_mask'].unsqueeze(-1).expand(out.size()).float()
        vec = (out*att).sum(1)/torch.clamp(att.sum(1), min=1e-9)
        return vec.cpu().numpy()
sbert = SBERT()

# ───────────────────────── 5. GRAPH HELPERS ───────────────────────────────────
def keyword_graph(sentences):
    G = nx.Graph()
    for s in sentences:
        kws = extract_keywords(s)
        for i in range(len(kws)):
            for j in range(i+1,len(kws)): G.add_edge(kws[i], kws[j])
    return G

def louvain_concepts(G):               # greedy_modularity ≈ Louvain for small graphs
    from networkx.algorithms.community import greedy_modularity_communities
    return [list(c) for c in greedy_modularity_communities(G)]

def sent2concept(sentences, concepts, thresh=0.2):
    c_vec = [global_tfidf.transform([" ".join(c)]) for c in concepts]
    s2c, c2s_A, c2s_B = defaultdict(list), defaultdict(list), defaultdict(list)
    for idx,s in enumerate(sentences):
        sv = global_tfidf.transform([s])
        for cid,cv in enumerate(c_vec):
            if cosine_similarity(sv,cv)[0,0] >= thresh:
                s2c[idx].append(cid)
                (c2s_A if idx<len(sentences)//2 else c2s_B)[cid].append(s)
        if not s2c[idx]: s2c[idx]=[-1]
    return s2c, c2s_A, c2s_B          # all concepts present even if empty

def pseudograph(sentences, s2c):
    P = nx.MultiDiGraph()
    for i in range(len(sentences)-1):
        for a in s2c[i]:
            for b in s2c[i+1]:
                if a!=-1 and b!=-1 and a!=b: P.add_edge(a,b)
    return P

def hp_direction_graph(pseudo: nx.MultiDiGraph, node_emb: np.ndarray) -> nx.DiGraph:
    G_dir = nx.DiGraph()
    fwd_cnt = defaultdict(int)

    # Count directed edges (ignore the key)
    for u, v, _ in pseudo.edges(keys=True):
        if u != v:
            fwd_cnt[(u, v)] += 1

    # For each observed ordered pair, compare to reverse
    for (u, v), cnt_uv in fwd_cnt.items():
        cnt_vu = fwd_cnt.get((v, u), 0)
        # cosine sims
        sim_uv = cosine_similarity([node_emb[u]], [node_emb[v]])[0, 0]
        sim_vu = cosine_similarity([node_emb[v]], [node_emb[u]])[0, 0]
        if cnt_uv > cnt_vu:
            G_dir.add_edge(u, v, weight=sim_uv)
        elif cnt_vu > cnt_uv:
            G_dir.add_edge(v, u, weight=sim_vu)
        else:
            # tie: pick direction with higher sim
            if sim_uv >= sim_vu:
                G_dir.add_edge(u, v, weight=sim_uv)
            else:
                G_dir.add_edge(v, u, weight=sim_vu)

    return G_dir


def match_vectors(num_concepts, c2s_A, c2s_B):
    vecs=[]
    for cid in range(num_concepts):
        vA = sbert.encode(c2s_A.get(cid,[])).mean(0)
        vB = sbert.encode(c2s_B.get(cid,[])).mean(0)
        vecs.append(np.concatenate([np.abs(vA-vB), vA*vB]))  # 768-dim
    return np.stack(vecs)                                   # (C,768)

def pyg_graph(G_dir, node_feat, label):
    if G_dir.number_of_edges()==0: return None
    ei = torch.tensor(list(G_dir.edges)).t().long()
    ew = torch.tensor([G_dir[u][v]['weight'] for u,v in G_dir.edges()], dtype=torch.float32)
    x  = torch.tensor(node_feat, dtype=torch.float32)
    y  = torch.tensor([float(label)], dtype=torch.float32)
    return Data(x=x, edge_index=ei, edge_attr=ew, y=y)

# ─────────────────────── 6. BUILD ALL JCIG GRAPHS ────────────────────────────
graphs = []
labels = []
for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    sA, sB = extract_steps(documents[row.doc1_id]), extract_steps(documents[row.doc2_id])
    sents  = sA + sB
    if len(sents) < 2:
        continue

    C = louvain_concepts(keyword_graph(sents))
    s2c, c2sA, c2sB = sent2concept(sents, C)
    node_feat       = match_vectors(len(C), c2sA, c2sB)
    G_dir           = hp_direction_graph(pseudograph(sents, s2c), node_feat)

    g = pyg_graph(G_dir, node_feat, row.label)
    if g is not None:
        graphs.append(g)
        labels.append(int(row.label))    # <<–– collect labels here

print(f"✔ Built {len(graphs)} directional JCIG graphs")

# ─────────────────────── 7. MODEL DEFINITION ────────────────────────────────
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class JCIGClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.mlp  = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gcn1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = self.gcn2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        return self.mlp(x)


# ─────────────────────────── 8. 5-FOLD CROSS-VALIDATION ──────────────────────────
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_fold_acc = []

for fold, (train_idx, test_idx) in enumerate(kf.split(graphs, labels), 1):
    # Split out train+val vs test
    train_val_graphs = [graphs[i] for i in train_idx]
    train_val_labels = [labels[i] for i in train_idx]
    test_graphs      = [graphs[i] for i in test_idx]
    test_labels      = [labels[i] for i in test_idx]

    # Further carve out a 20% validation from train_val
    val_size = int(0.2 * len(train_val_graphs))
    train_graphs, val_graphs = train_val_graphs[val_size:], train_val_graphs[:val_size]
    train_labels, val_labels = train_val_labels[val_size:], train_val_labels[:val_size]

    train_loader = DataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader   = DataLoader(val_graphs,   batch_size=4)
    test_loader  = DataLoader(test_graphs,  batch_size=4)

    # Initialize fresh model
    model = JCIGClassifier(input_dim=train_graphs[0].x.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # Train for 50 epochs (or early stop on val)
    best_val_acc = 0.0
    for epoch in range(1, 51):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            pred  = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            loss  = loss_fn(pred, batch.y)
            loss.backward(); optimizer.step(); optimizer.zero_grad()

        # validate
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
                y_true.extend(batch.y.cpu().numpy())
                y_pred.extend((torch.sigmoid(out) > 0.5).cpu().numpy())
        val_acc = accuracy_score(y_true, y_pred)
        best_val_acc = max(best_val_acc, val_acc)

    # finally test
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_attr, batch.batch).view(-1)
            y_true.extend(batch.y.cpu().numpy())
            y_pred.extend((torch.sigmoid(out) > 0.5).cpu().numpy())
    test_acc = accuracy_score(y_true, y_pred)
    print(f"Fold {fold} — val_acc: {best_val_acc:.3f} — test_acc: {test_acc:.3f}")
    all_fold_acc.append(test_acc)

print(f"\n→ Mean test accuracy over 5 folds: {np.mean(all_fold_acc):.3f} ± {np.std(all_fold_acc):.3f}")