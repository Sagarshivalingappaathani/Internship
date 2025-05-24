import json
import spacy
import pytextrank
import networkx as nx
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# === 1. Setup TextRank + SBERT ===
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

class SBERTEmbedder:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    def encode(self, sents):
        inputs = self.tokenizer(sents, padding=True, truncation=True, return_tensors="pt")
        with torch.no_grad():
            out = self.model(**inputs)
        # mean pooling
        mask = inputs["attention_mask"].unsqueeze(-1)
        emb = out.last_hidden_state * mask
        return (emb.sum(1) / mask.sum(1)).cpu().numpy()

sbert = SBERTEmbedder()

# === 2. Load data ===
with open("Appliance.json") as f:
    documents = [json.loads(l) for l in f if l.strip()]
pairs_df = pd.read_csv("labeled_pairs_rule_based.csv")

# === 3. Utilities ===
def extract_steps(doc):
    steps = []
    for step in doc.get("Steps", []):
        if "Text_raw" in step: steps.append(step["Text_raw"].strip())
        elif "Lines" in step:
            lines = [L["Text"] for L in step["Lines"] if "Text" in L]
            txt = " ".join(lines).strip()
            if txt: steps.append(txt)
    return [s for s in steps if len(s.split())>2]

def extract_keywords(sent, top_k=15):
    doc = nlp(sent)
    return [p.text.lower() for p in doc._.phrases[:top_k]]

# Global TF–IDF over *all* sentences
all_sents = []
for d in documents: all_sents += extract_steps(d)
tfidf = TfidfVectorizer().fit(all_sents)

def build_jcig(doc1, doc2):
    # A) collect sentences
    s1, s2 = extract_steps(doc1), extract_steps(doc2)
    S = s1 + s2
    if not S: return None

    # B) build keyword co‐occurrence graph
    G = nx.Graph()
    for sent in S:
        kws = extract_keywords(sent)
        for i in range(len(kws)):
            for j in range(i+1, len(kws)):
                G.add_edge(kws[i], kws[j])

    # C) detect concept communities
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(G))
    concepts = [list(c) for c in comms]

    # D) sentence→concept TF–IDF assignment
    cvecs = [tfidf.transform([" ".join(c)]).toarray()[0] for c in concepts]
    s2c = defaultdict(list); c2sA, c2sB = defaultdict(list), defaultdict(list)
    for i, sent in enumerate(S):
        v = tfidf.transform([sent]).toarray()[0]
        for j, cv in enumerate(cvecs):
            if cosine_similarity([v],[cv])[0][0] > 0.2:
                s2c[i].append(j)
                if i < len(s1): c2sA[j].append(sent)
                else:          c2sB[j].append(sent)
        if not s2c[i]: s2c[i] = [-1]

    # E) build undirected JCIG: connect every concept pair with cosine(weight)>0.3
    #    weight = cosine between their TF–IDF node vectors
    node_feats = []
    for j in range(len(concepts)):
        txt = " ".join(c2sA[j] + c2sB[j])
        vec = tfidf.transform([txt]).toarray()[0]
        node_feats.append(vec)
    simM = cosine_similarity(node_feats)
    edges, ew = [], []
    for i in range(len(simM)):
        for j in range(i+1, len(simM)):
            if simM[i][j] > 0.3:
                edges += [[i,j],[j,i]]
                ew    += [simM[i][j]]*2

    if not edges: return None

    # F) build SBERT‐Siamese match vectors per concept
    match_vecs = []
    for j in range(len(concepts)):
        a = sbert.encode(c2sA[j] or [""])[0]
        b = sbert.encode(c2sB[j] or [""])[0]
        diff = np.abs(a-b)
        mul  = a*b
        match_vecs.append(np.concatenate([diff,mul]))

    # G) construct PyG Data
    ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
    ew = torch.tensor(ew, dtype=torch.float32)
    x  = torch.tensor(match_vecs, dtype=torch.float32)
    y  = torch.tensor([float(label)], dtype=torch.float32)

    return Data(x=x, edge_index=ei, edge_attr=ew, y=y)

# === 4. build baseline dataset ===
graphs=[]
for _,r in tqdm(pairs_df.iterrows(), total=len(pairs_df)):
    label = r.label
    g = build_jcig(documents[r.doc1_id], documents[r.doc2_id])
    if g: graphs.append(g)

# === 5. 60-20-20 split ===
train_val, test = train_test_split(graphs, test_size=0.2, random_state=42)
train, val      = train_test_split(train_val, test_size=0.25, random_state=42)
train_loader = DataLoader(train, batch_size=4, shuffle=True)
val_loader   = DataLoader(val,   batch_size=4)
test_loader  = DataLoader(test,  batch_size=4)

# === 6. GCN + MLP ===
class JCIGBaseline(torch.nn.Module):
    def __init__(self,d,h=64):
        super().__init__()
        self.c1 = GCNConv(d,h)
        self.c2 = GCNConv(h,h)
        self.m = torch.nn.Sequential(torch.nn.Linear(h,32),
                                     torch.nn.ReLU(),
                                     torch.nn.Linear(32,1))
    def forward(self,x,ei,ew,b):
        x = F.relu(self.c1(x,ei,ew))
        x = F.relu(self.c2(x,ei,ew))
        x = global_mean_pool(x,b)
        return self.m(x)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = JCIGBaseline(train[0].x.shape[1]).to(device)
opt   = torch.optim.Adam(model.parameters(), lr=0.005)
lossf = torch.nn.BCEWithLogitsLoss()

# === 7. train & eval ===
for epoch in range(1,51):
    model.train()
    tloss=0
    for batch in train_loader:
        batch = batch.to(device)
        out   = model(batch.x,batch.edge_index,batch.edge_attr,batch.batch).view(-1)
        loss  = lossf(out,batch.y)
        loss.backward(); opt.step(); opt.zero_grad()
        tloss+= loss.item()
    # val
    model.eval()
    y_t,y_p=[],[]
    with torch.no_grad():
        for b in val_loader:
            b=b.to(device)
            p = torch.sigmoid(model(b.x,b.edge_index,b.edge_attr,b.batch).view(-1))>0.5
            y_t+=b.y.cpu().tolist(); y_p+=p.cpu().tolist()
    acc = accuracy_score(y_t,y_p)
    print(f"Epoch {epoch:2d} loss {tloss:.4f}  val_acc {acc:.3f}")

# final test
model.eval()
y_t,y_p = [],[]
with torch.no_grad():
    for b in test_loader:
        b=b.to(device)
        p=torch.sigmoid(model(b.x,b.edge_index,b.edge_attr,b.batch).view(-1))>0.5
        y_t+=b.y.cpu().tolist(); y_p+=p.cpu().tolist()
print("✅ JCIG baseline test acc:", accuracy_score(y_t,y_p))
