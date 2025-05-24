# jcig_baseline_kfold.py

import json
import pandas as pd
import spacy
import pytextrank
import networkx as nx
import numpy as np
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from transformers import AutoTokenizer, AutoModel

# ─────────────────── 1. LOAD & SETUP ───────────────────
with open("Appliance.json","r") as f:
    documents = [json.loads(line) for line in f if line.strip()]
pairs = pd.read_csv("labeled_pairs_rule_based.csv")

# SpaCy + TextRank
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract_steps(doc):
    out=[]
    for st in doc.get("Steps",[]):
        if "Text_raw" in st: out.append(st["Text_raw"].strip())
        elif "Lines" in st:
            txt=" ".join(l.get("Text","") for l in st["Lines"] if "Text" in l).strip()
            if txt: out.append(txt)
    return [s for s in out if len(s.split())>2]

def extract_keywords(sent,k=15):
    return [p.text.lower().strip() for p in nlp(sent)._.phrases[:k]]

# Global TF–IDF
all_sents = [s for d in documents for s in extract_steps(d)]
tfidf = TfidfVectorizer().fit(all_sents)

# SBERT wrapper
class SBERT:
    def __init__(self,m="sentence-transformers/all-MiniLM-L6-v2"):
        self.tok=AutoTokenizer.from_pretrained(m)
        self.mod=AutoModel.from_pretrained(m)
    def encode(self,ss):
        if not ss: return np.zeros((1,384))
        ip=self.tok(ss,padding=True,truncation=True,return_tensors="pt")
        with torch.no_grad(): out=self.mod(**ip).last_hidden_state
        att=ip["attention_mask"].unsqueeze(-1).expand(out.size()).float()
        vec=(out*att).sum(1)/torch.clamp(att.sum(1),min=1e-9)
        return vec.cpu().numpy()
sbert=SBERT()

# ─────────────────── 2. JCIG CONSTRUCTION ───────────────────
def keyword_graph(sents):
    G=nx.Graph()
    for s in sents:
        kws=extract_keywords(s)
        for i in range(len(kws)):
            for j in range(i+1,len(kws)):
                G.add_edge(kws[i],kws[j])
    return G

def detect_concepts(KG):
    from networkx.algorithms.community import greedy_modularity_communities
    return [list(c) for c in greedy_modularity_communities(KG)]

def sent2concept(sents,concepts,th=0.2):
    cvecs=[tfidf.transform([" ".join(c)]).toarray()[0] for c in concepts]
    s2c=defaultdict(list)
    c2sA,c2sB=defaultdict(list),defaultdict(list)
    half=len(sents)//2
    for i,s in enumerate(sents):
        sv=tfidf.transform([s]).toarray()[0]
        for cid,cv in enumerate(cvecs):
            if cosine_similarity([sv],[cv])[0][0]>=th:
                s2c[i].append(cid)
                (c2sA if i<half else c2sB)[cid].append(s)
        if not s2c[i]:
            s2c[i]=[-1]
    return s2c,c2sA,c2sB

def build_match_vectors(C,c2sA,c2sB):
    M=[]
    for cid in range(C):
        vA=sbert.encode(c2sA.get(cid,[])).mean(0)
        vB=sbert.encode(c2sB.get(cid,[])).mean(0)
        M.append(np.concatenate([np.abs(vA-vB),vA*vB]))
    return np.stack(M)

def build_jcig(sents,concepts,s2c,wt=0.2):
    c2s=defaultdict(list)
    for i,cs in s2c.items():
        for cid in cs:
            if cid>=0: c2s[cid].append(sents[i])
    cvecs=[tfidf.transform([" ".join(c2s[cid])]).toarray()[0] for cid in range(len(concepts))]
    G=nx.Graph()
    C=len(concepts)
    for i in range(C):
        for j in range(i+1,C):
            sim=cosine_similarity([cvecs[i]],[cvecs[j]])[0][0]
            if sim>=wt:
                G.add_edge(i,j,weight=sim)
    return G

def pyg_from_undirected(G,feat,label):
    if G.number_of_edges()==0: return None
    ei=torch.tensor(list(G.edges())).t().long()
    ew=torch.tensor([G[u][v]['weight'] for u,v in G.edges()],dtype=torch.float32)
    x=torch.tensor(feat,dtype=torch.float32)
    y=torch.tensor([label],dtype=torch.float32)
    return Data(x=x,edge_index=ei,edge_attr=ew,y=y)

# Build all graphs + labels
graphs,labels=[],[]
for _,r in tqdm(pairs.iterrows(),total=len(pairs)):
    sA=extract_steps(documents[r.doc1_id])
    sB=extract_steps(documents[r.doc2_id])
    sents=sA+sB
    if len(sents)<2: continue

    KG=keyword_graph(sents)
    Cs=detect_concepts(KG)
    s2c,c2sA,c2sB=sent2concept(sents,Cs)
    feat=build_match_vectors(len(Cs),c2sA,c2sB)
    Gj=build_jcig(sents,Cs,s2c)
    g=pyg_from_undirected(Gj,feat,int(r.label))
    if g: 
        graphs.append(g)
        labels.append(int(r.label))

print(f"Built {len(graphs)} undirected JCIG graphs")

# ─────────────────── 3. 5-FOLD STRATIFIED CV ───────────────────
kf = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
fold_acc=[]

for fold,(train_idx,test_idx) in enumerate(kf.split(graphs,labels),1):
    train_graphs=[graphs[i] for i in train_idx]
    train_labels=[labels[i] for i in train_idx]
    test_graphs =[graphs[i] for i in test_idx]
    test_labels =[labels[i] for i in test_idx]

    # carve off 20% val from train
    val_n = int(0.2*len(train_graphs))
    val_graphs=train_graphs[:val_n];  val_labels=train_labels[:val_n]
    train_graphs=train_graphs[val_n:]; train_labels=train_labels[val_n:]

    train_loader=DataLoader(train_graphs,batch_size=8,shuffle=True)
    val_loader  =DataLoader(val_graphs,  batch_size=8)
    test_loader =DataLoader(test_graphs, batch_size=8)

    # model definition
    class JCIGClassifier(torch.nn.Module):
        def __init__(self,dim,hd=64):
            super().__init__()
            self.g1=GCNConv(dim,hd)
            self.g2=GCNConv(hd,hd)
            self.mlp=torch.nn.Sequential(
                torch.nn.Linear(hd,32),
                torch.nn.ReLU(),
                torch.nn.Linear(32,1)
            )
        def forward(self,x,ei,ew,b):
            x=F.relu(self.g1(x,ei,ew))
            x=F.relu(self.g2(x,ei,ew))
            x=global_mean_pool(x,b)
            return self.mlp(x).view(-1)

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=JCIGClassifier(train_graphs[0].x.shape[1]).to(device)
    opt=torch.optim.Adam(model.parameters(),lr=5e-3)
    lossf=torch.nn.BCEWithLogitsLoss()

    best_val=0.0
    # train 30 epochs
    for ep in range(1,31):
        model.train()
        for batch in train_loader:
            batch=batch.to(device)
            pred=model(batch.x,batch.edge_index,batch.edge_attr,batch.batch)
            loss=lossf(pred,batch.y)
            loss.backward();opt.step();opt.zero_grad()

        # val
        model.eval()
        y_t,y_p=[],[]
        with torch.no_grad():
            for batch in val_loader:
                batch=batch.to(device)
                out=torch.sigmoid(model(batch.x,batch.edge_index,batch.edge_attr,batch.batch))
                y_t.extend(batch.y.cpu().numpy())
                y_p.extend((out>0.5).cpu().numpy())
        best_val=max(best_val,accuracy_score(y_t,y_p))

    # test
    model.eval()
    y_t,y_p=[],[]
    with torch.no_grad():
        for batch in test_loader:
            batch=batch.to(device)
            out=torch.sigmoid(model(batch.x,batch.edge_index,batch.edge_attr,batch.batch))
            y_t.extend(batch.y.cpu().numpy())
            y_p.extend((out>0.5).cpu().numpy())
    acc=accuracy_score(y_t,y_p)
    print(f"Fold {fold}: val_acc={best_val:.3f}  test_acc={acc:.3f}")
    fold_acc.append(acc)

print(f"\n→ 5-fold mean test acc: {np.mean(fold_acc):.3f} ± {np.std(fold_acc):.3f}")
